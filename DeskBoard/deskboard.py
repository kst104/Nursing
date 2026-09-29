"""DeskBoard - 모니터1(주 모니터) 우측 상단에 붙는 갈색 파일 게시판 위젯.

- 파일 최대 8개(4열 x 2행)를 게시할 수 있습니다.
- 추가: [+ 파일] 버튼, 또는 탐색기에서 끌어다 놓기(tkinterdnd2 설치 시).
- 더블클릭: 파일 열기 / 우클릭: 열기·폴더에서 보기·게시판에서 떼기
- 상단 제목줄을 끌면 위치 이동, 우클릭하면 메뉴(항상 위, 위치 초기화, 종료).
- 게시한 목록·위치는 사용자 폴더에 저장되어 다시 켜도 유지됩니다.
- 윈도우 로그인 시 자동 실행되도록 스스로 등록합니다(제목줄 우클릭 메뉴에서 끌 수 있음).
"""

import json
import os
import socket
import subprocess
import sys
import tkinter as tk
from tkinter import filedialog, messagebox

try:  # 선택: 탐색기에서 드래그 앤 드롭 (pip install tkinterdnd2)
    from tkinterdnd2 import DND_FILES, TkinterDnD
except ImportError:
    TkinterDnD = None

# --- 크기 / 색상 ---
COLS, ROWS = 4, 2
MAX_FILES = COLS * ROWS
TILE_W, TILE_H = 96, 92
GAP = 10
MARGIN = 16  # 화면 가장자리와의 간격

FRAME = "#4E2E14"      # 바깥 나무 테두리
BOARD = "#8B5A2B"      # 갈색 보드판
BOARD_DOT = "#7A4E24"  # 보드 질감 점
HEADER_FG = "#F5E6CC"
PAPER = "#FFF8E7"
PAPER_MISSING = "#D9CFC0"
SLOT = "#9C6A38"
PIN = "#C0392B"

BADGE_COLORS = {
    "pdf": "#D64541", "doc": "#2B579A", "docx": "#2B579A", "hwp": "#1E88E5",
    "hwpx": "#1E88E5", "xls": "#217346", "xlsx": "#217346", "csv": "#217346",
    "ppt": "#D24726", "pptx": "#D24726", "txt": "#607D8B", "png": "#8E44AD",
    "jpg": "#8E44AD", "jpeg": "#8E44AD", "zip": "#795548", "lnk": "#455A64",
}

if sys.platform == "win32":
    DATA_DIR = os.path.join(os.environ.get("APPDATA", os.path.expanduser("~")), "DeskBoard")
else:
    DATA_DIR = os.path.join(os.path.expanduser("~"), ".deskboard")
DATA_FILE = os.path.join(DATA_DIR, "board.json")

RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
RUN_NAME = "DeskBoard"


def autostart_command():
    """콘솔 창 없이 이 스크립트를 실행하는 명령 (pythonw 우선)."""
    exe = sys.executable
    pyw = os.path.join(os.path.dirname(exe), "pythonw.exe")
    if os.path.exists(pyw):
        exe = pyw
    return f'"{exe}" "{os.path.abspath(__file__)}"'


def set_autostart(enabled):
    """윈도우 로그인 시 자동 실행 등록/해제 (HKCU Run, 관리자 권한 불필요)."""
    if sys.platform != "win32":
        return False
    import winreg
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_SET_VALUE) as key:
            if enabled:
                winreg.SetValueEx(key, RUN_NAME, 0, winreg.REG_SZ, autostart_command())
            else:
                try:
                    winreg.DeleteValue(key, RUN_NAME)
                except FileNotFoundError:
                    pass
        return True
    except OSError:
        return False


def primary_work_area(root):
    """모니터1(주 모니터)의 작업 영역 (left, top, right, bottom). 작업표시줄 제외."""
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes
            rect = wintypes.RECT()
            SPI_GETWORKAREA = 0x0030
            if ctypes.windll.user32.SystemParametersInfoW(SPI_GETWORKAREA, 0, ctypes.byref(rect), 0):
                return rect.left, rect.top, rect.right, rect.bottom
        except Exception:
            pass
    return 0, 0, root.winfo_screenwidth(), root.winfo_screenheight()


def open_path(path):
    if sys.platform == "win32":
        os.startfile(path)
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


def reveal_path(path):
    if sys.platform == "win32":
        subprocess.Popen(["explorer", "/select,", os.path.normpath(path)])
    else:
        open_path(os.path.dirname(path))


class DeskBoard:
    def __init__(self):
        self.root = TkinterDnD.Tk() if TkinterDnD else tk.Tk()
        self.root.title("DeskBoard")
        self.root.overrideredirect(True)  # 테두리 없는 위젯 형태
        self.root.configure(bg=FRAME)

        self.state = self.load()
        self.files = self.state.get("files", [])[:MAX_FILES]
        self.topmost = self.state.get("topmost", False)
        self.root.attributes("-topmost", self.topmost)
        # 기본값: 자동 실행 켜짐. 실행할 때마다 현재 경로로 다시 등록해 폴더를 옮겨도 유지됨.
        self.autostart = self.state.get("autostart", True)
        set_autostart(self.autostart)

        self.board_w = COLS * TILE_W + (COLS + 1) * GAP
        self.board_h = ROWS * TILE_H + (ROWS + 1) * GAP
        self.header_h = 34

        self.build()
        self.place_window(self.state.get("pos"))
        self.render()

    # ---------- 저장 ----------
    def load(self):
        try:
            with open(DATA_FILE, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return {}

    def save(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        data = {
            "files": self.files,
            "topmost": self.topmost,
            "autostart": self.autostart,
            "pos": self.state.get("pos"),
        }
        # 임시 파일에 쓴 뒤 교체: 저장 중 전원이 꺼져도 기존 목록이 깨지지 않음
        tmp = DATA_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, DATA_FILE)

    # ---------- 화면 구성 ----------
    def build(self):
        outer = tk.Frame(self.root, bg=FRAME, padx=8, pady=6)
        outer.pack()

        header = tk.Frame(outer, bg=FRAME, height=self.header_h)
        header.pack(fill="x")
        self.title_lbl = tk.Label(header, text="📌 게시판", bg=FRAME, fg=HEADER_FG,
                                  font=("맑은 고딕", 11, "bold"))
        self.title_lbl.pack(side="left", padx=(2, 0))
        self.count_lbl = tk.Label(header, bg=FRAME, fg=HEADER_FG, font=("맑은 고딕", 9))
        self.count_lbl.pack(side="left", padx=6)

        btn_opts = dict(bg=FRAME, fg=HEADER_FG, activebackground=BOARD, activeforeground="white",
                        bd=0, font=("맑은 고딕", 10, "bold"), cursor="hand2")
        tk.Button(header, text="✕", command=self.quit, **btn_opts).pack(side="right", padx=2)
        tk.Button(header, text="+ 파일", command=self.pick_files, **btn_opts).pack(side="right", padx=6)

        for w in (header, self.title_lbl, self.count_lbl):
            w.bind("<ButtonPress-1>", self.start_move)
            w.bind("<B1-Motion>", self.on_move)
            w.bind("<ButtonRelease-1>", self.end_move)
            w.bind("<Button-3>", self.header_menu)

        self.canvas = tk.Canvas(outer, width=self.board_w, height=self.board_h,
                                bg=BOARD, highlightthickness=0)
        self.canvas.pack(pady=(4, 2))

        if TkinterDnD:
            self.canvas.drop_target_register(DND_FILES)
            self.canvas.dnd_bind("<<Drop>>", self.on_drop)

    def place_window(self, pos=None):
        self.root.update_idletasks()
        w, h = self.root.winfo_reqwidth(), self.root.winfo_reqheight()
        left, top, right, bottom = primary_work_area(self.root)
        if pos:
            x, y = pos
        else:  # 기본: 모니터1 우측 상단
            x, y = right - w - MARGIN, top + MARGIN
        self.root.geometry(f"{w}x{h}+{x}+{y}")

    def slot_xy(self, i):
        c, r = i % COLS, i // COLS
        x = GAP + c * (TILE_W + GAP)
        y = GAP + r * (TILE_H + GAP)
        return x, y

    def render(self):
        cv = self.canvas
        cv.delete("all")
        # 코르크 질감 점
        for yy in range(4, self.board_h, 11):
            for xx in range(4 + (yy // 11) % 2 * 5, self.board_w, 11):
                cv.create_oval(xx, yy, xx + 2, yy + 2, fill=BOARD_DOT, outline="")

        for i in range(MAX_FILES):
            x, y = self.slot_xy(i)
            if i < len(self.files):
                self.draw_tile(i, x, y, self.files[i])
            else:
                cv.create_rectangle(x, y, x + TILE_W, y + TILE_H, outline=SLOT, dash=(4, 3), width=2)
        if not self.files:
            hint = "파일을 끌어다 놓거나\n[+ 파일]을 눌러 게시하세요" if TkinterDnD else "[+ 파일]을 눌러 게시하세요"
            cv.create_text(self.board_w / 2, self.board_h / 2, text=hint, fill=HEADER_FG,
                           font=("맑은 고딕", 10), justify="center")
        self.count_lbl.config(text=f"{len(self.files)}/{MAX_FILES}")

    def draw_tile(self, i, x, y, path):
        cv = self.canvas
        tag = f"tile{i}"
        exists = os.path.exists(path)
        name = os.path.basename(path.rstrip("\\/")) or path
        is_dir = os.path.isdir(path)
        ext = "폴더" if is_dir else (os.path.splitext(name)[1][1:].lower() or "파일")

        # 그림자 + 종이
        cv.create_rectangle(x + 3, y + 3, x + TILE_W + 3, y + TILE_H + 3, fill="#5C3A1A", outline="", tags=tag)
        cv.create_rectangle(x, y, x + TILE_W, y + TILE_H, fill=PAPER if exists else PAPER_MISSING,
                            outline="#E0D2B4", tags=tag)
        # 압정
        cv.create_oval(x + TILE_W / 2 - 5, y - 3, x + TILE_W / 2 + 5, y + 7, fill=PIN, outline="#7B1E14", tags=tag)
        # 확장자 배지
        color = "#C9A227" if is_dir else BADGE_COLORS.get(ext, "#6D4C41")
        cv.create_rectangle(x + 18, y + 14, x + TILE_W - 18, y + 42, fill=color, outline="", tags=tag)
        cv.create_text(x + TILE_W / 2, y + 28, text=ext.upper()[:5], fill="white",
                       font=("맑은 고딕", 10, "bold"), tags=tag)
        # 파일 이름 (2줄까지)
        label = name if len(name) <= 22 else name[:20] + "…"
        cv.create_text(x + TILE_W / 2, y + 66, text=label, width=TILE_W - 8, justify="center",
                       fill="#3E2723" if exists else "#8D6E63", font=("맑은 고딕", 8), tags=tag)
        if not exists:
            cv.create_text(x + TILE_W - 6, y + 6, text="!", fill="#B71C1C", anchor="ne",
                           font=("맑은 고딕", 9, "bold"), tags=tag)

        cv.tag_bind(tag, "<Double-Button-1>", lambda e, p=path: self.open_file(p))
        cv.tag_bind(tag, "<Button-3>", lambda e, idx=i: self.tile_menu(e, idx))
        cv.tag_bind(tag, "<Enter>", lambda e: cv.config(cursor="hand2"))
        cv.tag_bind(tag, "<Leave>", lambda e: cv.config(cursor=""))

    # ---------- 파일 동작 ----------
    def add_files(self, paths):
        added = 0
        for p in paths:
            p = os.path.normpath(p)
            if p in self.files:
                continue
            if len(self.files) >= MAX_FILES:
                messagebox.showinfo("게시판", f"게시판이 가득 찼습니다 (최대 {MAX_FILES}개).", parent=self.root)
                break
            self.files.append(p)
            added += 1
        if added:
            self.save()
            self.render()

    def pick_files(self):
        paths = filedialog.askopenfilenames(parent=self.root, title="게시할 파일 선택")
        if paths:
            self.add_files(paths)

    def on_drop(self, event):
        self.add_files(self.root.tk.splitlist(event.data))

    def open_file(self, path):
        if not os.path.exists(path):
            messagebox.showwarning("게시판", f"파일을 찾을 수 없습니다.\n{path}", parent=self.root)
            return
        try:
            open_path(path)
        except OSError as e:
            messagebox.showerror("게시판", str(e), parent=self.root)

    def remove(self, idx):
        del self.files[idx]
        self.save()
        self.render()

    def move_tile(self, idx, delta):
        j = idx + delta
        if 0 <= j < len(self.files):
            self.files[idx], self.files[j] = self.files[j], self.files[idx]
            self.save()
            self.render()

    def tile_menu(self, event, idx):
        path = self.files[idx]
        m = tk.Menu(self.root, tearoff=0)
        m.add_command(label="열기", command=lambda: self.open_file(path))
        m.add_command(label="폴더에서 보기", command=lambda: reveal_path(path))
        m.add_separator()
        m.add_command(label="◀ 앞으로", command=lambda: self.move_tile(idx, -1))
        m.add_command(label="뒤로 ▶", command=lambda: self.move_tile(idx, 1))
        m.add_separator()
        m.add_command(label="게시판에서 떼기", command=lambda: self.remove(idx))
        m.tk_popup(event.x_root, event.y_root)

    # ---------- 창 이동 / 메뉴 ----------
    def start_move(self, e):
        self._drag = (e.x_root - self.root.winfo_x(), e.y_root - self.root.winfo_y())

    def on_move(self, e):
        dx, dy = self._drag
        self.root.geometry(f"+{e.x_root - dx}+{e.y_root - dy}")

    def end_move(self, e):
        self.state["pos"] = [self.root.winfo_x(), self.root.winfo_y()]
        self.save()

    def header_menu(self, event):
        m = tk.Menu(self.root, tearoff=0)
        m.add_command(label=("✓ " if self.topmost else "   ") + "항상 위에 표시", command=self.toggle_topmost)
        if sys.platform == "win32":
            m.add_command(label=("✓ " if self.autostart else "   ") + "윈도우 시작 시 자동 실행",
                          command=self.toggle_autostart)
        m.add_command(label="우측 상단으로 위치 초기화", command=self.reset_position)
        m.add_separator()
        m.add_command(label="모두 떼기", command=self.clear_all)
        m.add_command(label="종료", command=self.quit)
        m.tk_popup(event.x_root, event.y_root)

    def toggle_topmost(self):
        self.topmost = not self.topmost
        self.root.attributes("-topmost", self.topmost)
        self.save()

    def toggle_autostart(self):
        if set_autostart(not self.autostart):
            self.autostart = not self.autostart
            self.save()
        else:
            messagebox.showerror("게시판", "자동 실행 설정을 바꾸지 못했습니다.", parent=self.root)

    def reset_position(self):
        self.state["pos"] = None
        self.place_window()
        self.save()

    def clear_all(self):
        if self.files and messagebox.askyesno("게시판", "게시된 파일을 모두 뗄까요?\n(원본 파일은 삭제되지 않습니다)", parent=self.root):
            self.files = []
            self.save()
            self.render()

    def quit(self):
        self.save()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def already_running():
    """중복 실행 방지: 로컬 포트를 선점한 인스턴스가 있으면 True."""
    global _lock_sock
    _lock_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        _lock_sock.bind(("127.0.0.1", 47823))
        return False
    except OSError:
        return True


if __name__ == "__main__":
    if not already_running():
        DeskBoard().run()
