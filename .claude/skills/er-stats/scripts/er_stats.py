#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
응급실 통계 집계기 (HSSM '부서별 대장 조회 > 통계 > 응급실-통계' 연동)

두 가지 모드로 동작한다.

  raw     : 원자료(내원/처방 원장 1행=1건)를 읽어 항목별 통계를 직접 집계한다.
  summary : HSSM 각 통계 항목을 '파일' 버튼으로 내보낸 집계표들을 한 리포트로 병합한다.
            (HSSM이 이미 집계해 준 표를 그대로 싣는다 - 수치를 재계산하지 않는다)

사용 예:
    python er_stats.py raw 응급실원장.xlsx --start 2026-08-01 --end 2026-08-29 --out reports/
    python er_stats.py summary exports/2026-08/ --start 2026-08-01 --end 2026-08-29
    python er_stats.py raw 원장.csv --period monthly     # 지난달 전체
    python er_stats.py raw 원장.csv --period daily       # 어제 하루
"""

import argparse
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

# --- 컬럼 별칭 -------------------------------------------------------------
# HSSM 내보내기 헤더가 화면·버전마다 달라서 흔한 이름을 모두 매핑한다.
ALIASES = {
    "내원일시": ["내원일시", "내원일자", "내원일", "내원시각", "접수일시", "접수일자", "접수일",
                "방문일자", "방문일시", "처방일자", "처방일시", "오더일시", "등록일시", "진료일자",
                "visit_date", "visit_datetime", "order_date"],
    "퇴실일시": ["퇴실일시", "퇴실일자", "퇴실시각", "퇴원일시", "종료일시", "귀가일시",
                "discharge_datetime"],
    "내원번호": ["내원번호", "접수번호", "방문번호", "응급번호", "visit_id", "encounter_id"],
    "환자ID": ["환자번호", "환자ID", "등록번호", "차트번호", "patient_id", "chart_no"],
    "성별": ["성별", "sex", "gender"],
    "연령": ["연령", "나이", "age"],
    # HSSM 응급실-통계 메뉴 항목에 대응
    "최초분류": ["최초분류", "최초KTAS", "초기분류", "최초중증도", "최초응급도"],
    "수정분류": ["수정분류", "최종분류", "수정KTAS", "최종KTAS", "수정중증도"],
    "KTAS": ["KTAS", "ktas", "KTAS등급", "중증도", "중증도분류", "응급도", "진료분류", "triage"],
    "내원수단": ["내원수단", "내원방법", "이송수단", "내원차량", "arrival_mode"],
    "내원경로": ["내원경로", "유입경로", "내원구분"],
    "내원사유": ["내원사유", "내원이유", "방문사유", "발병사유"],
    "의도성": ["의도성", "의도성여부", "손상의도", "자·타해여부", "intent"],
    "119지역": ["119지역", "119센터", "구급대지역", "관할지역", "출동지역"],
    "주진료과": ["주진료과", "진료과", "담당과", "전문과목", "과", "department", "dept"],
    "진료결과": ["진료결과", "퇴실결과", "퇴실구분", "퇴실유형", "내원결과", "전원여부",
                "귀가여부", "disposition", "outcome"],
    "주호소": ["주호소", "주증상", "주진단", "진단명", "상병명", "상병", "chief_complaint",
              "cc", "diagnosis"],
    "처방항목": ["처방명", "처방항목", "오더명", "처방코드명", "약품명", "검사명", "수가명",
                "order_name", "item_name"],
    "처방구분": ["처방구분", "오더구분", "처방유형", "처방분류", "오더유형", "분류", "order_type"],
}

# 리포트 '항목' 순서 - HSSM 응급실-통계 트리 순서를 따른다.
CATEGORY_ITEMS = [
    ("최초분류(KTAS)별", "최초분류"),
    ("수정분류(KTAS)별", "수정분류"),
    ("진료분류(KTAS)별", "KTAS"),
    ("내원수단별", "내원수단"),
    ("내원경로별", "내원경로"),
    ("내원사유별", "내원사유"),
    ("의도성별", "의도성"),
    ("119지역별", "119지역"),
    ("주진료과별", "주진료과"),
    ("진료결과별", "진료결과"),
    ("주호소·진단별", "주호소"),
    ("처방구분별", "처방구분"),
    ("처방항목별", "처방항목"),
    ("성별", "성별"),
]
TOP_N_ITEMS = {"주호소", "처방항목", "119지역"}  # 값 종류가 많아 상위 N개만 싣는 항목

AGE_BINS = [0, 1, 10, 20, 30, 40, 50, 60, 70, 80, 200]
AGE_LABELS = ["0세", "1-9세", "10대", "20대", "30대", "40대", "50대", "60대", "70대", "80세 이상"]
WEEKDAY_KO = ["월", "화", "수", "목", "금", "토", "일"]
DATA_SUFFIXES = (".csv", ".tsv", ".txt", ".xlsx", ".xls", ".xlsm")


# --- 입출력 ----------------------------------------------------------------
def read_any(path):
    """CSV/TSV/XLS(X)를 인코딩·구분자 자동 판별로 읽어 DataFrame 리스트로 반환."""
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xls", ".xlsm"):
        return list(pd.read_excel(path, sheet_name=None).values())
    sep = "\t" if path.suffix.lower() in (".tsv", ".txt") else None
    for enc in ("utf-8-sig", "cp949", "euc-kr", "utf-8"):
        try:
            return [pd.read_csv(path, encoding=enc, sep=sep, engine="python")]
        except UnicodeDecodeError:
            continue
    raise ValueError(f"인코딩을 판별하지 못했습니다: {path}")


def collect_files(inputs):
    """파일/디렉터리 경로 목록을 데이터 파일 경로 목록으로 펼친다."""
    files = []
    for item in inputs:
        p = Path(item)
        if not p.exists():
            raise FileNotFoundError(f"경로를 찾을 수 없습니다: {p}")
        if p.is_dir():
            files.extend(sorted(f for f in p.rglob("*")
                                if f.is_file() and f.suffix.lower() in DATA_SUFFIXES
                                and not f.name.startswith("~$")))
        else:
            files.append(p)
    if not files:
        raise ValueError("읽을 데이터 파일이 없습니다.")
    return files


def resolve_columns(df):
    """실제 헤더를 표준 컬럼키로 매핑한다. {표준키: 실제컬럼명}"""
    lowered = {}
    for c in df.columns:
        key = str(c).strip().lower().replace(" ", "").replace("_", "")
        lowered.setdefault(key, c)
    found = {}
    for key, names in ALIASES.items():
        for name in names:
            probe = name.lower().replace(" ", "").replace("_", "")
            if probe in lowered:
                found[key] = lowered[probe]
                break
    return found


def resolve_period(args):
    """--period 프리셋 또는 --start/--end 를 (시작일, 종료일)로 환산한다."""
    if args.start or args.end:
        start = pd.to_datetime(args.start).date() if args.start else date.min
        end = pd.to_datetime(args.end).date() if args.end else date.today()
        return start, end
    today = date.today()
    if args.period == "daily":                      # 어제 하루 (저녁 정기 집계용)
        d = today - timedelta(days=1)
        return d, d
    if args.period == "weekly":                     # 최근 7일
        end = today - timedelta(days=1)
        return end - timedelta(days=6), end
    if args.period == "monthly":                    # 지난달 전체 (월간 정기 집계용)
        end = today.replace(day=1) - timedelta(days=1)
        return end.replace(day=1), end
    if args.period == "mtd":                        # 이번 달 누계
        return today.replace(day=1), today
    return date.min, today


# --- 표 유틸 ---------------------------------------------------------------
def value_counts_table(series, top=None):
    """빈도 + 구성비 표를 만든다."""
    s = series.dropna().astype(str).str.strip()
    s = s[(s != "") & (~s.str.lower().isin(["nan", "none", "null"]))]
    if s.empty:
        return None
    total = len(s)
    vc = s.value_counts()
    trimmed = vc.head(top) if top else vc
    out = trimmed.rename_axis("항목").reset_index(name="건수")
    out["구성비(%)"] = (out["건수"] / total * 100).round(1)
    if top and len(vc) > top:
        rest = int(vc.iloc[top:].sum())
        out.loc[len(out)] = [f"기타({len(vc) - top}종)", rest, round(rest / total * 100, 1)]
    out.loc[len(out)] = ["합계", total, 100.0]
    return out


def md_table(df):
    """DataFrame을 GitHub Markdown 표로 변환한다."""
    if df is None or len(df) == 0:
        return "_해당 항목의 데이터가 없습니다._\n"
    header = [str(h) for h in df.columns]
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    for _, row in df.iterrows():
        cells = ["" if pd.isna(v) else str(v) for v in row.tolist()]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


# --- raw 모드 --------------------------------------------------------------
def build_raw_report(df, cols, start, end, top_n, title):
    """원자료에서 항목별 통계를 계산해 (markdown, {섹션명: DataFrame}) 반환."""
    if "내원일시" not in cols:
        raise ValueError(
            "날짜 컬럼을 찾지 못했습니다. 내원일시/내원일자/처방일자 중 하나가 필요합니다.\n"
            f"현재 컬럼: {', '.join(str(c) for c in df.columns)}"
        )

    d = df.copy()
    d["_dt"] = pd.to_datetime(d[cols["내원일시"]], errors="coerce")
    dropped = int(d["_dt"].isna().sum())
    d = d.dropna(subset=["_dt"])
    d = d[(d["_dt"].dt.date >= start) & (d["_dt"].dt.date <= end)]

    days = (end - start).days + 1
    tables = {}
    parts = [f"# {title}", "",
             f"- **집계 기간**: {start:%Y-%m-%d} ~ {end:%Y-%m-%d} ({days}일)",
             f"- **생성 일시**: {datetime.now():%Y-%m-%d %H:%M}",
             "- **자료 출처**: HSSM 부서별 대장 조회 > 통계 > 응급실-통계 (원자료 재집계)"]

    if d.empty:
        parts += ["", "> 해당 기간에 해당하는 데이터가 없습니다."]
        return "\n".join(parts), tables

    # 총계
    rows = len(d)
    if "내원번호" in cols:
        visits, basis = int(d[cols["내원번호"]].nunique()), "내원번호 고유값"
    elif "환자ID" in cols:
        visits = int(d.groupby([d[cols["환자ID"]], d["_dt"].dt.date]).ngroups)
        basis = "환자ID + 일자 조합"
    else:
        visits, basis = rows, "행 수"

    parts += [f"- **총 레코드 수**: {rows:,}건",
              f"- **총 내원 건수**: {visits:,}건 (기준: {basis})",
              f"- **일평균 내원**: {visits / days:.1f}건/일"]
    if "환자ID" in cols:
        parts.append(f"- **고유 환자 수**: {d[cols['환자ID']].nunique():,}명")
    if dropped:
        parts.append(f"- ⚠️ 날짜 파싱 실패로 제외된 행: {dropped:,}건")
    parts.append("")

    # 1. 일자별
    daily = d.groupby(d["_dt"].dt.date).size().rename("건수").rename_axis("일자").reset_index()
    daily = pd.DataFrame({"일자": pd.date_range(start, end).date}).merge(daily, on="일자", how="left")
    daily["건수"] = daily["건수"].fillna(0).astype(int)
    daily["요일"] = [WEEKDAY_KO[pd.Timestamp(x).weekday()] for x in daily["일자"]]
    daily = daily[["일자", "요일", "건수"]]
    tables["01_일자별"] = daily
    peak = daily.loc[daily["건수"].idxmax()]
    parts += ["## 1. 일자별 내원 추이", "",
              f"최다 일자 **{peak['일자']}({peak['요일']}) {int(peak['건수']):,}건** · "
              f"최소 {int(daily['건수'].min()):,}건 · 일평균 {daily['건수'].mean():.1f}건", "",
              md_table(daily)]

    # 2. 요일별
    wd_counts = d.groupby(d["_dt"].dt.weekday).size()
    wd = pd.DataFrame({"요일": [WEEKDAY_KO[i] for i in wd_counts.index], "건수": wd_counts.values})
    wd["구성비(%)"] = (wd["건수"] / wd["건수"].sum() * 100).round(1)
    tables["02_요일별"] = wd
    parts += ["## 2. 요일별", "", md_table(wd)]

    n = 3
    # 3. 시간대별 / 주야간별 (HSSM '월별내원시간대', '응급실현황-주야간별현황' 대응)
    if d["_dt"].dt.hour.nunique() > 1:
        hr = d.groupby(d["_dt"].dt.hour).size().rename_axis("시각").reset_index(name="건수")
        hr["시간대"] = hr["시각"].apply(lambda h: f"{h:02d}:00-{h:02d}:59")
        hr["구성비(%)"] = (hr["건수"] / hr["건수"].sum() * 100).round(1)
        tables[f"{n:02d}_시간대별"] = hr[["시간대", "건수", "구성비(%)"]]
        parts += [f"## {n}. 시간대별 내원", "", md_table(hr[["시간대", "건수", "구성비(%)"]])]
        n += 1

        night = d["_dt"].dt.hour.apply(lambda h: "주간(08-17시)" if 8 <= h < 18 else
                                       ("야간(18-23시)" if h >= 18 else "심야(00-07시)"))
        dn = value_counts_table(night)
        tables[f"{n:02d}_주야간별"] = dn
        parts += [f"## {n}. 주야간별", "", md_table(dn)]
        n += 1

    # 연령대별
    if "연령" in cols:
        ages = pd.to_numeric(d[cols["연령"]], errors="coerce")
        tbl = value_counts_table(pd.cut(ages, bins=AGE_BINS, labels=AGE_LABELS, right=False))
        if tbl is not None:
            body = tbl[tbl["항목"] != "합계"].set_index("항목").reindex(AGE_LABELS).dropna(
                subset=["건수"]).reset_index()
            body["건수"] = body["건수"].astype(int)
            body.loc[len(body)] = ["합계", int(body["건수"].sum()), 100.0]
            tables[f"{n:02d}_연령대별"] = body
            parts += [f"## {n}. 연령대별", "",
                      f"평균 {ages.mean():.1f}세 · 중앙값 {ages.median():.0f}세", "", md_table(body)]
            n += 1

    # 범주형 항목별
    for label, key in CATEGORY_ITEMS:
        if key not in cols:
            continue
        tbl = value_counts_table(d[cols[key]], top=top_n if key in TOP_N_ITEMS else None)
        if tbl is None:
            continue
        tables[f"{n:02d}_{label}"] = tbl
        head = f"## {n}. {label}" + (f" (상위 {top_n})" if key in TOP_N_ITEMS else "")
        parts += [head, "", md_table(tbl)]
        n += 1

    # 재실시간 (HSSM '응급실평균체류시간', '체류시간별 총 통계현황' 대응)
    if "퇴실일시" in cols:
        mins = (pd.to_datetime(d[cols["퇴실일시"]], errors="coerce") - d["_dt"]
                ).dt.total_seconds() / 60
        mins = mins[(mins >= 0) & (mins < 60 * 24 * 7)]
        if not mins.empty:
            stay = pd.DataFrame({
                "지표": ["평균", "중앙값", "75분위", "95분위", "최대"],
                "재실시간(분)": [round(mins.mean(), 1), round(mins.median(), 1),
                              round(mins.quantile(.75), 1), round(mins.quantile(.95), 1),
                              round(mins.max(), 1)]})
            bands = pd.cut(mins, bins=[0, 60, 120, 240, 360, 720, 10 ** 6],
                           labels=["1시간 미만", "1-2시간", "2-4시간", "4-6시간", "6-12시간", "12시간 이상"],
                           right=False)
            band_tbl = value_counts_table(bands)
            tables[f"{n:02d}_재실시간"] = stay
            tables[f"{n + 1:02d}_재실시간구간별"] = band_tbl
            parts += [f"## {n}. 응급실 재실(체류)시간", "", f"산출 대상 {len(mins):,}건", "",
                      md_table(stay), "", f"### {n}-1. 체류시간 구간별", "", md_table(band_tbl)]

    return "\n".join(parts), tables


# --- summary 모드 ----------------------------------------------------------
def build_summary_report(files, start, end, title):
    """HSSM에서 '파일'로 내보낸 항목별 집계표들을 한 리포트로 병합한다."""
    tables, parts, toc = {}, [], []
    for i, f in enumerate(files, start=1):
        name = f.stem.strip()
        try:
            frames = read_any(f)
        except Exception as exc:                                   # 깨진 파일은 건너뛰고 기록
            parts += [f"## {i}. {name}", "", f"> ⚠️ 읽기 실패: {exc}", ""]
            toc.append(f"{i}. {name} — 읽기 실패")
            continue
        for j, frame in enumerate(frames):
            frame = frame.dropna(axis=1, how="all").dropna(axis=0, how="all")
            label = name if len(frames) == 1 else f"{name} ({j + 1})"
            key = f"{i:02d}_{label}".replace("/", "-")
            tables[key] = frame
            parts += [f"## {i}. {label}", "", f"행 {len(frame):,} · 열 {len(frame.columns)}", "",
                      md_table(frame), ""]
            toc.append(f"{i}. {label} — {len(frame):,}행")

    header = [f"# {title}", "",
              f"- **집계 기간**: {start:%Y-%m-%d} ~ {end:%Y-%m-%d} ({(end - start).days + 1}일)",
              f"- **생성 일시**: {datetime.now():%Y-%m-%d %H:%M}",
              "- **자료 출처**: HSSM 부서별 대장 조회 > 통계 > 응급실-통계 (항목별 내보내기 원본)",
              f"- **병합 항목 수**: {len(files)}개", "",
              "> 아래 수치는 HSSM이 산출한 값을 그대로 옮긴 것으로, 재계산하지 않았습니다.", "",
              "## 목차", ""] + [f"- {t}" for t in toc] + [""]
    return "\n".join(header + parts), tables


# --- main ------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="응급실 통계 집계기 (HSSM 연동)")
    ap.add_argument("mode", choices=["raw", "summary"],
                    help="raw=원자료 재집계, summary=HSSM 항목별 내보내기 병합")
    ap.add_argument("input", nargs="+", help="파일 또는 디렉터리 경로")
    ap.add_argument("--start", help="집계 시작일 (YYYY-MM-DD)")
    ap.add_argument("--end", help="집계 종료일 (YYYY-MM-DD)")
    ap.add_argument("--period", default="all",
                    choices=["all", "daily", "weekly", "monthly", "mtd"],
                    help="기간 프리셋. daily=어제, weekly=최근7일, monthly=지난달, mtd=이번달 누계")
    ap.add_argument("--out", default="reports", help="리포트 출력 디렉터리 (기본 reports)")
    ap.add_argument("--top", type=int, default=20, help="값 종류가 많은 항목의 상위 N개 (기본 20)")
    ap.add_argument("--title", default="응급실 통계 리포트", help="리포트 제목")
    ap.add_argument("--no-csv", action="store_true", help="항목별 CSV 저장 생략")
    args = ap.parse_args(argv)

    files = collect_files(args.input)
    start, end = resolve_period(args)

    if args.mode == "raw":
        df = pd.concat([f for p in files for f in read_any(p)], ignore_index=True)
        cols = resolve_columns(df)
        if "내원일시" not in cols:
            print("날짜 컬럼을 인식하지 못했습니다. 현재 컬럼:", list(df.columns), file=sys.stderr)
            return 2
        if start == date.min:                       # 기간 미지정이면 자료 전체 범위
            parsed = pd.to_datetime(df[cols["내원일시"]], errors="coerce")
            if parsed.notna().any():
                start, end = parsed.min().date(), max(end, parsed.max().date())
        report, tables = build_raw_report(df, cols, start, end, args.top, args.title)
        col_note = "인식된 컬럼: " + ", ".join(f"{k}←{v}" for k, v in cols.items())
    else:
        if start == date.min:
            start = end
        report, tables = build_summary_report(files, start, end, args.title)
        col_note = "병합한 파일: " + ", ".join(f.name for f in files)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"er_stats_{start:%Y%m%d}_{end:%Y%m%d}"
    md_path = out_dir / f"{stem}.md"
    md_path.write_text(report, encoding="utf-8")

    if not args.no_csv and tables:
        csv_dir = out_dir / stem
        csv_dir.mkdir(exist_ok=True)
        for name, tbl in tables.items():
            tbl.to_csv(csv_dir / f"{name}.csv", index=False, encoding="utf-8-sig")

    print(report)
    print(f"\n---\n리포트 저장: {md_path}")
    if not args.no_csv and tables:
        print(f"항목별 CSV: {out_dir / stem}/ ({len(tables)}개)")
    print(col_note)
    return 0


if __name__ == "__main__":
    sys.exit(main())
