# N-Map 로그인(OAuth/OIDC) 설정 가이드

N-Map은 임상 데이터를 다루므로 **로그인하지 않으면 앱이 열리지 않습니다**(fail-closed).
아래 절차대로 OIDC 공급자를 연결하세요. 15분이면 됩니다.

## 동작 방식

Streamlit 네이티브 인증(`st.login()` / `st.user` / `st.logout()`)을 사용합니다.
비밀번호를 앱이 직접 받지 않고, Google·Microsoft 같은 공급자가 인증한 결과만 받습니다.

```
사용자 → [로그인 버튼] → 공급자 로그인 화면 → 앱으로 복귀(/oauth2callback)
      → 서명된 세션 쿠키 발급 → 허용목록 검사 → 앱 진입
```

> **GitHub는 사용할 수 없습니다.** Streamlit 네이티브 인증은 OIDC 전용인데,
> GitHub는 일반 OAuth 2.0이라 OIDC 디스커버리 문서를 제공하지 않습니다.
> 쓸 수 있는 공급자: Google, Microsoft Entra ID, Okta, Auth0 등 OIDC 지원 공급자.

## 1단계 — 공급자에서 OAuth 클라이언트 발급

이 단계는 **본인 계정으로 직접** 하셔야 합니다.

### Google을 쓰는 경우

1. [Google Cloud Console](https://console.cloud.google.com/) → 프로젝트 생성(또는 선택)
2. **API 및 서비스 → OAuth 동의 화면** 구성
   - 조직 계정만 쓸 거면 사용자 유형 **내부**, 아니면 **외부**
3. **API 및 서비스 → 사용자 인증 정보 → 사용자 인증 정보 만들기 → OAuth 클라이언트 ID**
   - 애플리케이션 유형: **웹 애플리케이션**
   - **승인된 리디렉션 URI** 에 아래를 등록 (2단계의 `redirect_uri`와 **문자 하나까지 똑같아야** 합니다)
     - 로컬 개발: `http://localhost:8501/oauth2callback`
     - 배포본: `https://<앱주소>/oauth2callback`
4. 발급된 **클라이언트 ID**와 **클라이언트 보안 비밀번호**를 복사

### Microsoft Entra ID(구 Azure AD)를 쓰는 경우

1. Entra 관리센터 → **앱 등록 → 새 등록**
2. 리디렉션 URI: 플랫폼 **웹**, 값은 위와 동일한 `/oauth2callback` 주소
3. **인증서 및 비밀 → 새 클라이언트 비밀** 생성 후 값 복사
4. **개요**에서 애플리케이션(클라이언트) ID와 디렉터리(테넌트) ID 확인

## 2단계 — secrets.toml 작성

```bash
cd N-Map_Deploy
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
python -c "import secrets; print(secrets.token_hex(32))"   # cookie_secret 생성
```

`.streamlit/secrets.toml` 을 열어 값을 채웁니다.

```toml
[auth]
redirect_uri = "http://localhost:8501/oauth2callback"
cookie_secret = "위에서 생성한 랜덤 문자열"

[auth.google]
client_id = "발급받은 클라이언트 ID"
client_secret = "발급받은 보안 비밀번호"
server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"
```

접근 가능한 사람을 제한하려면 `[access]` 섹션을 추가합니다.
이 섹션이 없으면 **로그인에 성공한 모든 계정**이 들어올 수 있으니, 실제 운영에서는 반드시 지정하세요.

```toml
[access]
allowed_domains = ["yourhospital.or.kr"]      # 병원 도메인 계정 전체 허용
allowed_emails = ["nurse.kim@gmail.com"]      # 개별 계정 허용
```

> `.streamlit/secrets.toml` 은 `.gitignore` 에 등록되어 있어 커밋되지 않습니다.
> 이 파일은 절대 저장소·메신저·이메일로 공유하지 마세요.

## 3단계 — 실행

```bash
pip install -r requirements.txt
streamlit run app.py
```

로그인 화면이 뜨고, 인가된 계정으로 들어가면 사이드바 상단에 이름과 **로그아웃** 버튼이 보입니다.

## 배포 시 (Streamlit Community Cloud)

1. 앱 대시보드 → **Settings → Secrets** 에 `secrets.toml` 내용을 그대로 붙여넣기
2. `redirect_uri` 를 배포 주소로 변경: `https://<앱이름>.streamlit.app/oauth2callback`
3. 공급자 콘솔의 승인된 리디렉션 URI에도 **같은 주소를 추가** (로컬 주소와 별개로 둘 다 등록 가능)

## 문제 해결

| 증상 | 원인과 해결 |
|---|---|
| `redirect_uri_mismatch` | 공급자에 등록한 URI와 `secrets.toml` 의 `redirect_uri` 가 다릅니다. `http`/`https`, 포트, 끝의 `/oauth2callback` 까지 정확히 일치시키세요. |
| "로그인 설정 필요" 화면 | `[auth]` 섹션이 없거나 `client_id` 가 비어 있습니다. |
| "접근 권한 없음" 화면 | 로그인은 됐지만 `[access]` 허용목록에 없는 계정입니다. 목록에 추가하세요. |
| `Authlib` 관련 오류 | `pip install -r requirements.txt` 로 `Authlib>=1.3.2` 를 설치하세요. |
| 로그인 후 계속 로그인 화면 | `cookie_secret` 이 비었거나 매 실행마다 바뀌는 값입니다. 고정된 랜덤 문자열을 쓰세요. |

## 개발 중 인증 끄기

```bash
NMAP_ALLOW_ANONYMOUS=1 streamlit run app.py
```

경고 배너가 뜨고 로그인 없이 열립니다. **실제 환자 데이터로는 절대 사용하지 마세요.**
`[auth]` 가 설정되어 있으면 이 환경변수가 있어도 인증을 건너뛰지 않습니다.
