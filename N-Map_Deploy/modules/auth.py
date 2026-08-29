"""OIDC 로그인 게이트.

Streamlit 네이티브 인증(st.login / st.user / st.logout)을 감싸서
N-Map 앱 접근을 인가된 사용자로 제한한다.

임상 데이터를 다루는 앱이므로 **기본은 fail-closed** 이다.
`[auth]` 설정이 없으면 앱을 열지 않고 설정 안내를 띄운다.
로컬 개발 중 인증 없이 띄우려면 환경변수 NMAP_ALLOW_ANONYMOUS=1 을 명시적으로 설정한다.

설정은 .streamlit/secrets.toml 에 둔다 (secrets.toml.example 참고).
"""

import os

import streamlit as st

# 로그인 버튼에 표시할 이름. secrets.toml 의 [auth.<키>] 섹션명과 일치해야 한다.
PROVIDER_LABELS = {
    "google": "Google 계정으로 로그인",
    "microsoft": "Microsoft 계정으로 로그인",
    "okta": "Okta 계정으로 로그인",
    "auth0": "Auth0 계정으로 로그인",
}

# [auth] 아래에서 공급자 섹션이 아닌 공통 설정 키
_SHARED_AUTH_KEYS = {
    "redirect_uri", "cookie_secret", "expose_tokens",
    "client_id", "client_secret", "server_metadata_url", "client_kwargs",
}


def _secrets_section(name):
    """secrets.toml 의 섹션을 dict 로 돌려준다. 없으면 빈 dict."""
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        # secrets.toml 자체가 없는 경우 등
        pass
    return {}


def _configured_providers():
    """[auth] 아래에 정의된 공급자 키 목록.

    단일 공급자 설정([auth] 에 client_id 를 직접 둔 형태)이면 빈 리스트를 반환하며,
    이 경우 st.login() 을 인자 없이 호출한다.
    """
    auth = _secrets_section("auth")
    return [k for k in auth if k not in _SHARED_AUTH_KEYS]


def is_configured():
    """OIDC 인증이 설정되어 있는지."""
    auth = _secrets_section("auth")
    if not auth:
        return False
    # 단일 공급자 형태이거나, [auth.<provider>] 하위 섹션이 하나라도 있으면 설정된 것
    return bool(auth.get("client_id")) or bool(_configured_providers())


def _is_logged_in():
    """로그인 여부. 인증 컨텍스트가 없으면 미로그인으로 간주한다(fail-closed).

    st.user 는 인증이 초기화되지 않은 실행 환경에서 is_logged_in 속성 자체가
    없어 AttributeError 를 던진다. 그 경우 예외를 밖으로 내보내는 대신
    미로그인으로 처리해 로그인 화면을 보여준다.
    """
    try:
        return bool(st.user.is_logged_in)
    except Exception:
        return False


def _allow_anonymous():
    return os.environ.get("NMAP_ALLOW_ANONYMOUS", "").strip() in ("1", "true", "True")


def _is_allowed(user_email):
    """접근 허용 목록 검사.

    [access] 섹션이 없으면 로그인한 모든 사용자를 허용한다.
    allowed_emails / allowed_domains 중 하나라도 설정되어 있으면 그 목록으로 제한한다.
    """
    access = _secrets_section("access")
    emails = [e.strip().lower() for e in access.get("allowed_emails", []) if e.strip()]
    domains = [d.strip().lower().lstrip("@") for d in access.get("allowed_domains", []) if d.strip()]

    if not emails and not domains:
        return True
    if not user_email:
        return False

    email = user_email.strip().lower()
    if email in emails:
        return True
    return any(email.endswith("@" + d) for d in domains)


def _render_setup_help():
    st.title("🔒 N-Map 로그인 설정 필요")
    st.error("OIDC 인증이 설정되지 않아 앱을 열 수 없습니다.")
    st.markdown(
        """
        이 앱은 임상 데이터를 다루므로 로그인 없이 열리지 않습니다.
        `.streamlit/secrets.toml` 을 만들고 아래 항목을 채워주세요.
        양식은 저장소의 **`.streamlit/secrets.toml.example`** 에 있습니다.

        ```toml
        [auth]
        redirect_uri = "http://localhost:8501/oauth2callback"
        cookie_secret = "<python -c \\"import secrets;print(secrets.token_hex(32))\\" 결과>"

        [auth.google]
        client_id = "<Google Cloud Console 에서 발급>"
        client_secret = "<동일>"
        server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"
        ```

        설정 후 앱을 다시 시작하세요.

        ---
        로컬에서 인증 없이 잠깐 확인만 하려면 `NMAP_ALLOW_ANONYMOUS=1` 환경변수를 설정하고
        실행하세요. **실제 환자 데이터로는 사용하지 마세요.**
        """
    )


def _render_login_screen():
    st.title("🏥 N-Map")
    st.markdown("간호 연관성 분석 도구입니다. 계속하려면 로그인하세요.")
    st.divider()

    providers = _configured_providers()
    if not providers:
        # 단일 공급자 설정
        st.button("로그인", type="primary", on_click=st.login, use_container_width=True)
    else:
        for provider in providers:
            label = PROVIDER_LABELS.get(provider, f"{provider} 계정으로 로그인")
            st.button(
                label,
                key=f"login_{provider}",
                type="primary",
                use_container_width=True,
                on_click=st.login,
                args=(provider,),
            )

    st.caption("인가된 계정만 접근할 수 있습니다. 접근 권한이 필요하면 관리자에게 문의하세요.")


def _render_denied(user_email):
    st.title("⛔ 접근 권한 없음")
    st.error(f"`{user_email or '알 수 없는 계정'}` 계정에는 이 앱의 접근 권한이 없습니다.")
    st.markdown("관리자에게 접근 권한을 요청하거나, 다른 계정으로 로그인하세요.")
    st.button("다른 계정으로 로그인", on_click=st.logout)


def require_login():
    """로그인 게이트. 인가되지 않은 접근이면 화면을 그리고 실행을 중단한다.

    Returns:
        인증을 통과했으면 True. 익명 허용 모드면 False (로그인하지 않은 상태로 진행).
    """
    if not is_configured():
        if _allow_anonymous():
            st.warning(
                "⚠️ 인증이 꺼진 상태로 실행 중입니다(NMAP_ALLOW_ANONYMOUS=1). "
                "실제 환자 데이터로는 사용하지 마세요.",
                icon="⚠️",
            )
            return False
        _render_setup_help()
        st.stop()

    if not _is_logged_in():
        _render_login_screen()
        st.stop()

    email = getattr(st.user, "email", None)
    if not _is_allowed(email):
        _render_denied(email)
        st.stop()

    return True


def render_user_box():
    """사이드바에 현재 로그인 사용자와 로그아웃 버튼을 표시한다."""
    if not is_configured() or not _is_logged_in():
        return

    name = getattr(st.user, "name", None) or getattr(st.user, "email", None) or "사용자"
    email = getattr(st.user, "email", "") or ""

    with st.sidebar:
        st.markdown(f"**{name}**")
        if email and email != name:
            st.caption(email)
        st.button("로그아웃", on_click=st.logout, use_container_width=True)
        st.markdown("---")
