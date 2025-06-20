# ===== 0. 패키지 임포트 =====
# 표준 라이브러리
import datetime
import re
import time

# 서드파티 라이브러리
import gspread
import koreanize_matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from google.oauth2.service_account import Credentials
from openai import OpenAI
from st_aggrid import AgGrid, GridOptionsBuilder

# 로컬 모듈
import utils as eda  # 사용자 정의 함수 모듈 (그대로 사용)


# ===== 1. 스트림릿 설정 =====
st.set_page_config(
    page_title="AI와 함께하는 통계적 문제해결",
    page_icon="🖼️",
)



# ===== 3. 상단 메뉴 =====
def top_menu() -> None:
    """홈 · 로그아웃 버튼"""
    _, col_home, col_logout = st.columns([0.5, 0.25, 0.25])

    # 홈 버튼
    if col_home.button("🏠 홈", use_container_width=True):
        st.rerun()

    # 로그아웃 버튼
    if col_logout.button("🔒 로그아웃", use_container_width=True):
        # 세션 상태 초기화
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        # cache_data 캐시 비우기
        st.cache_data.clear()
        st.rerun()


top_menu()


# ===== 4. 타이틀 · 앱 소개 =====
st.title("📊 AI와 함께하는 통계적 문제해결")
st.info(
    """**웹앱 소개**
이 웹앱은 데이터를 살펴보고, 탐구 질문을 작성·검토하고,
그래프 해석-결론까지 ‘PPDAC’ 전 과정을 경험하도록 설계되었습니다.
각 단계에서 AI(챗GPT)가 즉시 피드백을 주어 통계적 사고를 확장할 수 있습니다.
원시 자료를 살펴보고, 단계에 따라 통계적 탐구활동을 진행해봅시다."""
)
# ===== 2. 로그인 =====
def login() -> None:
    """세션 상태 기반 간단 로그인"""
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "sid" not in st.session_state:
        st.session_state.sid = None

    if not st.session_state.logged_in:
        with st.form("login", clear_on_submit=True):
            sid = st.text_input("이름")
            pw = st.text_input("인증코드", type="password")
            ok = st.form_submit_button("로그인", type="primary")

        if ok:
            if sid and pw == st.secrets.password.logincode:
                st.session_state.update({"logged_in": True, "sid": sid})
                st.rerun()
            elif sid and pw == "banpo":
                st.session_state.update({"logged_in": True, "sid": sid})
                st.rerun()
            else:
                st.error("이름 또는 인증코드를 확인하세요.")
        st.stop()


login()

def extract_number(text):
    circled = {"①":1,"②":2,"③":3,"④":4,"⑤":5,"⑥":6,"⑦":7}
    if isinstance(text,str) and text:
        if text[0] in circled:                # ①~⑦
            return circled[text[0]]
        m = re.match(r"\d+", text)            # ‘3번’, ‘6회’ 등
        if m: return int(m.group())
    return None



    
def convert_labeled_column(df, colname):
    """범주형 라벨에서 숫자를 추출해 수치형 + 범주형 컬럼 생성"""
    if colname not in df.columns:
        return
    df[f"{colname}_num"] = df[colname].apply(extract_number)
    df[f"{colname}_cat"] = df[colname].astype("category")

# ===== 5. 헬퍼 함수 =====
def split_sentences(text: str) -> list[str]:
    """텍스트를 문장 단위로 분리"""
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def insert_chat_bubble_css() -> None:
    """채팅 말풍선용 CSS"""
    st.markdown(
        """
        <style>
        .chat-container{display:flex;align-items:flex-start;margin-top:2px;}
        .chat-container:last-child{margin-bottom:10px;}
        .chat-icon{font-size:24px;margin-right:8px;margin-top:4px;}
        .chat-bubble{padding:10px 14px;border-radius:16px;max-width:80%;
                     font-size:15px;line-height:1.45;display:inline-block;}
        .bubble-color-0{background:#e0f0ff}.bubble-color-1{background:#fce4ec}
        .bubble-color-2{background:#e8f5e9}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ===== 6. 피드백 히스토리 =====
def show_bubble(icon: str, text: str, color_class: str) -> None:
    """문단을 말풍선 형태로 렌더링"""
    st.markdown(
        f"""
        <div class="chat-container">
            <div class="chat-icon">{icon}</div>
            <div class="chat-bubble {color_class}">{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


bubble = show_bubble  # 호환용 별칭


def render_feedback_history(feedbacks: list[str]) -> None:
    """최신 피드백은 펼치고, 과거 피드백은 접어 두는 히스토리 뷰"""
    if not feedbacks:
        return
    insert_chat_bubble_css()
    for i, fb in enumerate(reversed(feedbacks), start=1):
        idx = (i - 1) % 3
        orig_no = len(feedbacks) - i + 1
        with st.expander(f"피드백 {orig_no}", expanded=(i == 1)):
            for sent in split_sentences(fb):
                bubble("🧑🏻‍🏫", sent, f"bubble-color-{idx}")


# ===== 7. Google Sheets 전송 헬퍼 =====
def push_last_log() -> None:
    """ai_logs의 마지막 레코드를 Google Sheets에 즉시 전송"""
    if not st.session_state.ai_logs:
        return
    log = st.session_state.ai_logs[-1]
    row = [
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        st.session_state.sid,
        log.get("stage", ""),
        log.get("input", ""),
        log.get("ai_feedback", ""),
        log.get("ai_level", ""),
    ]
    try:
        connect_sheet().append_row(row, value_input_option="USER_ENTERED")
    except Exception as e:
        st.error(f"Sheets 전송 실패: {e}")


# ===== 8. 세션 초기값 =====
def set_default_session() -> None:
    defaults = dict(
        df=None,
        show_visualization=False,
        last_code="",
        ai_logs=[],
        var_list=[],
        plot_args=None,
        problem_feedbacks=[],
        problem_feedback_count=0,
        da_feedbacks=[],
        da_fb_count=0,
    )
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


set_default_session()


# ===== 9. OpenAI 클라이언트 =====
def get_openai_client():
    try:
        return OpenAI(api_key=st.secrets.openai.api_key)
    except Exception:
        st.error("🔑 OpenAI API 키가 없습니다. secrets.toml을 확인하세요.")
        st.stop()


client = get_openai_client()


# ===== 10. Google Sheets 유틸 =====
def connect_sheet(sheet: str = "response"):
    creds = Credentials.from_service_account_info(
        st.secrets.google_sheets,
        scopes=["https://www.googleapis.com/auth/spreadsheets"],
    )
    gc = gspread.authorize(creds)
    return gc.open_by_key(st.secrets.google_sheets.sheet_id).worksheet(sheet)

# ===== 11. 데이터 로드 =====
@st.cache_data(show_spinner="📥 시트에서 데이터 불러오는 중…")
def load_data() -> pd.DataFrame:
    ws = connect_sheet("data")
    raw = ws.get_all_values()
    df  = pd.DataFrame(raw[1:], columns=raw[0])

    # ① ‘(수)’로 시작하는 열은 무조건 숫자로 변환
    num_cols = [c for c in df.columns if c.startswith("(수)")]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

    # ② 나머지는 기존 문자열 그대로 유지
    st.session_state.df = df
    return df.copy()


df = load_data()


# ===== 12. 원시 데이터 뷰어 =====
def show_raw_data(df: pd.DataFrame) -> None:
    st.subheader("💽 원시데이터 살펴보기")
    with st.expander(
        "아래의 데이터는 중고등학생 200명의 신체 관련 설문조사 데이터입니다. "
        "이곳을 클릭하여 데이터 설명을 먼저 읽어보세요."
    ):
        st.success(
            """
            이 화면에서는 청소년의 건강과 생활습관에 관한 실제 데이터를 직접 살펴볼 수 있습니다.  
            (중략)
            *자료 출처: 청소년건강행태조사 원시자료(교육부, 질병관리청, 2024)*
            """
        )

    len_threshold, wide_px = 30, 150
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_default_column(
        wrapText=True,
        autoHeight=True,
        wrapHeaderText=True,
        autoHeaderHeight=True,
        resizable=True,
    )
    for col in df.columns:
        if df[col].dtype == "object" and df[col].astype(str).str.len().mean() > len_threshold:
            gb.configure_column(
                col,
                width=wide_px,
                minWidth=wide_px,
                maxWidth=wide_px,
                suppressSizeToFit=True,
            )
    AgGrid(
        df,
        gridOptions=gb.build(),
        fit_columns_on_grid_load=False,
        enable_enterprise_modules=False,
        theme="balham",
    )

    if st.checkbox("↔️ 자료의 세로와 가로를 바꿔서 볼래요!"):
        st.dataframe(df.T.reset_index().rename(columns={"index": "질문"}))


# 원시데이터는 한 번만 표시
show_raw_data(df)


# ===== 13. Google Sheets 전송 헬퍼 =====
def push_log(log: dict) -> None:
    if log.get("_pushed"):
        return
    row = [
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        st.session_state.sid,
        log.get("stage", ""),
        log.get("input", ""),
        log.get("ai_feedback", ""),
        log.get("ai_level", ""),
    ]
    try:
        connect_sheet().append_row(row, value_input_option="USER_ENTERED")
        log["_pushed"] = True
    except Exception as e:
        st.error(f"Sheets 전송 실패: {e}")


# ===== 14. AI 수준 추출 =====
def extract_level(text: str) -> str:
    m = re.search(r"단계[:\s]+(\d)", text)
    return m.group(1) if m else ""


# ===== 15. 말풍선 CSS & util =====
insert_bubble_css = insert_chat_bubble_css
split_sent = split_sentences


# ===== 16. 프롬프트 로더 =====
@st.cache_data(ttl=10)
def get_prompt(key: str, tab: str = "prompt") -> str:
    creds = Credentials.from_service_account_info(
        st.secrets.google_sheets,
        scopes=["https://www.googleapis.com/auth/spreadsheets"],
    )
    ws = (
        gspread.authorize(creds)
        .open_by_key(st.secrets.google_sheets.sheet_id)
        .worksheet(tab)
    )
    for row in ws.get_all_records():
        if row.get("key") == key:
            return row.get("prompt", "")
    return ""


# ===== 17. GPT 호출 (캐시 TTL 30분) =====
@st.cache_data(show_spinner="AI 피드백 생성 중...", ttl=1800)
def ask_gpt(prompt: str) -> str:
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    return res.choices[0].message.content.strip()



# ===== 숫자 접두어 기반 정렬 =====
def make_numeric_order(series: pd.Series) -> list[str]:
    def _num_key(label: str):
        m = re.match(r"[①②③④⑤⑥⑦⑧⑨⑩]|(\d+)", label.strip())
        if m:
            circled = "①②③④⑤⑥⑦⑧⑨⑩"
            if m.group(0) in circled:
                return circled.index(m.group(0)) + 1
            if m.group(1):
                return int(m.group(1))
        return float("inf")
    uniq = list(dict.fromkeys(series.dropna()))
    return sorted(uniq, key=_num_key)



# ===== 18. 1. Problem TAB =====
def problem_tab() -> None:
    st.subheader("🔍 통계적 문제 정의")
    st.info("""
    **1️⃣ 문제 정의하기**  
    이 단계에서는 탐구하고 싶은 주제를 정하고, 스스로 통계적 질문을 만들어봅니다.  
    주어진 데이터를 관찰하고 흥미롭거나 궁금한 점이 무엇인지 생각해보세요.  
    예를 들어 “중학생과 고등학생은 스트레스를 얼마나 다르게 느낄까?”처럼  
    비교하거나 예측할 수 있는 질문을 만들어보는 것이 좋습니다.
    """)
    # 입력값을 세션에 저장
    if "final_q" not in st.session_state:
        st.session_state.final_q = ""

    q = st.text_area(
        "탐구 질문을 입력하세요",
        value=st.session_state.final_q,
        placeholder="예) 운동 빈도와 스트레스 수준은 관련이 있을까?",
        key="question_input"
    )

    # 텍스트 입력이 바뀌면 세션값도 즉시 반영
    st.session_state.final_q = q

    st.markdown(f"**피드백 요청:** {st.session_state.problem_feedback_count} / 3")

    if st.button("🧑🏻‍🏫 AI 피드백 받기 (질문하기)", key="pb_fb",
                disabled=st.session_state.problem_feedback_count >= 3):
        if not st.session_state.final_q.strip():
            st.warning("먼저 질문을 입력하세요.")
        else:
            diag = (
                f"{get_prompt('step1_diagnosis')}\n\n"
                f"학생 질문: {st.session_state.final_q}\n"
                f"데이터 컬럼명: {list(df.columns)}"
            )
            level = ask_gpt(diag).strip()
            fbp = (
                f"{get_prompt('step1_feedback')}\n\n"
                f"학생 질문: {st.session_state.final_q}\n"
                f"학생 통계적 소양 수준: {level}\n"
            )
            fb = ask_gpt(fbp)
            st.session_state.problem_feedback_count += 1
            st.session_state.problem_feedbacks.append(fb)
            st.session_state.ai_logs.append({
                "stage": "1.Problem",
                "input": st.session_state.final_q.strip(),
                "ai_feedback": fb,
                "ai_level": level
            })
            push_last_log()

    render_feedback_history(st.session_state.problem_feedbacks)

    st.divider()
    if st.button("💾 저장하기(통계적 질문)", key="pb_confirm", type="primary", use_container_width=True):
        if not q.strip():
            st.warning("먼저 질문을 입력하세요.")
        else:
            st.session_state.final_q = q.strip()
            # for log in st.session_state.ai_logs:
            #     if log["stage"].startswith("1.") and not log.get("_pushed"):
            #         push_log(log)
            st.success("질문이 확정되었습니다! 2단계로 이동하세요.")

def show_user_card(title: str, items: dict[str, str]) -> None:
    """
    Streamlit HTML + inline-CSS 버전 ― 리렌더링에도 항상 스타일이 적용됩니다.
    """
    # ① 항목(라벨·값) HTML 조립
    rows = "".join(
        f"<p style='margin:4px 0; font-size:15px; line-height:1.45;'>"
        f"<strong style='color:#0d6efd;'>{label}</strong>  {val}</p>"
        for label, val in items.items()
    )

    # ② 카드 렌더링 (스타일을 div 안에 인라인으로 포함)
    st.markdown(
        f"""
        <div style="
            border:1px solid #dee2e6; border-radius:8px;
            background:#BEE4D0; padding:14px 18px; margin:6px 0;">
          <h4 style="margin:0 0 8px; font-size:18px;">{title}</h4>
          {rows}
        </div>
        """,
        unsafe_allow_html=True
    )

# ===== 19. 2. Plan TAB (변수 목록까지 로그 저장) =====
def plan_tab() -> None:
    st.subheader("📝 계획 수립")
    st.info(
        """
        **2️⃣ 계획 수립하기**  
        설정한 질문을 바탕으로 분석에 사용할 변수를 선택하고,  
        어떤 방법으로 분석할지 스스로 계획을 세웁니다.  
        예를 들어 ‘학교급’과 ‘스트레스 수준’을 선택해 비교 분석할 수 있습니다.  
        이 단계에서는 AI의 도움 없이 자신만의 계획을 세우는 것을 추천합니다.
        """
    )

    # 1) 1단계 완료 여부 확인 ────────────────────────────────────────────
    ready = bool(st.session_state.get("final_q"))
    if not ready:
        st.warning("⚠️ 먼저 1단계에서 [💾 저장하기(통계적 질문)] 버튼을 눌러 질문을 확정해야 합니다.")
        return

    # 2) 1단계 핵심 정보 카드 -------------------------------------------
    show_user_card(
        "📌 나의 입력 요약",
        {"최종 질문": st.session_state.get("final_q", "(미작성)")}
    )

    # 3) 입력 위젯 -------------------------------------------------------
    st.session_state.var_list = st.multiselect(
        "분석 변수 선택",
        options=df.columns,
        default=st.session_state.var_list or [],
    )
    st.session_state.myplan = st.text_area(
        "분석 계획 작성",
        value=st.session_state.get("myplan", ""),
    )

    # 4) 저장 버튼 -------------------------------------------------------
    if st.button("💾 저장하기 (Plan)", use_container_width=True, type="primary"):
        if not st.session_state.myplan.strip():
            st.warning("계획을 작성하세요.")
            return

        # (1) 같은 스테이지 로그 제거 후 재저장
        st.session_state.ai_logs = [
            l for l in st.session_state.ai_logs if l.get("stage") != "2. Plan"
        ]

        # (2) ‘선택한 변수’와 ‘학생 계획’을 한 문자열에 담아 input 필드로 저장
        log_input = (
            f"{st.session_state.var_list}\n"
            f"{st.session_state.myplan.strip()}"
        )
        st.session_state.ai_logs.append(
            {
                "stage": "2. Plan",
                "input": log_input,
                "ai_feedback": "",
                "ai_level": "",
            }
        )

        # (3) Google Sheets 전송
        push_last_log()
        st.success("계획이 저장되었습니다!")



from scipy.stats import t        # ← 신뢰구간 계산에 사용
import numpy as np 

def data_analysis_tab() -> None:
    """3단계: 시각화·통계요약·해석 + AI 피드백 + 신뢰구간 추정"""
    st.subheader("📈 데이터 시각화 · 통계 요약 · 해석")
    st.info(
        """
        **3️⃣ 자료 분석 및 해석**  
        선택한 변수들을 시각화하고 데이터를 분석해봅니다.  
        그래프나 통계값을 통해 패턴을 찾아보고, 질문에 대한 답을 추론해보세요.
        """
    )

    # # ───────────── 데이터 준비 ─────────────
    # # 0) 데이터프레임 존재 여부 확인
    # if "df" not in st.session_state or not isinstance(st.session_state.df, pd.DataFrame):
    #     st.error("먼저 데이터를 업로드하거나 Plan 탭에서 변수 선택을 완료하세요.")
    #     st.stop()
    # ───────────── 데이터 준비 ─────────────
    # 0) 데이터프레임 존재 여부 확인 (없으면 즉시 재로드 시도)
    if "df" not in st.session_state or st.session_state.df is None:
        try:
            st.session_state.df = load_data()       # Google Sheets에서 다시 불러오기
        except Exception as e:
            st.error(f"데이터를 불러오지 못했습니다: {e}")
            st.stop()

    # 1) 변수 목록 확인
    var_list: list[str] = st.session_state.get("var_list", [])
    if not var_list:
        st.error("Plan 탭에서 분석 변수를 먼저 고르세요.")
        st.stop()

    # 2) 선택한 변수가 실제 컬럼에 있는지 검증
    missing_vars = [v for v in var_list if v not in st.session_state.df.columns]
    if missing_vars:
        st.error(f"데이터에 없는 변수: {missing_vars} — Plan 탭에서 다시 선택해주세요.")
        st.stop()

    # 3) 검증 통과 후 서브데이터프레임 생성
    # ---------------------------------------------
    # 3) 검증 통과 후 서브데이터프레임 생성
    df_sel = st.session_state.df[var_list].copy()

    # ▶ 선택한 열 중 ‘숫자 추출 가능’한 범주형 → 두 갈래 컬럼 추가
    for col in var_list:
        if df_sel[col].dtype == "object":
            if df_sel[col].astype(str).str.contains(r"\d").any():
                df_sel[f"{col}_cat"] = df_sel[col]                 # 원본 라벨
                df_sel[f"{col}_num"] = df_sel[col].apply(extract_number)
    # ---------------------------------------------
                


    show_user_card(
        "📌 나의 입력 요약",
        {
            "최종 질문": st.session_state.get("final_q", "(미작성)"),
            "나의 계획": st.session_state.get("myplan", "(미작성)"),
        },
    )
    # # ──────────── 분석 모드 선택 ────────────
    # st.markdown("#### 🎛️ 분석 옵션 (하나만 체크)")

    # chk_all = st.checkbox("① 전체 그래프",     key="chk_all")
    # chk_uni = st.checkbox("② 단일 변수 분석",   key="chk_uni")
    # chk_sel = st.checkbox("③ 선택 그래프",     key="chk_sel", value=not (chk_all or chk_uni))
    # chk_ci  = st.checkbox("④ 신뢰구간 추정 🔹", key="chk_ci")

    # checked = [name for name, flag in [
    #     ("① 전체 그래프",   chk_all),
    #     ("② 단일 변수 분석", chk_uni),
    #     ("③ 선택 그래프",   chk_sel),
    #     ("④ 신뢰구간 추정 🔹", chk_ci)
    # ] if flag]

    # # 하나만 선택되었는지 검증
    # if len(checked) != 1:
    #     st.warning("분석 모드를 **하나만** 선택하세요.")
    #     st.stop()

    # mode = checked[0]      # 이후 로직은 기존 코드 그대로 사용


    # ─────────── 공통: 선택 그래프 캐시 ───────────
    @st.cache_data(show_spinner="🎨 그래프 렌더링 중…")
    def _cached_choice_plot(df: pd.DataFrame, args: tuple, rot: int):
        gtype, vx, vy, ox, oy = args
        if vx == vy:  # 단변량
            return eda.선택해서_그래프_그리기(
                df, col=vx, graph_type=gtype, order=ox, rot_angle=rot
            )
        return eda.선택해서_그래프_그리기_이변량(
            df,
            x_var=vx,
            y_var=vy,
            graph_type=gtype,
            order=ox,
            hue_order=oy,
            rot_angle=rot,
        )

    # AFTER ─────────────────────────────────────────────
    st.markdown("#### 🎛️ 분석 도구 (모든 기능 한눈에)")

    # ===== UI: ① 통계량 (단일·다변량, - 그룹 X) =====
    with st.expander("① 통계량 (그룹 없이 전체 집계)", expanded=False):

        # 1) 수치 컬럼만 추출 ----------------------------------------------
        num_cols = [c for c in df_sel.columns
                    if c.endswith("_num") or pd.api.types.is_numeric_dtype(df_sel[c])]

        # 2) 입력 위젯 -----------------------------------------------------
        target_vars = st.multiselect(
            "수치 변수 선택 (1개 이상)", num_cols,
            default=st.session_state.get("stat_nums", num_cols[:1]),
            key="stat_nums"
        )

        # 3) 실행 버튼 -----------------------------------------------------
        if st.button("📊 통계량 계산", key="btn_stats"):

            # ── (A) 입력 검증 ──────────────────────────────────────────
            if not target_vars:
                st.warning("👉 수치 변수를 하나 이상 선택하세요.")
                st.stop()

            # ── (B) 통계 요약 -------------------------------------------
            sel_df   = df_sel[target_vars].copy()
            df_stats = eda.summarize(sel_df, by=None)   # ← 그룹 변수 없음
            st.dataframe(df_stats)

            # ── (C) 상관계수(선택 변수 ≥2) -----------------------------
            if len(target_vars) >= 2:
                corr = sel_df[target_vars].corr().round(2)
                st.subheader("🔗 상관계수")
                st.dataframe(corr)


    with st.expander("② 시각화 (단일·다변량)", expanded=True):
        col1, col2 = st.columns(2)
        var_x = col1.selectbox("가로축(X)", df_sel.columns, key="viz_x")
        var_y = col2.selectbox("세로축(Y) — 단일 그래프는 X=Y로 설정", ["(같음)"] + list(df_sel.columns), key="viz_y")
        gtype = st.selectbox(
            "그래프 종류",
            ["막대그래프", "히스토그램", "도수분포다각형", "꺾은선그래프", "상자그림", "산점도"],
            key="viz_gtype"
        )


        if st.button("🖼️ 그래프 그리기", key="btn_viz"):
            vx = var_x
            vy = var_x if var_y == "(같음)" else var_y
            args = (
                gtype,
                vx,
                vy,
                make_numeric_order(df_sel[vx]) if not pd.api.types.is_numeric_dtype(df_sel[vx]) else None,
                make_numeric_order(df_sel[vy]) if not pd.api.types.is_numeric_dtype(df_sel[vy]) else None,
            )
            fig = _cached_choice_plot(df_sel, args, rot=0)
            if fig:
                st.pyplot(fig, use_container_width=True)
                st.session_state.last_code = f"# {gtype} for {vx}/{vy}, size=(10,10)"




    # ④ 신뢰구간 추정 ------------------------------------------------------
    with st.expander("③ 신뢰구간 추정 🔹", expanded=False):
        num_cols = [c for c in df_sel.columns if pd.api.types.is_numeric_dtype(df_sel[c])]
        cat_cols = [c for c in df_sel.columns if c not in num_cols]
        num_var  = st.selectbox("📐 수치형 변수", num_cols, key="ci_num")
        grp_var  = st.selectbox("🗂️ 그룹 변수 (없으면 ‘(단일)’) ", ["(단일)"] + cat_cols, key="ci_grp")
        conf     = st.radio("신뢰수준 선택", (95, 99), horizontal=True, key="ci_conf")
        alpha    = 1 - conf/100
        if st.button("📏 신뢰구간 추정"):

            # ― (단일) 전체 신뢰구간 ―
            if grp_var == "(단일)":
                s    = df_sel[num_var].dropna()
                n    = len(s)
                mean = s.mean()
                se   = s.std(ddof=1)/np.sqrt(n)
                h    = t.ppf(1-alpha/2, max(n-1,1)) * se
                ci_df = pd.DataFrame({
                    "label": ["전체"],
                    "mean":  [mean],
                    "lo":    [mean-h],
                    "hi":    [mean+h]
                })

            # ― 그룹별 신뢰구간 ―
            else:
                stats = (
                    df_sel
                    .groupby(grp_var)[num_var]
                    .agg(count="count", mean="mean", std="std")
                    .reset_index()
                    .rename(columns={grp_var: "label"})
                )
                stats["se"] = stats["std"] / np.sqrt(stats["count"])
                df_t = t.ppf(1-alpha/2, np.maximum(stats["count"]-1, 1))
                stats["lo"] = stats["mean"] - df_t * stats["se"]
                stats["hi"] = stats["mean"] + df_t * stats["se"]

                ci_df = stats[["label", "mean", "lo", "hi"]]

            # 2) 시각화
            fig, ax = plt.subplots(figsize=(8, 0.6*len(ci_df)+1))
            ax.errorbar(
                x=ci_df["mean"],
                y=ci_df["label"],
                xerr=[ci_df["mean"]-ci_df["lo"], ci_df["hi"]-ci_df["mean"]],
                fmt="o", capsize=6, elinewidth=2, markersize=5
            )
            ax.set_xlabel(f"{num_var}  (신뢰수준 {conf}%)")
            ax.set_title("그룹별 평균과 신뢰구간")
            ax.grid(axis="x", ls="--", alpha=0.4)
            st.pyplot(fig, use_container_width=True)

            # 3) LaTeX 부등호
            st.markdown("#### 📑 신뢰구간 결과")
            for _, row in ci_df.iterrows():
                lbl, m, lo, hi = row["label"], row["mean"], row["lo"], row["hi"]
                st.latex(
                    rf"\text{{{lbl}}}: \; {lo:.2f} \;\le\; \mu \;\le\; {hi:.2f}"
                )
            # 4) 코드 저장
            st.session_state.last_code = f"# CI for {num_var} by {grp_var}, {conf}%"

    # ------------------------------------------------------------------ #
    # 🧑🏻‍🏫 AI 피드백
    # ------------------------------------------------------------------ #
    st.session_state.setdefault("da_fb_count", 0)
    st.session_state.setdefault("da_feedbacks", [])

    interp = st.text_area("그래프를 해석해 보세요(2-3문장)", key="interp_da")
    st.session_state.interp = interp
    st.markdown(f"**🧠 피드백 요청: {st.session_state.da_fb_count} / 3회**")

    if st.button("🧑🏻‍🏫 AI 피드백 받기 (데이터 분석)", key="btn_da_fb",
                disabled=st.session_state.da_fb_count >= 3):
        if not interp.strip():
            st.warning("먼저 해석을 입력하세요.")
        else:
            diag = (
                f"{get_prompt('step3_diagnosis')}\n\n"
                f"학생 질문: {st.session_state.final_q}\n"
                f"데이터 컬럼명: {list(df.columns)}"
            )
            level = ask_gpt(diag).strip()

            fbp = (
                f"{get_prompt('step3_feedback')}\n\n"
                f"학생 질문: {st.session_state.final_q}\n"
                f"학생 통계적 소양 수준: {level}\n"
                f"학생 해석: {interp.strip()}"
            )

            fb = ask_gpt(fbp)
            st.session_state.da_fb_count += 1
            st.session_state.da_feedbacks.append(fb)
            st.session_state.ai_logs.append({
                "stage": "3. Data&Analysis",
                "input": interp.strip(),
                "code": st.session_state.get("last_code", ""),
                "ai_feedback": fb,
                "ai_level": level
            })
            push_last_log()

    render_feedback_history(st.session_state.da_feedbacks)
    # ✅✅✅ ―――――――――――――――――――――――――――――――――――――――――
    # 🔸 3단계 결과 저장 → 4단계(결론) 활성화 플래그
    if st.button("💾 저장하기 (Data & Analysis)", key="btn_da_save",
                 type="primary", use_container_width=True):
        st.session_state.da_saved = True      # ← 결론 탭 사용 허가
        st.success("3단계 결과가 저장되었습니다! ‘결론’ 단계로 이동해 보세요.")


## ─────────────────────────────────────────────────────────
##  A. 헬퍼 함수: 최종 소감(Google Sheets 전송)
## ─────────────────────────────────────────────────────────
def push_reflection(text: str) -> None:
    """마지막 소감·성찰을 reflection 시트에 저장"""
    row = [
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        st.session_state.sid,
        text,
    ]
    try:
        connect_sheet("reflection").append_row(row, value_input_option="USER_ENTERED")
    except Exception as e:
        st.error(f"Sheets 전송 실패: {e}")

def conclusion_tab() -> None:
    # 항상 헤더와 안내는 표시
    st.subheader("📝 결론 · 소감")
    st.info("""
    **4️⃣ 분석 결과를 정리하고 활동 소감을 작성하세요.**  
    - 먼저 설정한 질문에 대한 결론을 작성하고,  
    - 분석 과정에서 느낀 점이나 새롭게 알게 된 점을 함께 정리해봅니다.
    """)

    # 이전 단계 핵심 정보 표시
    show_user_card(
        "📌 나의 입력 요약",
        {
            "최종 질문":      st.session_state.get("final_q",    "(미작성)"),
            "나의 계획":      st.session_state.get("myplan",     "(미작성)"),
            "나의 분석 결과": st.session_state.get("interp_da", "(미작성)")
        }
    )

    # 3단계 저장 여부 확인 후 안내 또는 입력창 제공
    if not st.session_state.get("da_saved"):
        st.warning("⚠️ 먼저 3단계에서 **저장하기 (Data & Analysis)** 버튼을 눌러야 전송할 수 있습니다.")
        return

    # 통합 입력창
    combined_input = st.text_area(
        "🔖 결론과 소감 (필수) — 아래 항목을 함께 포함해 작성하세요.\n"
        "- 설정한 질문에 대한 결론\n"
        "- 분석에 사용한 근거 및 과정\n"
        "- 느낀 점, 어려웠던 점, 새롭게 알게 된 점 등",
        key="combined_conclusion_reflection",
        height=300,
    )

    # 전송 버튼
    if st.button("📤 결론·소감 전송", type="primary", use_container_width=True):
        if not combined_input.strip():
            st.warning("결론과 소감을 모두 포함해 작성해야 전송됩니다.")
            st.stop()

        # 저장: 결론 로그
        st.session_state.ai_logs = [
            lg for lg in st.session_state.ai_logs if lg.get("stage") != "4. Conclusion"
        ]
        st.session_state.ai_logs.append({
            "stage": "4. Conclusion",
            "input": combined_input.strip(),
            "ai_feedback": "",
            "ai_level": "",
        })
        push_last_log()

        # 저장: 소감은 별도로 추출해서 저장
        push_reflection(combined_input.strip())

        # 완료 메시지
        st.success("결론과 소감이 전송되었습니다! 🎉")
        st.balloons()

        # 입력 초기화
        st.session_state.combined_conclusion_reflection = ""


# ===== 22. 탭 UI =====
tab_problem, tab_plan, tab_da, tab_concl = st.tabs(
    ["1️⃣ 문제 정의하기", "2️⃣ 계획 수립하기", "3️⃣ 자료 분석 및 해석", "4️⃣ 결론 도출하기"]
)

with tab_problem:
    problem_tab()

with tab_plan:
    plan_tab()

with tab_da:
    data_analysis_tab()

with tab_concl:
    conclusion_tab()
