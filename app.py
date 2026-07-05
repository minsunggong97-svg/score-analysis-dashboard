import platform
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import streamlit as st


st.set_page_config(
    page_title="성적 분석 대시보드",
    page_icon="📊",
    layout="wide",
)


def set_korean_font() -> None:
    system = platform.system()
    if system == "Darwin":
        plt.rcParams["font.family"] = "AppleGothic"
    elif system == "Windows":
        plt.rcParams["font.family"] = "Malgun Gothic"
    else:
        nanum_font = Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf")
        if nanum_font.exists():
            fm.fontManager.addfont(nanum_font)
        plt.rcParams["font.family"] = "NanumGothic"

    plt.rcParams["axes.unicode_minus"] = False
    sns.set_theme(
        style="whitegrid",
        palette="husl",
        rc={
            "font.family": plt.rcParams["font.family"],
            "axes.unicode_minus": False,
        },
    )


def read_uploaded_file(uploaded_file) -> pd.DataFrame:
    file_name = uploaded_file.name.lower()
    if file_name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)


def make_sample_scores() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    scores = rng.normal(loc=72, scale=12, size=110)
    scores = np.clip(scores, 0, 100).round(1)
    return pd.DataFrame({"점수": scores})


def get_numeric_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include="number").columns.tolist()


def calculate_normality(scores: pd.Series) -> tuple[float | None, str]:
    if len(scores) < 3:
        return None, "표본 수가 3개 미만이라 정규성 검정을 계산할 수 없습니다."

    _, p_value = stats.shapiro(scores)
    if p_value >= 0.05:
        message = "정규분포와 크게 다르다고 보기 어렵습니다."
    else:
        message = "정규분포와 차이가 있을 가능성이 있습니다."
    return p_value, message


def render_summary(scores: pd.Series) -> None:
    q1 = scores.quantile(0.25)
    median = scores.median()
    q3 = scores.quantile(0.75)
    min_score = scores.min()
    max_score = scores.max()

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("총 인원", f"{len(scores)}명")
    col2.metric("평균", f"{scores.mean():.1f}점")
    col3.metric("표준편차", f"{scores.std():.1f}")
    col4.metric("중앙값", f"{median:.1f}점")
    col5.metric("최고점", f"{max_score:.1f}점")

    st.caption(
        f"최솟값 {min_score:.1f}점 · 1사분위수 {q1:.1f}점 · "
        f"중앙값 {median:.1f}점 · 3사분위수 {q3:.1f}점 · 최댓값 {max_score:.1f}점"
    )


def make_histogram(scores: pd.Series, bins: int):
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    sns.histplot(scores, kde=True, bins=bins, color="#3498db", edgecolor="white", ax=ax)
    ax.set_title("성적 히스토그램 및 밀도 곡선")
    ax.set_xlabel("점수")
    ax.set_ylabel("인원 수")
    ax.set_xlim(0, 100)
    fig.tight_layout()
    return fig


def make_boxplot(scores: pd.Series):
    fig, ax = plt.subplots(figsize=(6.4, 2.2))
    sns.boxplot(x=scores, color="#2ecc71", width=0.45, ax=ax)
    ax.set_title("성적 상자수염그림")
    ax.set_xlabel("점수")
    ax.set_xlim(0, 100)
    fig.tight_layout()
    return fig


def make_qq_plot(scores: pd.Series):
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    stats.probplot(scores, dist="norm", plot=ax)
    if len(ax.get_lines()) > 1:
        ax.get_lines()[1].set_color("#e74c3c")
        ax.get_lines()[1].set_linewidth(2)
    ax.set_title("Q-Q Plot")
    ax.set_xlabel("이론적 분위수")
    ax.set_ylabel("실제 점수")
    fig.tight_layout()
    return fig


def main() -> None:
    set_korean_font()

    with st.sidebar:
        st.header("⚙️ 분석 설정")
        uploaded_file = st.file_uploader("성적 파일 업로드", type=["xlsx", "csv"])
        use_sample = st.toggle("가상 110명 데이터 사용", value=uploaded_file is None)
        bins = st.slider("히스토그램 구간", min_value=5, max_value=30, value=15)

        st.divider()
        st.info("발표할 때는 사이드바를 접으면 그래프를 더 크게 보여줄 수 있습니다.")

    st.title("📊 확률과 통계 성적 분석 보고서")
    st.caption("성적 분포, 이상치, 정규성 여부를 한 화면에서 확인하는 발표용 대시보드입니다.")

    try:
        if uploaded_file is not None and not use_sample:
            df = read_uploaded_file(uploaded_file)
        else:
            df = make_sample_scores()
    except Exception as error:
        st.error(f"파일을 읽는 중 문제가 생겼습니다: {error}")
        return

    numeric_columns = get_numeric_columns(df)
    if not numeric_columns:
        st.error("분석할 수 있는 숫자형 점수 열이 없습니다. 점수가 숫자로 입력되어 있는지 확인해주세요.")
        st.dataframe(df.head())
        return

    with st.sidebar:
        score_column = st.selectbox("점수 열 선택", numeric_columns)

    scores = pd.to_numeric(df[score_column], errors="coerce").dropna()

    if scores.empty:
        st.error("선택한 열에서 숫자 점수를 찾을 수 없습니다.")
        return

    st.subheader(f"`{score_column}` 열 분석 결과")
    render_summary(scores)

    top_left, top_right = st.columns(2)
    with top_left:
        st.markdown("#### 📌 성적 분포")
        st.pyplot(make_histogram(scores, bins), use_container_width=True)

    with top_right:
        st.markdown("#### 📌 정규성 확인")
        st.pyplot(make_qq_plot(scores), use_container_width=True)

    bottom_left, bottom_right = st.columns([1.05, 0.95])
    with bottom_left:
        st.markdown("#### 📌 성적 밀집도와 이상치")
        st.pyplot(make_boxplot(scores), use_container_width=True)

    with bottom_right:
        p_value, normality_message = calculate_normality(scores)
        st.markdown("#### 📝 발표 요약")
        st.write(
            f"이 데이터는 평균 **{scores.mean():.1f}점**을 중심으로 분포하며, "
            f"표준편차는 **{scores.std():.1f}점**입니다."
        )
        if p_value is not None:
            st.write(f"Shapiro-Wilk 정규성 검정 p-value는 **{p_value:.4f}**입니다.")
        st.info(normality_message)

        with st.expander("데이터 미리보기"):
            st.dataframe(df.head(20), use_container_width=True)


if __name__ == "__main__":
    main()
