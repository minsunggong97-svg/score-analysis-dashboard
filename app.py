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


def render_summary(scores: pd.Series) -> None:
    q1 = scores.quantile(0.25)
    median = scores.median()
    q3 = scores.quantile(0.75)
    min_score = scores.min()
    max_score = scores.max()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("인원", f"{len(scores)}명")
    col2.metric("평균", f"{scores.mean():.1f}점")
    col3.metric("표준편차", f"{scores.std():.1f}점")
    col4.metric("중앙값", f"{median:.1f}점")

    st.caption(
        f"최솟값 {min_score:.1f}점 · 1사분위수 {q1:.1f}점 · "
        f"중앙값 {median:.1f}점 · 3사분위수 {q3:.1f}점 · 최댓값 {max_score:.1f}점"
    )


def render_histogram(scores: pd.Series, bins: int) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(scores, kde=True, bins=bins, color="skyblue", edgecolor="white", ax=ax)
    ax.set_title("성적 히스토그램 및 밀도 곡선")
    ax.set_xlabel("점수")
    ax.set_ylabel("학생 수")
    ax.set_xlim(0, 100)
    st.pyplot(fig)


def render_boxplot(scores: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.boxplot(x=scores, color="lightgreen", ax=ax)
    ax.set_title("성적 상자수염그림")
    ax.set_xlabel("점수")
    ax.set_xlim(0, 100)
    st.pyplot(fig)


def render_qq_plot(scores: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    stats.probplot(scores, dist="norm", plot=ax)
    ax.set_title("Q-Q Plot")
    ax.set_xlabel("이론적 분위수")
    ax.set_ylabel("실제 점수")
    st.pyplot(fig)

    if len(scores) >= 3:
        _, p_value = stats.shapiro(scores)
        st.write(f"Shapiro-Wilk 정규성 검정 p-value: `{p_value:.4f}`")
        if p_value >= 0.05:
            st.success("p-value가 0.05 이상이므로, 정규분포와 크게 다르다고 보기 어렵습니다.")
        else:
            st.warning("p-value가 0.05 미만이므로, 정규분포와 차이가 있을 가능성이 있습니다.")


def main() -> None:
    set_korean_font()

    st.title("📊 110명 확률과 통계 성적 분석 대시보드")
    st.write(
        "엑셀 또는 CSV 파일을 업로드하면 히스토그램, 상자수염그림, Q-Q Plot을 자동으로 그려줍니다. "
        "파일이 없을 때는 가상의 110명 데이터로 미리 체험할 수 있습니다."
    )

    uploaded_file = st.file_uploader("성적 파일을 업로드해주세요", type=["xlsx", "csv"])
    use_sample = st.toggle("가상 110명 데이터로 미리보기", value=uploaded_file is None)

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

    score_column = st.selectbox("점수가 있는 열을 선택하세요", numeric_columns)
    scores = pd.to_numeric(df[score_column], errors="coerce").dropna()

    if scores.empty:
        st.error("선택한 열에서 숫자 점수를 찾을 수 없습니다.")
        return

    st.subheader(f"`{score_column}` 열 분석 결과")
    render_summary(scores)

    with st.expander("데이터 미리보기"):
        st.dataframe(df.head(20), use_container_width=True)

    bins = st.slider("히스토그램 막대 개수", min_value=5, max_value=20, value=10)

    tab1, tab2, tab3 = st.tabs(["히스토그램", "상자수염그림", "Q-Q Plot"])
    with tab1:
        render_histogram(scores, bins)
    with tab2:
        render_boxplot(scores)
    with tab3:
        render_qq_plot(scores)


if __name__ == "__main__":
    main()
