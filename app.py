import base64
import io
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


def simulate_sample_means(scores: pd.Series, sample_size: int, num_trials: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    score_values = scores.to_numpy()
    sample_means = [
        rng.choice(score_values, size=sample_size, replace=False).mean()
        for _ in range(num_trials)
    ]
    return np.array(sample_means)


def make_clt_plot(
    sample_means: np.ndarray,
    population_mean: float,
    sample_size: int,
    num_trials: int,
    bins: int,
    xlim: tuple[float, float] | None = None,
):
    fig, ax = plt.subplots(figsize=(10, 4.8))
    sns.histplot(sample_means, kde=True, bins=bins, color="#9b59b6", edgecolor="white", ax=ax)
    ax.axvline(
        population_mean,
        color="#e74c3c",
        linestyle="--",
        linewidth=2,
        label=f"전체 평균 ({population_mean:.1f}점)",
    )
    ax.set_title(f"표본평균의 분포 (n={sample_size}, {num_trials}회 추출)")
    ax.set_xlabel("표본평균 점수")
    ax.set_ylabel("빈도")
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.legend()
    fig.tight_layout()
    return fig


def calculate_confidence_interval(sample_means: np.ndarray, confidence_level: float) -> tuple[float, float, float]:
    center = sample_means.mean()
    alpha = 1 - confidence_level / 100
    z_score = stats.norm.ppf(1 - alpha / 2)
    margin = z_score * sample_means.std(ddof=1)
    lower = center - margin
    upper = center + margin
    return lower, upper, upper - lower


def simulate_confidence_intervals(
    scores: pd.Series,
    sample_size: int,
    num_trials: int,
    confidence_level: float,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    score_values = scores.to_numpy()
    population_mean = scores.mean()
    z_score = stats.norm.ppf(1 - (1 - confidence_level / 100) / 2)
    rows = []

    for trial in range(1, num_trials + 1):
        sample = rng.choice(score_values, size=sample_size, replace=False)
        sample_mean = sample.mean()
        standard_error = sample.std(ddof=1) / np.sqrt(sample_size)
        margin = z_score * standard_error
        lower = sample_mean - margin
        upper = sample_mean + margin
        contains_mean = lower <= population_mean <= upper
        rows.append(
            {
                "trial": trial,
                "sample_mean": sample_mean,
                "lower": lower,
                "upper": upper,
                "length": upper - lower,
                "contains_mean": contains_mean,
            }
        )

    return pd.DataFrame(rows)


def calculate_single_sample_interval(
    scores: pd.Series,
    sample_size: int,
    confidence_level: float,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    sample = rng.choice(scores.to_numpy(), size=sample_size, replace=False)
    sample_mean = sample.mean()
    sample_std = sample.std(ddof=1)
    standard_error = sample_std / np.sqrt(sample_size)
    z_score = stats.norm.ppf(1 - (1 - confidence_level / 100) / 2)
    margin = z_score * standard_error
    lower = sample_mean - margin
    upper = sample_mean + margin
    population_mean = scores.mean()

    return {
        "sample": sample,
        "sample_mean": sample_mean,
        "sample_std": sample_std,
        "standard_error": standard_error,
        "lower": lower,
        "upper": upper,
        "length": upper - lower,
        "population_mean": population_mean,
        "contains_mean": lower <= population_mean <= upper,
    }


def make_single_sample_interval_plot(
    result: dict,
    confidence_level: float,
    bins: int,
    xlim: tuple[float, float],
    show_curve: bool,
):
    fig, ax = plt.subplots(figsize=(10, 4.8))
    color = "#2ecc71" if result["contains_mean"] else "#e74c3c"

    sns.histplot(
        result["sample"],
        bins=bins,
        kde=show_curve and len(result["sample"]) >= 5,
        color="#95a5a6",
        edgecolor="white",
        alpha=0.65,
        ax=ax,
    )
    ax.axvspan(
        result["lower"],
        result["upper"],
        color=color,
        alpha=0.16,
        label=f"{confidence_level:.1f}% 신뢰구간",
    )
    ax.axvline(result["lower"], color=color, linestyle=":", linewidth=2, label=f"하한 ({result['lower']:.1f}점)")
    ax.axvline(result["upper"], color=color, linestyle=":", linewidth=2, label=f"상한 ({result['upper']:.1f}점)")
    ax.axvline(
        result["sample_mean"],
        color="#2980b9",
        linestyle="-",
        linewidth=2,
        label=f"표본평균 ({result['sample_mean']:.1f}점)",
    )
    ax.axvline(
        result["population_mean"],
        color="#f1c40f",
        linestyle="--",
        linewidth=2,
        label=f"전체 평균 ({result['population_mean']:.1f}점)",
    )
    ax.set_title("단일 표본으로 만든 신뢰구간")
    ax.set_xlabel("점수")
    ax.set_ylabel("표본 학생 수")
    ax.set_xlim(*xlim)
    ax.legend()
    fig.tight_layout()
    return fig


def make_confidence_interval_plot(
    interval_df: pd.DataFrame,
    population_mean: float,
    confidence_level: float,
    xlim: tuple[float, float],
):
    fig_width = max(10.0, min(28.0, 0.12 * len(interval_df) + 4.0))
    fig, ax = plt.subplots(figsize=(fig_width, 5.2))

    for _, row in interval_df.iterrows():
        color = "#2ecc71" if row["contains_mean"] else "#e74c3c"
        ax.vlines(row["trial"], row["lower"], row["upper"], color=color, linewidth=1.8)
        ax.plot(row["trial"], row["sample_mean"], marker="o", color=color, markersize=3.8)

    ax.axhline(
        population_mean,
        color="#f1c40f",
        linestyle="--",
        linewidth=2,
        label=f"전체 평균 ({population_mean:.1f}점)",
    )
    ax.set_ylim(*xlim)
    ax.set_xlim(0.5, len(interval_df) + 0.5)
    ax.set_title(f"{confidence_level:.1f}% 신뢰구간 반복 시뮬레이션")
    ax.set_xlabel("시행 번호")
    ax.set_ylabel("점수")
    ax.legend()
    fig.tight_layout()
    return fig


def render_scrollable_figure(fig) -> None:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    encoded = base64.b64encode(buffer.getvalue()).decode()
    st.markdown(
        f"""
        <div style="overflow-x: auto; border: 1px solid #e5e7eb; border-radius: 8px; padding: 8px;">
            <img src="data:image/png;base64,{encoded}" style="max-width: none; height: 520px;">
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    set_korean_font()

    with st.sidebar:
        st.header("⚙️ 분석 설정")
        uploaded_file = st.file_uploader("성적 파일 업로드", type=["xlsx", "csv"])
        use_sample = st.toggle("가상 예시 데이터 사용", value=uploaded_file is None)
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

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "1단계: 모집단 성적 분석",
            "2단계: 중심극한정리 시뮬레이션",
            "3단계: 신뢰구간 시뮬레이션",
            "4단계: 단일 표본 신뢰구간",
        ]
    )

    with tab1:
        st.subheader(f"`{score_column}` 열 분석 결과")
        render_summary(scores)

        top_left, top_right = st.columns(2)
        with top_left:
            st.markdown("#### 성적 분포")
            st.pyplot(make_histogram(scores, bins), use_container_width=True)

        with top_right:
            st.markdown("#### 정규성 확인")
            st.pyplot(make_qq_plot(scores), use_container_width=True)

        bottom_left, bottom_right = st.columns([1.05, 0.95])
        with bottom_left:
            st.markdown("#### 성적 밀집도와 이상치")
            st.pyplot(make_boxplot(scores), use_container_width=True)

        with bottom_right:
            p_value, normality_message = calculate_normality(scores)
            st.markdown("#### 발표 요약")
            st.write(
                f"이 데이터는 평균 **{scores.mean():.1f}점**을 중심으로 분포하며, "
                f"표준편차는 **{scores.std():.1f}점**입니다."
            )
            if p_value is not None:
                st.write(f"Shapiro-Wilk 정규성 검정 p-value는 **{p_value:.4f}**입니다.")
            st.info(normality_message)

            with st.expander("데이터 미리보기"):
                st.dataframe(df.head(20), use_container_width=True)

    with tab2:
        st.subheader("표본평균 시뮬레이터")
        st.write(
            "전체 성적 데이터에서 무작위로 일부 학생을 뽑아 표본평균을 구하고, "
            "이 과정을 반복했을 때 표본평균들이 어떤 분포를 이루는지 확인합니다. "
            "슬라이더를 조정하면 현재 값으로 그래프가 자동 갱신됩니다."
        )

        control_col, result_col = st.columns([0.9, 2.1])

        with control_col:
            st.markdown("#### 조작 패널")
            if "clt_trial_mode" not in st.session_state:
                st.session_state.clt_trial_mode = "slider"

            max_sample_size = min(len(scores), 50)
            default_sample_size = min(30, max_sample_size)
            sample_size = st.slider(
                "한 번에 뽑을 학생 수 (표본 크기 n)",
                min_value=1,
                max_value=max_sample_size,
                value=default_sample_size,
            )

            mode_col1, mode_col2, mode_col3 = st.columns(3)
            with mode_col1:
                if st.button("1회", use_container_width=True):
                    st.session_state.clt_trial_mode = "single"
            with mode_col2:
                if st.button("10000회", use_container_width=True):
                    st.session_state.clt_trial_mode = "many"
            with mode_col3:
                if st.button("슬라이더", use_container_width=True):
                    st.session_state.clt_trial_mode = "slider"

            trial_slider_disabled = st.session_state.clt_trial_mode != "slider"
            slider_trials = st.slider(
                "반복 추출 횟수",
                min_value=10,
                max_value=1000,
                value=500,
                step=10,
                disabled=trial_slider_disabled,
            )
            if st.session_state.clt_trial_mode == "single":
                num_trials = 1
                st.caption("현재 모드: 1회 보기. 반복 횟수 슬라이더가 비활성화됩니다.")
            elif st.session_state.clt_trial_mode == "many":
                num_trials = 10000
                st.caption("현재 모드: 10000회 보기. 반복 횟수 슬라이더가 비활성화됩니다.")
            else:
                num_trials = slider_trials

            seed = st.number_input("난수 시드", min_value=0, max_value=9999, value=42, step=1)
            clt_bins = st.slider("표본평균 히스토그램 구간", min_value=5, max_value=40, value=15)
            auto_x_axis = st.toggle("x축 자동 조절", value=True)
            x_axis_range = None
            if not auto_x_axis:
                x_min = st.slider("x축 최소값", min_value=0, max_value=100, value=0)
                x_max = st.slider("x축 최대값", min_value=0, max_value=100, value=100)
                if x_min >= x_max:
                    st.warning("x축 최소값은 최대값보다 작아야 합니다. 기본 범위 0~100점으로 표시합니다.")
                    x_axis_range = (0, 100)
                else:
                    x_axis_range = (x_min, x_max)
            st.caption("슬라이더를 조정하면 오른쪽 그래프가 자동으로 갱신됩니다.")

        sample_means = simulate_sample_means(scores, sample_size, num_trials, int(seed))
        population_mean = scores.mean()
        sample_means_std = sample_means.std(ddof=1) if len(sample_means) >= 2 else 0.0
        x_axis_label = "자동" if x_axis_range is None else f"{x_axis_range[0]}~{x_axis_range[1]}점"

        with result_col:
            st.caption(
                f"현재 설정: `{score_column}` 열, n={sample_size}, {num_trials}회, 시드={seed}. "
                f"x축: {x_axis_label}. 슬라이더를 조정하면 그래프가 자동으로 갱신됩니다."
            )
            st.pyplot(
                make_clt_plot(
                    sample_means,
                    population_mean,
                    sample_size,
                    num_trials,
                    clt_bins,
                    x_axis_range,
                ),
                use_container_width=True,
            )

            result_col1, result_col2, result_col3 = st.columns(3)
            result_col1.metric("전체 평균", f"{population_mean:.1f}점")
            result_col2.metric("표본평균들의 평균", f"{sample_means.mean():.1f}점")
            result_col3.metric("표본평균들의 표준편차", f"{sample_means_std:.2f}")

            st.info(
                f"표본 크기 **n={sample_size}**로 {num_trials}번 반복하면 표본평균들이 "
                f"전체 평균 **{population_mean:.1f}점** 주변에 모입니다. "
                "표본 크기를 키울수록 분포가 더 좁고 뾰족해지는지 비교해 보세요."
            )

        with st.expander("발표 조작 예시"):
            st.markdown(
                """
                1. 먼저 표본 크기를 `3`, 반복 횟수를 `50`으로 맞추고 슬라이더를 놓아 표본평균이 크게 흔들리는 모습을 보여줍니다.
                2. `x축 자동 조절`을 끄고 x축을 `0~100점`으로 고정하면 분포가 얼마나 넓게 퍼졌는지 비교하기 쉽습니다.
                3. 다음으로 표본 크기를 `30`, 반복 횟수를 `500`으로 늘려 그래프가 전체 평균 주변으로 모이는 모습을 비교합니다.
                4. 결론으로 표본 하나하나는 불안정할 수 있지만, 충분한 크기의 표본평균은 예측 가능한 분포를 만든다고 설명합니다.
                """
            )

    with tab3:
        st.subheader("신뢰구간 반복 시뮬레이터")
        if len(scores) < 2:
            st.warning("신뢰구간 시뮬레이션은 최소 2개 이상의 점수 데이터가 필요합니다.")
            return

        st.write(
            "같은 방식으로 표본을 여러 번 뽑아 신뢰구간을 만들고, 각 구간이 실제 전체 평균을 포함하는지 확인합니다. "
            "초록색 구간은 전체 평균을 포함하고, 빨간색 구간은 포함하지 못한 경우입니다."
        )

        ci_control_col, ci_result_col = st.columns([0.9, 2.1])

        with ci_control_col:
            st.markdown("#### 조작 패널")
            ci_max_sample_size = min(len(scores), 100)
            ci_default_sample_size = min(30, ci_max_sample_size)
            ci_sample_size = st.slider(
                "표본 크기 n",
                min_value=2,
                max_value=ci_max_sample_size,
                value=max(2, ci_default_sample_size),
                key="ci_sample_size",
            )
            ci_trials = st.slider("반복 횟수", min_value=20, max_value=200, value=100, step=10, key="ci_trials")
            ci_confidence = st.slider(
                "신뢰수준 (%)",
                min_value=50.0,
                max_value=99.9,
                value=95.0,
                step=0.5,
                key="ci_confidence",
            )
            ci_seed = st.number_input("난수 시드", min_value=0, max_value=9999, value=7, step=1, key="ci_seed")
            ci_x_min = st.slider("점수축 최소값", min_value=0, max_value=100, value=40, key="ci_x_min")
            ci_x_max = st.slider("점수축 최대값", min_value=0, max_value=100, value=100, key="ci_x_max")
            if ci_x_min >= ci_x_max:
                st.warning("점수축 최소값은 최대값보다 작아야 합니다. 기본 범위 40~100점으로 표시합니다.")
                ci_x_range = (40, 100)
            else:
                ci_x_range = (ci_x_min, ci_x_max)
            st.caption("슬라이더를 조정하면 오른쪽 신뢰구간 차트가 자동으로 갱신됩니다.")

        interval_df = simulate_confidence_intervals(
            scores,
            ci_sample_size,
            ci_trials,
            ci_confidence,
            int(ci_seed),
        )
        population_mean = scores.mean()
        hit_count = int(interval_df["contains_mean"].sum())
        hit_rate = hit_count / len(interval_df) * 100
        average_length = interval_df["length"].mean()

        with ci_result_col:
            st.caption(
                f"현재 설정: n={ci_sample_size}, {ci_trials}회, 신뢰수준={ci_confidence:.1f}%, "
                f"점수축={ci_x_range[0]}~{ci_x_range[1]}점. 0점은 아래, 100점은 위쪽 방향입니다."
            )
            fig = make_confidence_interval_plot(interval_df, population_mean, ci_confidence, ci_x_range)
            render_scrollable_figure(fig)

            ci_metric1, ci_metric2, ci_metric3, ci_metric4 = st.columns(4)
            ci_metric1.metric("전체 평균", f"{population_mean:.1f}점")
            ci_metric2.metric("성공 구간", f"{hit_count}/{len(interval_df)}개")
            ci_metric3.metric("실제 포함 비율", f"{hit_rate:.1f}%")
            ci_metric4.metric("평균 구간 길이", f"{average_length:.1f}점")

            st.info(
                f"{ci_confidence:.1f}% 신뢰수준은 같은 방법을 반복했을 때 구간이 실제 평균을 포함하는 비율을 뜻합니다. "
                "표본 크기 n을 키우면 구간 길이가 짧아져 추정이 더 정밀해지는지 확인해 보세요."
            )

        with st.expander("발표 조작 예시"):
            st.markdown(
                """
                1. 신뢰수준을 `95%`, 반복 횟수를 `100`으로 두고 초록색 구간과 빨간색 구간의 비율을 확인합니다.
                2. 점수축은 아래가 낮은 점수, 위가 높은 점수입니다. 시행 번호는 가로 방향으로 나열됩니다.
                3. 표본 크기 `n`을 키우면 세로 신뢰구간 길이가 짧아지는지 확인합니다.
                4. 신뢰수준을 높이면 구간 길이가 길어지지만 성공 비율이 높아지는 경향을 비교합니다.
                5. 결론으로 신뢰구간은 한 번의 구간이 맞을 확률이 아니라, 반복 절차의 성공률을 뜻한다고 설명합니다.
                """
            )

    with tab4:
        st.subheader("단일 표본 신뢰구간")
        if len(scores) < 2:
            st.warning("단일 표본 신뢰구간은 최소 2개 이상의 점수 데이터가 필요합니다.")
            return

        st.write(
            "현실에서는 보통 표본을 한 번만 뽑고, 그 표본의 평균과 표준편차로 신뢰구간을 계산합니다. "
            "이 탭은 단 한 번 뽑은 표본의 히스토그램 위에 신뢰구간과 실제 전체 평균을 함께 표시합니다."
        )

        single_control_col, single_result_col = st.columns([0.9, 2.1])

        with single_control_col:
            st.markdown("#### 조작 패널")
            single_max_sample_size = min(len(scores), 100)
            single_default_sample_size = min(30, single_max_sample_size)
            single_sample_size = st.slider(
                "표본 크기 n",
                min_value=2,
                max_value=single_max_sample_size,
                value=max(2, single_default_sample_size),
                key="single_sample_size",
            )
            single_confidence = st.slider(
                "신뢰수준 (%)",
                min_value=50.0,
                max_value=99.9,
                value=95.0,
                step=0.5,
                key="single_confidence",
            )
            single_seed = st.number_input("난수 시드", min_value=0, max_value=9999, value=21, step=1, key="single_seed")
            single_bins = st.slider("히스토그램 구간", min_value=5, max_value=30, value=10, key="single_bins")
            show_single_curve = st.toggle("곡선 표시", value=True, key="show_single_curve")
            single_x_min = st.slider("점수축 최소값", min_value=0, max_value=100, value=0, key="single_x_min")
            single_x_max = st.slider("점수축 최대값", min_value=0, max_value=100, value=100, key="single_x_max")
            if single_x_min >= single_x_max:
                st.warning("점수축 최소값은 최대값보다 작아야 합니다. 기본 범위 0~100점으로 표시합니다.")
                single_x_range = (0, 100)
            else:
                single_x_range = (single_x_min, single_x_max)
            st.caption("시드를 바꾸면 새로운 단일 표본을 확인할 수 있습니다.")

        single_result = calculate_single_sample_interval(
            scores,
            single_sample_size,
            single_confidence,
            int(single_seed),
        )

        with single_result_col:
            status_text = "포함" if single_result["contains_mean"] else "미포함"
            st.caption(
                f"현재 설정: n={single_sample_size}, 신뢰수준={single_confidence:.1f}%, "
                f"시드={int(single_seed)}, 실제 평균 {status_text}"
            )
            st.pyplot(
                make_single_sample_interval_plot(
                    single_result,
                    single_confidence,
                    single_bins,
                    single_x_range,
                    show_single_curve,
                ),
                use_container_width=True,
            )

            single_metric1, single_metric2, single_metric3, single_metric4 = st.columns(4)
            single_metric1.metric("표본평균", f"{single_result['sample_mean']:.1f}점")
            single_metric2.metric("표본표준편차", f"{single_result['sample_std']:.1f}")
            single_metric3.metric("신뢰구간 길이", f"{single_result['length']:.1f}점")
            single_metric4.metric("실제 평균 포함", status_text)

            interval_col1, interval_col2, interval_col3 = st.columns(3)
            interval_col1.metric("신뢰구간 하한", f"{single_result['lower']:.1f}점")
            interval_col2.metric("신뢰구간 상한", f"{single_result['upper']:.1f}점")
            interval_col3.metric("전체 평균", f"{single_result['population_mean']:.1f}점")

            if single_result["contains_mean"]:
                st.success("이번 표본으로 만든 신뢰구간 안에 실제 전체 평균이 들어 있습니다.")
            else:
                st.error("이번 표본으로 만든 신뢰구간은 실제 전체 평균을 포함하지 못했습니다.")

            st.info(
                "3단계는 이 절차를 여러 번 반복했을 때의 성공률을 보여주고, "
                "4단계는 현실처럼 표본을 한 번만 뽑아 하나의 신뢰구간을 만드는 장면을 보여줍니다."
            )

        with st.expander("발표 조작 예시"):
            st.markdown(
                """
                1. 표본 크기와 신뢰수준을 정한 뒤, 한 번 뽑은 표본의 히스토그램을 봅니다.
                2. `곡선 표시`를 켜면 표본 분포의 부드러운 모양을 함께 볼 수 있습니다.
                3. 파란 선은 표본평균, 노란 점선은 실제 전체 평균입니다.
                4. 색칠된 신뢰구간 안에 노란 점선이 들어오면 이번 표본은 성공한 사례입니다.
                5. 난수 시드를 바꾸면 다른 한 번의 표본 추출 결과를 확인할 수 있습니다.
                """
            )


if __name__ == "__main__":
    main()
