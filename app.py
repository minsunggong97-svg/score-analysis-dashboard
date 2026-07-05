import hashlib
import io
import platform
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import streamlit as st
from streamlit_extras.local_storage_manager import local_storage_manager


STORAGE_KEY = "score_dashboard_saved_tables"
SELECTED_STORAGE_KEY = "score_dashboard_selected_table_id"


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


def read_uploaded_file(file_name: str, file_bytes: bytes) -> pd.DataFrame:
    file_name = file_name.lower()
    if file_name.endswith(".csv"):
        return pd.read_csv(io.BytesIO(file_bytes))
    return pd.read_excel(io.BytesIO(file_bytes))


def make_storage_entry(file_name: str, file_bytes: bytes, df: pd.DataFrame) -> dict:
    file_hash = hashlib.sha256(file_bytes).hexdigest()
    return {
        "id": file_hash,
        "file_name": file_name,
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "row_count": int(len(df)),
        "columns": [str(column) for column in df.columns],
        "data_json": df.to_json(orient="split", date_format="iso", force_ascii=False),
    }


def restore_dataframe(entry: dict) -> pd.DataFrame:
    return pd.read_json(io.StringIO(entry["data_json"]), orient="split")


def format_saved_file_label(entry: dict) -> str:
    return f"{entry['file_name']} · {entry['row_count']}행 · {entry['saved_at']}"


def save_uploaded_files(uploaded_files: list, saved_entries: list[dict], storage) -> list[dict]:
    if "processed_upload_hashes" not in st.session_state:
        st.session_state.processed_upload_hashes = set()

    updated_entries = list(saved_entries)
    added_count = 0

    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.getvalue()
        file_hash = hashlib.sha256(file_bytes).hexdigest()
        if file_hash in st.session_state.processed_upload_hashes:
            continue

        st.session_state.processed_upload_hashes.add(file_hash)

        try:
            df = read_uploaded_file(uploaded_file.name, file_bytes)
        except Exception as error:
            st.sidebar.error(f"`{uploaded_file.name}` 파일을 읽지 못했습니다: {error}")
            continue

        entry = make_storage_entry(uploaded_file.name, file_bytes, df)
        updated_entries = [saved for saved in updated_entries if saved.get("id") != entry["id"]]
        updated_entries.insert(0, entry)
        storage[SELECTED_STORAGE_KEY] = entry["id"]
        added_count += 1

    if added_count:
        storage[STORAGE_KEY] = updated_entries
        st.sidebar.success(f"{added_count}개 파일 데이터를 브라우저에 저장했습니다.")

    return updated_entries


def delete_storage_key(storage, key: str) -> None:
    if key in storage:
        del storage[key]


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
    storage = local_storage_manager(key="score_dashboard_storage")

    if not storage.ready():
        st.info("브라우저 저장소를 불러오는 중입니다. 잠시 후 자동으로 준비됩니다.")
        st.stop()

    saved_entries = storage.get(STORAGE_KEY, [])
    if not isinstance(saved_entries, list):
        saved_entries = []

    with st.sidebar:
        st.header("⚙️ 분석 설정")
        uploaded_files = st.file_uploader(
            "성적 파일 업로드",
            type=["xlsx", "csv"],
            accept_multiple_files=True,
            help="여러 파일을 올리면 각 파일의 표 데이터가 이 브라우저에 따로 저장됩니다.",
        )
        if uploaded_files:
            st.session_state.use_sample_toggle = False
        saved_entries = save_uploaded_files(uploaded_files, saved_entries, storage)

        use_sample = st.toggle(
            "가상 110명 데이터 사용",
            value=not saved_entries and not uploaded_files,
            key="use_sample_toggle",
        )
        bins = st.slider("히스토그램 구간", min_value=5, max_value=30, value=15)
        preview_option = st.selectbox("데이터 미리보기 행 수", ["10", "20", "50", "전체"], index=1)

        st.divider()
        selected_entry = None
        if saved_entries:
            label_by_id = {entry["id"]: format_saved_file_label(entry) for entry in saved_entries}
            saved_ids = list(label_by_id)
            stored_selected_id = storage.get(SELECTED_STORAGE_KEY, saved_ids[0])
            selected_index = saved_ids.index(stored_selected_id) if stored_selected_id in saved_ids else 0
            selected_id = st.selectbox(
                "저장된 데이터 선택",
                saved_ids,
                index=selected_index,
                format_func=lambda entry_id: label_by_id[entry_id],
            )
            selected_entry = next(entry for entry in saved_entries if entry["id"] == selected_id)
            storage[SELECTED_STORAGE_KEY] = selected_id

            st.caption("저장 위치: 이 브라우저의 localStorage")
            delete_col1, delete_col2 = st.columns(2)
            with delete_col1:
                if st.button("선택 삭제", use_container_width=True):
                    saved_entries = [entry for entry in saved_entries if entry["id"] != selected_id]
                    storage[STORAGE_KEY] = saved_entries
                    if saved_entries:
                        storage[SELECTED_STORAGE_KEY] = saved_entries[0]["id"]
                    else:
                        delete_storage_key(storage, SELECTED_STORAGE_KEY)
                    st.rerun()
            with delete_col2:
                if st.button("전체 삭제", use_container_width=True):
                    delete_storage_key(storage, STORAGE_KEY)
                    delete_storage_key(storage, SELECTED_STORAGE_KEY)
                    st.rerun()
        else:
            st.caption("저장된 업로드 데이터가 없습니다.")

        st.info("발표할 때는 사이드바를 접으면 그래프를 더 크게 보여줄 수 있습니다.")
        st.warning("발표 후 성적 데이터가 남지 않도록 `전체 삭제`를 눌러주세요.")

    st.title("📊 확률과 통계 성적 분석 보고서")
    st.caption("성적 분포, 이상치, 정규성 여부를 한 화면에서 확인하는 발표용 대시보드입니다.")

    try:
        if use_sample:
            df = make_sample_scores()
            data_source = "가상 110명 데이터"
        elif selected_entry is not None:
            df = restore_dataframe(selected_entry)
            data_source = f"저장된 파일: {selected_entry['file_name']}"
        else:
            st.warning("사이드바에서 파일을 업로드하거나 가상 데이터를 켜주세요.")
            return
    except Exception as error:
        st.error(f"데이터를 불러오는 중 문제가 생겼습니다: {error}")
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

    st.success(f"현재 분석 중: {data_source}")
    if not use_sample:
        st.caption("새로고침해도 이 브라우저에 저장된 표 데이터로 다시 분석할 수 있습니다.")

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
            if preview_option == "전체":
                preview_df = df
            else:
                preview_df = df.head(int(preview_option))
            st.dataframe(preview_df, use_container_width=True)


if __name__ == "__main__":
    main()
