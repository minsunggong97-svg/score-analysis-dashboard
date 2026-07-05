# 성적 분석 대시보드

성적 데이터를 업로드해 히스토그램, 상자수염그림, Q-Q Plot을 확인할 수 있는 Streamlit 대시보드입니다.

## 주요 기능

- 엑셀 `.xlsx` 또는 CSV 파일 업로드
- 사이드바에서 점수 열과 히스토그램 구간 선택
- 평균, 표준편차, 중앙값, 사분위수 확인
- 히스토그램과 밀도 곡선 표시
- 상자수염그림으로 이상치 확인
- Q-Q Plot과 Shapiro-Wilk 검정으로 정규성 확인
- 파일 없이도 가상 예시 데이터로 미리보기
- 발표 전후에 활용할 수 있는 `presentation.html` 슬라이드 포함

## 로컬 실행 방법

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 발표 슬라이드

`presentation.html` 파일을 브라우저로 열면 발표용 슬라이드를 볼 수 있습니다.

## Streamlit Community Cloud 배포

1. 이 폴더를 GitHub 저장소에 업로드합니다.
2. [Streamlit Community Cloud](https://share.streamlit.io/)에 접속합니다.
3. GitHub 계정으로 로그인합니다.
4. `Create app`을 누릅니다.
5. Repository를 선택하고 Main file path에 `app.py`를 입력합니다.
6. `Deploy!`를 누릅니다.

## 업로드 파일 형식

엑셀 또는 CSV 파일 안에 숫자로 된 점수 열이 하나 이상 있으면 됩니다.

예시:

| 번호 | 이름 | 반 | 점수 |
| --- | --- | --- | --- |
| 1 | 학생001 | 1반 | 82 |
| 2 | 학생002 | 1반 | 76 |
