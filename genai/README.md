# 🤖 GenAI - XGBoost + RAG 통합 시스템

기업 성장 가능성 분석을 위한 **정량적 예측(XGBoost) + 정성적 분석(RAG)** 통합 시스템

---

## 📂 파일 구조

```
genai/
├── rag_model.py              # 메인 실행 파일 (XGBoost + RAG 통합)
├── xgb_integration.py        # XGBoost ↔ RAG 통합 로직 ⭐ NEW
├── company_search.py         # 회사명 기반 특허/재무 검색
├── data_loader.py            # CSV 데이터 로드 및 변환
├── chunking.py               # 텍스트 청킹
├── embedding.py              # OpenAI 임베딩 & ChromaDB
├── prompt.py                 # 기술금융 심사역 프롬프트
├── rag_chain.py              # RAG 체인 구축 (LCEL)
├── metadata_filter.py        # 피인용횟수 기반 재정렬
├── chroma_db/                # ChromaDB 벡터 DB (17,406개 문서)
├── XGB_RAG_INTEGRATION_GUIDE.md  # 통합 가이드 ⭐ NEW
└── README.md                 # 이 파일
```

---

## 🔄 시스템 아키텍처

### 전체 흐름도

```
입력: 기업명 (예: "비츠로셀")
│
├─► [1] XGBoost 모델
│    ├─ 성장 확률 예측 (82%)
│    ├─ SHAP 값 계산
│    └─ 주요 기여 피처 추출
│         (rnd_intensity, patent_count 등)
│
├─► [2] RAG 시스템
│    ├─ ChromaDB 검색 (특허 18건, 재무 9건)
│    ├─ 피인용횟수 기반 재정렬
│    └─ 컨텍스트 생성
│
└─► [3] LLM (GPT-4o)
     ├─ XGBoost 결과 + RAG 컨텍스트
     └─ 종합 분석 보고서 생성
          ├─ 1. 성장 가능성 요약
          ├─ 2. 기술적 강점 (특허 중심)
          ├─ 3. 재무적 안정성
          └─ 4. 여신 심사 유의사항
```

---

## 🚀 빠른 시작

### 1. 환경 설정

```bash
cd /Users/roychoi/Documents/Github/sesac_project/Growth-Company-Prediction
source venv/bin/activate
pip install -r requirements.txt
```

### 2. 실행

```bash
# XGBoost + RAG 통합 분석
python genai/rag_model.py

# ChromaDB 재생성 (필요 시)
python genai/rag_model.py --recreate
```

### 3. 결과 확인

```
======================================================================
[1단계] CSV 데이터 로드 및 텍스트 변환
======================================================================
  총 17406개의 문서 생성 완료
  - 특허 정보: 13212건
  - 재무 정보: 4194건
  - 피인용 5회 이상 특허: 643건

======================================================================
[6단계] XGBoost + RAG 통합 분석
======================================================================

[XGBoost 모델 예측 수행]
✓ XGBoost 모델 로드: .../xgboost_model.json

✓ XGBoost 예측 완료:
  성장 확률: 82.34%
  예측: 상위 30%

✓ SHAP 분석 완료 (상위 10개 피처)
  1. rnd_intensity: 긍정적 (SHAP: +0.2341)
  2. patent_count: 긍정적 (SHAP: +0.1823)
  3. operating_margin: 부정적 (SHAP: -0.0912)
  ...

검색 결과: 특허 18건, 재무 9건

======================================================================
생성된 보고서:
======================================================================
## 1. 성장 가능성 요약
- XGBoost 모델은 비츠로셀의 성장 확률을 82.34%로 예측하였으며...
...
```

---

## 🎯 핵심 기능

### 1️⃣ XGBoost 정량적 예측

**기능:**
- 기업의 2024년 성장 확률 예측 (이진 분류)
- SHAP 값으로 주요 기여 피처 해석
- Top 10 피처의 긍정/부정 기여도 제공

**예시 출력:**
```
성장 확률: 82.34%
주요 기여 피처:
  📈 R&D 집약도: 긍정적 (+0.2341)
  📈 특허 건수: 긍정적 (+0.1823)
  📉 영업이익률: 부정적 (-0.0912)
```

---

### 2️⃣ RAG 정성적 분석

**기능:**
- ChromaDB에서 특허 및 재무 정보 검색
- 피인용횟수 기반 특허 우선순위 부여
- GPT-4o로 근거 기반 보고서 생성

**데이터 소스:**
- `특허정보_final_v2.csv` (13,212건)
- `재무정보_final_imputed.csv` (4,194건)

**프롬프트 구조:**
```
당신은 은행 본점 기술금융부의 전문 심사역입니다.

[작성 원칙]
1. 환각 방지: 데이터에 없는 사실을 지어내지 말 것
2. 근거 표기: (근거: 특허 #1, 재무정보 #2)
3. 특허 우선 분석: 피인용횟수 높은 특허 중심

[출력 양식]
1. 성장 가능성 요약
2. 기술적 강점 분석 (특허 중심)
   2.1 핵심 특허 기술력
   2.2 기술 분야 및 경쟁력 (IPC 코드)
   2.3 특허 포트폴리오 강도
   2.4 R&D 투자와의 연계
3. 재무적 안정성
4. 여신 심사 시 유의사항 (리스크)
```

---

### 3️⃣ 통합 분석

**XGBoost + RAG 시너지:**

| 측면     | XGBoost       | RAG            | 통합 효과            |
| -------- | ------------- | -------------- | -------------------- |
| **예측** | 82% 확률      | -              | ✅ 정량적 근거        |
| **해석** | SHAP 피처     | 실제 데이터    | ✅ 피처 → 데이터 매핑 |
| **특허** | patent_count  | 특허 18건 상세 | ✅ 숫자 → 내용 해석   |
| **재무** | rnd_intensity | R&D 73억원     | ✅ 비율 → 금액 명시   |
| **종합** | 모델 예측     | 근거 설명      | ✅ 완전한 보고서      |

**예시: R&D 집약도 분석**

1. **XGBoost**: `rnd_intensity = 0.085` → SHAP +0.2341 (가장 큰 긍정 기여)
2. **RAG 검색**: 재무정보에서 "2024년 연구개발비: 7,323,008,000원" 발견
3. **LLM 생성**: "SHAP 분석에서 R&D 집약도(0.085)가 가장 큰 긍정적 기여를 하였으며, 실제로 비츠로셀은 2024년 73억원을 R&D에 투자하였습니다. 이는 매출 대비 약 3.5%에 해당하며... (근거: SHAP 분석, 재무정보 #1)"

---

## 📊 데이터 흐름

### 입력 데이터

```python
# 1. XGBoost 피처 데이터
test_dataset_patent_feature_add.csv
├─ 기업명: 비츠로셀
├─ 연도: 2023
├─ 피처: rnd_intensity, patent_count, operating_margin, ...
└─ (50+ 피처)

# 2. RAG 소스 데이터
특허정보_final_v2.csv + 재무정보_final_imputed.csv
└─ ChromaDB 임베딩 (17,406개 문서)
```

### 처리 과정

```python
# Step 1: XGBoost 예측
xgb_result = {
    'growth_probability': 0.8234,
    'predicted_growth': 1,
    'feature_values': {...}
}

# Step 2: SHAP 분석
shap_contributions = [
    {'feature': 'rnd_intensity', 'shap_value': 0.2341, 'impact': '긍정적'},
    {'feature': 'patent_count', 'shap_value': 0.1823, 'impact': '긍정적'},
    ...
]

# Step 3: RAG 검색
patent_docs = [...18건...]
financial_docs = [...9건...]

# Step 4: 프롬프트 생성
prompt = f"""
{company_name}의 성장 가능성을 분석해주세요.

[XGBoost 예측 결과]
성장 확률: {xgb_result['growth_probability']:.2%}

[주요 기여 피처]
{format_shap_contributions(shap_contributions)}

위 XGBoost 결과를 참고하여 특허 및 재무 정보를 분석해주세요.
"""

# Step 5: LLM 생성
report = llm.invoke(prompt)
```

---

## 🛠️ 커스터마이징

### 1. 분석 대상 기업 변경

`genai/rag_model.py` 수정:

```python
company_name = "비츠로셀"  # ← 여기를 변경
year = 2023
```

### 2. XGBoost 사용 여부 설정

```python
use_xgboost = True  # False로 변경하면 RAG만 사용
```

### 3. SHAP Top N 피처 변경

```python
xgb_result, shap_contributions, rag_prompt = analyze_company_with_xgb_and_rag(
    company_name=company_name,
    year=year,
    top_n_features=15  # 10 → 15로 변경
)
```

### 4. 프롬프트 수정

`genai/prompt.py`의 `create_reviewer_prompt` 함수 수정

### 5. 검색 결과 개수 변경

```python
patent_docs, financial_docs = search_by_company(vector_db, company_name, k=100)  # 50 → 100
```

---

## 📖 상세 가이드

- **XGBoost + RAG 통합 가이드**: [`XGB_RAG_INTEGRATION_GUIDE.md`](./XGB_RAG_INTEGRATION_GUIDE.md)
- **프롬프트 가이드**: [`PROMPT_GUIDE.md`](./PROMPT_GUIDE.md) (기존)

---

## 🐛 문제 해결

### 1. XGBoost 모델 로드 실패

**오류:** `FileNotFoundError: 모델 파일을 찾을 수 없습니다`

**해결:**
```bash
cd /Users/roychoi/Documents/Github/sesac_project/Growth-Company-Prediction
python model/XGBoost/XGBoost.py  # 모델 학습 및 저장
```

### 2. SHAP 설치 오류

**해결:**
```bash
pip install shap
```

### 3. ChromaDB 오류

**해결:**
```bash
python genai/rag_model.py --recreate  # 벡터 DB 재생성
```

### 4. OpenAI API 키 오류

**해결:**
```bash
# .env 파일 생성
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

---

## 📈 향후 개선 사항

### 1. 배치 처리
```python
companies = ["비츠로셀", "기업A", "기업B"]
for company in companies:
    analyze_company_with_xgb_and_rag(company)
```

### 2. 웹 인터페이스
- FastAPI 백엔드
- React 프론트엔드
- 실시간 보고서 생성

### 3. 추가 데이터 소스
- 뉴스 데이터
- 경쟁사 분석
- 시장 트렌드

### 4. 모델 앙상블
- XGBoost + LightGBM + CatBoost
- 앙상블 예측 → RAG 입력

---

## 📝 라이선스

MIT License

---

## 👥 기여

이슈 및 PR 환영합니다!

---

**마지막 업데이트:** 2026-01-21
