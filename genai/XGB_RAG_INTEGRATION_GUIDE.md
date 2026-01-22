# XGBoost + RAG 통합 가이드

## 🎯 개요

XGBoost 예측 모델과 RAG 생성 모델을 결합하여 **정량적 예측 + 정성적 분석**을 동시에 제공합니다.

```
┌─────────────────────┐
│   XGBoost 모델      │
│ - 성장 확률: 82%    │
│ - SHAP Top 10       │ ──┐
│ - 주요 기여 피처    │   │
└─────────────────────┘   │
                          ▼
                    ┌──────────────────┐
                    │  RAG 프롬프트    │
                    │  (XGBoost 결과   │
                    │   + 질문)        │
                    └──────────────────┘
                          │
                          ▼
                    ┌──────────────────┐
                    │   RAG 모델       │
                    │ - 특허 검색      │
                    │ - 재무 검색      │
                    │ - LLM 생성       │
                    └──────────────────┘
                          │
                          ▼
              ┌────────────────────────┐
              │  통합 분석 보고서      │
              │ 1. 성장 가능성 요약    │
              │ 2. 기술적 강점 (특허)  │
              │ 3. 재무적 안정성       │
              │ 4. 리스크 분석         │
              │ + XGBoost SHAP 해석   │
              └────────────────────────┘
```

---

## 📦 필요한 패키지 설치

```bash
pip install shap
```

---

## 🚀 사용 방법

### 1️⃣ 기본 사용 (rag_model.py 실행)

```bash
cd /Users/roychoi/Documents/Github/sesac_project/Growth-Company-Prediction
source venv/bin/activate
python genai/rag_model.py
```

**결과:**
- XGBoost가 자동으로 "비츠로셀" 기업 분석
- SHAP 값으로 주요 기여 피처 추출
- RAG가 XGBoost 결과를 반영한 종합 보고서 생성

---

### 2️⃣ 특정 기업 분석 (Python 코드)

```python
from genai.xgb_integration import analyze_company_with_xgb_and_rag
from genai.company_search import search_by_company, format_company_context
from genai.embedding import load_vector_db
from genai.rag_chain import create_llm
from genai.prompt import create_reviewer_prompt

# 1. XGBoost + SHAP 분석
company_name = "비츠로셀"
year = 2023

xgb_result, shap_contributions, rag_prompt = analyze_company_with_xgb_and_rag(
    company_name=company_name,
    year=year,
    top_n_features=10
)

print(f"성장 확률: {xgb_result['growth_probability']:.2%}")
print(f"주요 기여 피처:")
for contrib in shap_contributions[:5]:
    print(f"  - {contrib['feature']}: {contrib['impact']} (SHAP: {contrib['shap_value']:+.4f})")

# 2. RAG 분석
vector_db = load_vector_db('genai/chroma_db', collection_name='company_data')
patent_docs, financial_docs = search_by_company(vector_db, company_name, k=50)
context = format_company_context(patent_docs, financial_docs)

# 3. 보고서 생성
llm = create_llm(model_name="gpt-4o", temperature=0)
prompt_template = create_reviewer_prompt()
full_prompt = prompt_template.format(context=context, question=rag_prompt)

report = llm.invoke(full_prompt).content
print("\n" + "="*70)
print(report)
```

---

### 3️⃣ XGBoost 없이 RAG만 사용

`rag_model.py` 수정:

```python
use_xgboost = False  # False로 변경
```

---

## 📊 출력 예시

### XGBoost 예측 결과

```
✓ XGBoost 모델 로드: .../xgboost_model.json

✓ XGBoost 예측 완료:
  성장 확률: 82.34%
  예측: 상위 30%

✓ SHAP 분석 완료 (상위 10개 피처)
  1. rnd_intensity: 긍정적 (SHAP: +0.2341)
  2. patent_count: 긍정적 (SHAP: +0.1823)
  3. operating_margin: 부정적 (SHAP: -0.0912)
  4. debt_ratio: 부정적 (SHAP: -0.0745)
  5. citation_count: 긍정적 (SHAP: +0.0634)
```

### RAG 질문 (XGBoost 결과 포함)

```
비츠로셀의 기술적 강점과 재무 안정성을 종합적으로 분석하여 성장 가능성을 평가해주세요.

[XGBoost 예측 결과]
기업명: 비츠로셀
성장 확률: 82.34%
예측 결과: 상위 30% 성장 예측

[주요 기여 피처 (SHAP 분석)]
1. R&D 집약도 (rnd_intensity): 0.0850 → 긍정적 기여 (SHAP: +0.2341)
2. 특허 건수 (patent_count): 18.0000 → 긍정적 기여 (SHAP: +0.1823)
3. 영업이익률 (operating_margin): 0.2139 → 부정적 기여 (SHAP: -0.0912)
...

위의 XGBoost 모델 예측 결과를 참고하여, 주요 기여 피처와 관련된 특허 및 재무 정보를 중심으로 분석해주세요.
```

### RAG 생성 보고서

```
## 1. 성장 가능성 요약
- XGBoost 모델은 비츠로셀의 성장 확률을 82.34%로 예측하였으며, 이는 상위 30%에 속하는 높은 성장 가능성을 시사합니다. (근거: XGBoost 예측)
- 주요 긍정 요인으로는 R&D 집약도(0.085)와 특허 건수(18건)가 있으며, 이는 기술 혁신에 대한 지속적인 투자를 보여줍니다. (근거: SHAP 분석)
- 부정 요인으로는 영업이익률(21.39%)이 SHAP 분석에서 부정적 기여를 보였으나, 절대값은 양호한 수준입니다. (근거: SHAP 분석, 재무정보 #1)

## 2. 기술적 강점 분석 (특허 중심)
### 2.1 핵심 특허 기술력
- SHAP 분석에서 특허 건수가 두 번째로 큰 긍정적 기여를 하였으며(SHAP: +0.1823), 실제로 비츠로셀은 리튬 전지 관련 18건의 특허를 보유하고 있습니다. (근거: SHAP 분석, 특허 #1-#18)
- 주요 특허 중 '리튬 일차전지의 양극 제조방법'은 피인용 5회로 높은 기술 영향력을 보이고 있습니다. (근거: 특허 #1)
...
```

---

## 🔧 커스터마이징

### 1. XGBoost 모델 경로 변경

```python
from genai.xgb_integration import analyze_company_with_xgb_and_rag

xgb_result, shap_contributions, rag_prompt = analyze_company_with_xgb_and_rag(
    company_name="비츠로셀",
    year=2023,
    model_path="/custom/path/to/xgboost_model.json",  # 커스텀 경로
    top_n_features=15  # SHAP Top 15 피처
)
```

### 2. 프롬프트 커스터마이징

`genai/xgb_integration.py`의 `create_integrated_prompt` 함수 수정:

```python
def create_integrated_prompt(company_name, xgb_result, shap_contributions):
    # 여기서 프롬프트 형식 변경
    prompt = f"""
    [커스텀 프롬프트]
    {company_name}에 대한 XGBoost 예측:
    - 성장 확률: {xgb_result['growth_probability']:.2%}
    - 주요 피처: ...
    """
    return prompt
```

### 3. 피처명 한글화 확장

`genai/xgb_integration.py`의 `translate_feature_name` 함수에 피처 추가:

```python
translation_dict = {
    'revenue': '매출액',
    'operating_profit': '영업이익',
    'your_feature': '당신의 피처',  # 추가
    # ...
}
```

---

## 💡 주요 장점

### 1. **정량적 근거 제공**
- XGBoost: "82% 확률로 성장 예상"
- RAG: "왜 그런가?" (특허, 재무 데이터 기반 설명)

### 2. **SHAP 해석 통합**
- XGBoost가 어떤 피처를 중요하게 봤는지 명확히 제시
- RAG가 해당 피처와 관련된 실제 데이터 검색

### 3. **종합 분석**
- 정량적 예측 + 정성적 분석 = 완전한 보고서
- 금융 심사역이 의사결정에 필요한 모든 정보 제공

---

## 🐛 문제 해결

### 1. "모델 파일을 찾을 수 없습니다"

**원인:** `xgboost_model.json`이 없음

**해결:**
```bash
cd /Users/roychoi/Documents/Github/sesac_project/Growth-Company-Prediction
python model/XGBoost/XGBoost.py  # 모델 학습 및 저장
```

### 2. "shap 패키지가 설치되지 않았습니다"

**해결:**
```bash
pip install shap
```

### 3. "기업 데이터를 찾을 수 없습니다"

**원인:** `test_dataset_patent_feature_add.csv` 또는 `train_dataset_patent_feature_add.csv`가 없음

**해결:** 데이터 파일 경로 확인

---

## 📈 성능 최적화

### 1. SHAP 계산 캐싱

```python
import pickle

# SHAP 값 저장
with open('shap_cache.pkl', 'wb') as f:
    pickle.dump(shap_contributions, f)

# SHAP 값 로드
with open('shap_cache.pkl', 'rb') as f:
    shap_contributions = pickle.load(f)
```

### 2. 배치 처리

```python
companies = ["비츠로셀", "다른기업1", "다른기업2"]
results = []

for company in companies:
    result = analyze_company_with_xgb_and_rag(company, year=2023)
    results.append(result)
```

---

## 📝 라이선스

MIT License
