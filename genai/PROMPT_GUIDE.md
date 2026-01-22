# 기술금융 심사 프롬프트 가이드

## 필수 입력 데이터

### 1. **기본 정보** (필수)

```python
{
    "overview": "기업명 / 주요 사업 내용",
    "prediction": "모델 예측 결과 (성장 확률 등)"
}
```

### 2. **SHAP 특성** (필수)

```python
{
    "shap_features": [
        {
            "feature": "변수명 (설명)",
            "impact": "Positive" 또는 "Negative",
            "value": 기여도 (float)
        },
        # ... Top 5~10개 추천
    ]
}
```

**주요 특성 예시:**
- `rnd_intensity`: R&D 집약도 (R&D 투자액 / 매출액)
- `operating_profit_margin`: 영업이익률
- `patent_activity_score`: 특허 활동 점수
- `asset_growth_rate`: 자산 성장률
- `debt_ratio`: 부채비율
- `revenue_growth_rate`: 매출 성장률
- `current_ratio`: 유동비율

### 3. **재무지표** (필수)

```python
{
    "financial_metrics": {
        "매출액": "금액 (년도)",
        "영업이익": "금액 (영업이익률: %)",
        "당기순이익": "금액",
        "R&D 투자액": "금액 (매출 대비 %)",
        "자산총계": "금액",
        "부채비율": "%",
        "유동비율": "%"
    }
}
```

### 4. **특허 정보** (권장)

```python
{
    "patent_info": {
        "patent_count": 특허 수 (int),
        "ipc_codes": [
            {
                "code": "IPC 코드",
                "desc": "기술 분야 설명"
            },
            # ... 주요 IPC 3~5개
        ]
    }
}
```

**주요 IPC 코드 예시:**
- `G06N`: 인공지능/머신러닝
- `G05B`: 제어/자동화 시스템
- `G06Q`: 데이터 처리/비즈니스 로직
- `H01L`: 반도체 소자
- `A61K`: 의약용 제제
- `C12N`: 생명공학

### 5. **뉴스/이슈** (선택)

```python
{
    "news_list": [
        {
            "id": "뉴스 ID",
            "title": "뉴스 제목"
        },
        # ... 최근 3~5개
    ]
}
```

---

## 추가로 필요한 데이터

### ✅ **현재 데이터로 충분한 경우**
위 5가지 카테고리면 기본적인 기술금융 심사 보고서 생성 가능

### 🔍 **추가하면 좋은 데이터**

#### 1. **특허 상세 정보**
```python
{
    "patent_details": [
        {
            "patent_number": "특허번호",
            "title": "발명의 명칭",
            "registration_date": "등록일",
            "citation_count": 인용 횟수,  # 기술 영향력 지표
            "status": "등록/출원중"
        }
    ]
}
```

#### 2. **경쟁사 비교 데이터**
```python
{
    "industry_benchmark": {
        "industry": "산업 분류 (예: 스마트 팩토리)",
        "avg_rnd_ratio": "업계 평균 R&D 비율",
        "avg_operating_margin": "업계 평균 영업이익률",
        "company_rank": "업계 내 순위 (특허 수 기준 등)"
    }
}
```

#### 3. **시계열 데이터**
```python
{
    "trend_data": {
        "revenue_trend": [
            {"year": 2021, "value": 100},
            {"year": 2022, "value": 120},
            {"year": 2023, "value": 152}
        ],
        "rnd_trend": [...],
        "patent_trend": [...]
    }
}
```

#### 4. **투자 히스토리**
```python
{
    "investment_history": [
        {
            "date": "2024-03",
            "round": "Series A",
            "amount": "50억원",
            "investors": ["KB인베스트먼트", "..."]
        }
    ]
}
```

#### 5. **기술 평가 점수** (있는 경우)
```python
{
    "tech_evaluation": {
        "tech_grade": "T-3",  # 기술보증기금 등급
        "innovation_score": 85,  # 혁신성 점수
        "commercialization_score": 70  # 사업화 가능성
    }
}
```

#### 6. **주요 고객/계약**
```python
{
    "major_clients": [
        {
            "client": "현대자동차",
            "contract_type": "PoC (실증)",
            "contract_amount": "5억원",
            "period": "2024.01 ~ 2024.12"
        }
    ]
}
```

---

## 우선순위

### 🔴 필수 (현재 구현됨)
1. ✅ 기업 개요
2. ✅ 모델 예측 결과
3. ✅ SHAP 특성
4. ✅ 재무지표
5. ✅ 특허 정보

### 🟡 권장 (추가 시 품질 향상)
6. 뉴스/이슈
7. 특허 상세 정보 (인용 수 등)
8. 시계열 트렌드

### 🟢 선택 (있으면 더 좋음)
9. 경쟁사 비교
10. 투자 히스토리
11. 기술 평가 점수
12. 주요 고객/계약

---

## XGBoost/SHAP과 연동 방법

### 모델 출력에서 데이터 추출

```python
import shap

# 1. SHAP 값 계산
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

# 2. 특정 기업의 Top 특성 추출
def extract_shap_features(shap_values, feature_names, idx, top_n=5):
    """특정 기업의 SHAP Top N 특성 추출"""
    shap_row = shap_values[idx]
    
    # 절댓값 기준 정렬
    feature_importance = list(zip(feature_names, shap_row))
    sorted_features = sorted(feature_importance, 
                            key=lambda x: abs(x[1]), 
                            reverse=True)[:top_n]
    
    return [
        {
            "feature": name,
            "impact": "Positive" if value > 0 else "Negative",
            "value": float(value)
        }
        for name, value in sorted_features
    ]

# 3. 재무 데이터 추출
def extract_financial_metrics(df, company_idx):
    """재무지표 추출"""
    row = df.iloc[company_idx]
    return {
        "매출액": f"{row['매출액']/100000000:.0f}억원 ({row['연도']}년)",
        "영업이익": f"{row['영업이익']/100000000:.0f}억원 (영업이익률: {row['영업이익률']*100:.1f}%)",
        "R&D 투자액": f"{row['R&D투자액']/100000000:.0f}억원 (매출 대비 {row['rnd_intensity']*100:.1f}%)",
        # ...
    }

# 4. 특허 정보 추출 (CSV에서)
def extract_patent_info(patent_df, company_name):
    """특허 정보 추출"""
    company_patents = patent_df[patent_df['company_name'] == company_name]
    
    # IPC 코드 집계
    ipc_counts = company_patents['ipcNumber_IPC코드'].value_counts().head(3)
    
    return {
        "patent_count": len(company_patents),
        "ipc_codes": [
            {"code": code, "desc": get_ipc_description(code)}
            for code in ipc_counts.index
        ]
    }
```

---

## 사용 예시

```python
from genai.prompt import generate_tech_finance_prompt

# 데이터 준비
company_data = {
    "overview": f"{company_name} / {business_description}",
    "prediction": f"상위 30% 성장 확률: {probability*100:.0f}%",
    "shap_features": extract_shap_features(shap_values, feature_names, idx),
    "financial_metrics": extract_financial_metrics(df, idx),
    "patent_info": extract_patent_info(patent_df, company_name)
}

# 프롬프트 생성
system_instruction, user_context = generate_tech_finance_prompt(company_data)

# LLM 호출
response = llm.invoke(system_instruction + "\n\n" + user_context)
```
