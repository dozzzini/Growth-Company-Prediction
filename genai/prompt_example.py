"""
기술금융 심사 프롬프트 사용 예제
"""
from prompt import generate_tech_finance_prompt


# --- 실제 데이터 입력 예시 ---
example_data = {
    "overview": "(주)에이아이솔루션 / 스마트 팩토리 공정 최적화 AI 개발 기업",
    
    "prediction": "상위 30% 성장 확률: 82% (모델: XGBoost)",
    
    "shap_features": [
        {"feature": "rnd_intensity (R&D 집약도)", "impact": "Positive", "value": 0.85},
        {"feature": "operating_profit_margin (영업이익률)", "impact": "Negative", "value": -0.32},
        {"feature": "patent_activity_score (특허 활동 점수)", "impact": "Positive", "value": 0.45},
        {"feature": "asset_growth_rate (자산 성장률)", "impact": "Positive", "value": 0.38},
        {"feature": "debt_ratio (부채비율)", "impact": "Negative", "value": -0.25}
    ],
    
    "financial_metrics": {
        "매출액": "152억원 (2023년)",
        "영업이익": "-8억원 (영업이익률: -5.3%)",
        "당기순이익": "-12억원",
        "R&D 투자액": "23억원 (매출 대비 15.1%)",
        "자산총계": "89억원",
        "부채비율": "145%",
        "유동비율": "98%"
    },
    
    "patent_info": {
        "patent_count": 15,
        "ipc_codes": [
            {"code": "G06N", "desc": "컴퓨터 알고리즘 기반 예측 모델 (인공지능)"},
            {"code": "G05B", "desc": "제어 또는 조정 시스템 (스마트 팩토리)"},
            {"code": "G06Q", "desc": "데이터 처리 시스템 또는 방법 (비즈니스 로직)"}
        ]
    },
    
    "news_list": [
        {
            "id": "N-2024-001", 
            "title": "글로벌 자동차 제조사와 스마트 팩토리 실증 사업(PoC) 착수"
        },
        {
            "id": "N-2024-003", 
            "title": "Series A 투자 유치 50억원 완료 (주관: KB인베스트먼트)"
        },
        {
            "id": "N-2024-005", 
            "title": "핵심 인력 유출로 인한 기술 유출 우려 기사 보도"
        }
    ]
}


# 최소 데이터 예시 (뉴스 없음)
minimal_data = {
    "overview": "(주)바이오테크 / 신약 개발 바이오 기업",
    "prediction": "상위 30% 성장 확률: 65%",
    "shap_features": [
        {"feature": "rnd_intensity", "impact": "Positive", "value": 1.2},
        {"feature": "revenue_growth", "impact": "Positive", "value": 0.6}
    ],
    "financial_metrics": {
        "매출액": "20억원",
        "R&D 투자액": "35억원 (매출 대비 175%)"
    },
    "patent_info": {
        "patent_count": 8,
        "ipc_codes": [
            {"code": "A61K", "desc": "의약용 제제"}
        ]
    }
}


if __name__ == "__main__":
    # 프롬프트 생성
    system_instruction, user_context = generate_tech_finance_prompt(example_data)
    
    print("="*70)
    print("[시스템 지침]")
    print("="*70)
    print(system_instruction)
    
    print("\n" + "="*70)
    print("[사용자 컨텍스트]")
    print("="*70)
    print(user_context)
    
    print("\n" + "="*70)
    print("[최소 데이터 예시]")
    print("="*70)
    system_min, context_min = generate_tech_finance_prompt(minimal_data)
    print(context_min)
