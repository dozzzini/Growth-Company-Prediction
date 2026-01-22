"""
XGBoost 모델과 RAG 모델 통합
XGBoost의 예측 결과와 SHAP 값을 RAG에 전달하여 종합 보고서 생성
"""
import os
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
from typing import Dict, List, Tuple, Optional


def load_xgboost_model(model_path: Optional[str] = None) -> xgb.Booster:
    """
    저장된 XGBoost 모델 로드
    
    Args:
        model_path: 모델 파일 경로 (None이면 기본 경로 사용)
    
    Returns:
        XGBoost Booster 모델
    """
    if model_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        model_path = os.path.join(project_root, 'model', 'XGBoost', 'xgboost_model.json')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
    
    model = xgb.Booster()
    model.load_model(model_path)
    print(f"✓ XGBoost 모델 로드: {model_path}")
    
    return model


def get_company_features(company_name: str, year: int = 2023) -> Optional[pd.DataFrame]:
    """
    특정 기업의 피처 데이터 추출
    
    Args:
        company_name: 기업명
        year: 연도 (기본값: 2023)
    
    Returns:
        해당 기업의 피처 DataFrame
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # test_dataset 또는 train_dataset에서 로드
    test_path = os.path.join(project_root, 'data', 'test_dataset_patent_feature_add.csv')
    train_path = os.path.join(project_root, 'data', 'train_dataset_patent_feature_add.csv')
    
    df = None
    if os.path.exists(test_path):
        df = pd.read_csv(test_path, encoding='utf-8-sig')
    elif os.path.exists(train_path):
        df = pd.read_csv(train_path, encoding='utf-8-sig')
    else:
        print("경고: 피처 데이터 파일을 찾을 수 없습니다.")
        return None
    
    # 기업명과 연도로 필터링
    company_data = df[(df['기업명'] == company_name) & (df['연도'] == year)]
    
    if len(company_data) == 0:
        # 정확한 매칭 실패 시 부분 매칭 시도
        company_data = df[df['기업명'].str.contains(company_name, na=False) & (df['연도'] == year)]
    
    if len(company_data) == 0:
        print(f"경고: '{company_name}' 기업의 {year}년 데이터를 찾을 수 없습니다.")
        return None
    
    return company_data


def prepare_xgb_input(company_data: pd.DataFrame) -> Tuple[xgb.DMatrix, List[str]]:
    """
    XGBoost 모델 입력을 위한 데이터 준비
    
    Args:
        company_data: 기업 피처 데이터
    
    Returns:
        (DMatrix, 피처 컬럼 리스트)
    """
    # 제외할 컬럼
    exclude_cols = ['기업명', '연도', 'target_year', 'target_growth', 'target_top30', 'growth_rate']
    
    # 임베딩 컬럼 제외
    embedding_cols = [col for col in company_data.columns if 'patent_emb' in col.lower()]
    exclude_cols.extend(embedding_cols)
    
    # 피처 컬럼 추출
    feature_cols = [col for col in company_data.columns if col not in exclude_cols]
    X = company_data[feature_cols].copy()
    
    # DMatrix 생성
    dmatrix = xgb.DMatrix(X)
    
    return dmatrix, feature_cols


def predict_with_xgboost(
    model: xgb.Booster,
    company_name: str,
    year: int = 2023
) -> Optional[Dict]:
    """
    XGBoost 모델로 특정 기업의 성장 확률 예측
    
    Args:
        model: XGBoost 모델
        company_name: 기업명
        year: 연도
    
    Returns:
        예측 결과 딕셔너리 (확률, 피처값 등)
    """
    # 기업 데이터 로드
    company_data = get_company_features(company_name, year)
    
    if company_data is None:
        return None
    
    # XGBoost 입력 준비
    dmatrix, feature_cols = prepare_xgb_input(company_data)
    
    # 예측 수행
    growth_proba = model.predict(dmatrix)[0]
    growth_pred = 1 if growth_proba >= 0.4 else 0
    
    result = {
        'company_name': company_name,
        'year': year,
        'growth_probability': float(growth_proba),
        'predicted_growth': growth_pred,
        'feature_values': company_data[feature_cols].iloc[0].to_dict(),
        'feature_cols': feature_cols
    }
    
    return result


def calculate_shap_values(
    model: xgb.Booster,
    company_data: pd.DataFrame,
    feature_cols: List[str],
    top_n: int = 10
) -> List[Dict]:
    """
    SHAP 값을 계산하여 주요 기여 피처 추출
    
    Args:
        model: XGBoost 모델
        company_data: 기업 피처 데이터
        feature_cols: 피처 컬럼 리스트
        top_n: 상위 N개 피처
    
    Returns:
        SHAP 기여도 리스트 (피처명, 값, 기여도)
    """
    try:
        import shap
    except ImportError:
        print("경고: shap 패키지가 설치되지 않았습니다. pip install shap")
        return []
    
    # DMatrix 생성
    X = company_data[feature_cols].copy()
    dmatrix = xgb.DMatrix(X)
    
    # SHAP 값 계산
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # 절대값 기준 상위 N개 피처 추출
    shap_abs = np.abs(shap_values[0])
    top_indices = np.argsort(shap_abs)[::-1][:top_n]
    
    shap_contributions = []
    for idx in top_indices:
        feature_name = feature_cols[idx]
        feature_value = X.iloc[0, idx]
        shap_value = shap_values[0][idx]
        impact = "긍정적" if shap_value > 0 else "부정적"
        
        shap_contributions.append({
            'feature': feature_name,
            'value': float(feature_value),
            'shap_value': float(shap_value),
            'impact': impact,
            'abs_shap': float(abs(shap_value))
        })
    
    return shap_contributions


def format_xgb_insight_for_rag(
    xgb_result: Dict,
    shap_contributions: List[Dict]
) -> str:
    """
    XGBoost 예측 결과를 RAG 프롬프트에 적합한 형식으로 포맷팅
    
    Args:
        xgb_result: XGBoost 예측 결과
        shap_contributions: SHAP 기여도 리스트
    
    Returns:
        포맷팅된 텍스트
    """
    company = xgb_result['company_name']
    proba = xgb_result['growth_probability']
    pred = "상위 30% 성장 예측" if xgb_result['predicted_growth'] == 1 else "하위 70% 예측"
    
    text_parts = []
    text_parts.append(f"[XGBoost 예측 결과]")
    text_parts.append(f"기업명: {company}")
    text_parts.append(f"성장 확률: {proba:.2%}")
    text_parts.append(f"예측 결과: {pred}")
    text_parts.append("")
    
    if shap_contributions:
        text_parts.append("[주요 기여 피처 (SHAP 분석)]")
        for i, contrib in enumerate(shap_contributions, 1):
            feature = contrib['feature']
            value = contrib['value']
            impact = contrib['impact']
            shap_val = contrib['shap_value']
            
            # 피처명 한글화 (선택적)
            feature_kr = translate_feature_name(feature)
            
            text_parts.append(
                f"{i}. {feature_kr} ({feature}): {value:.4f} "
                f"→ {impact} 기여 (SHAP: {shap_val:+.4f})"
            )
    
    return "\n".join(text_parts)


def translate_feature_name(feature: str) -> str:
    """
    피처명을 한글로 번역 (주요 피처만)
    
    Args:
        feature: 영문 피처명
    
    Returns:
        한글 피처명
    """
    translation_dict = {
        'revenue': '매출액',
        'operating_profit': '영업이익',
        'net_income': '당기순이익',
        'total_assets': '총자산',
        'total_liabilities': '총부채',
        'equity': '자본총계',
        'rnd_investment': 'R&D 투자',
        'rnd_intensity': 'R&D 집약도',
        'capex': '설비투자',
        'patent_count': '특허 건수',
        'citation_count': '피인용 횟수',
        'patent_diversity': '특허 다양성',
        'operating_margin': '영업이익률',
        'net_margin': '순이익률',
        'roa': 'ROA',
        'roe': 'ROE',
        'debt_ratio': '부채비율',
        'current_ratio': '유동비율'
    }
    
    # 부분 매칭 시도
    for eng, kor in translation_dict.items():
        if eng.lower() in feature.lower():
            return kor
    
    return feature


def create_integrated_prompt(
    company_name: str,
    xgb_result: Dict,
    shap_contributions: List[Dict]
) -> str:
    """
    XGBoost 결과를 포함한 통합 프롬프트 생성
    
    Args:
        company_name: 기업명
        xgb_result: XGBoost 예측 결과
        shap_contributions: SHAP 기여도
    
    Returns:
        통합 프롬프트 텍스트
    """
    xgb_insight = format_xgb_insight_for_rag(xgb_result, shap_contributions)
    
    prompt = f"""
{company_name}의 기술적 강점과 재무 안정성을 종합적으로 분석하여 성장 가능성을 평가해주세요.

{xgb_insight}

위의 XGBoost 모델 예측 결과를 참고하여, 주요 기여 피처와 관련된 특허 및 재무 정보를 중심으로 분석해주세요.
특히 SHAP 분석에서 긍정적/부정적 기여를 한 피처들이 실제 데이터에서 어떻게 나타나는지 근거를 명시해주세요.
"""
    
    return prompt.strip()


# 사용 예제 함수
def analyze_company_with_xgb_and_rag(
    company_name: str,
    year: int = 2023,
    model_path: Optional[str] = None,
    top_n_features: int = 10
) -> Tuple[Dict, List[Dict], str]:
    """
    XGBoost와 RAG를 통합하여 기업 분석
    
    Args:
        company_name: 기업명
        year: 연도
        model_path: XGBoost 모델 경로
        top_n_features: 상위 N개 SHAP 피처
    
    Returns:
        (XGBoost 결과, SHAP 기여도, RAG 프롬프트)
    """
    # 1. XGBoost 모델 로드
    model = load_xgboost_model(model_path)
    
    # 2. XGBoost 예측
    xgb_result = predict_with_xgboost(model, company_name, year)
    
    if xgb_result is None:
        print(f"❌ '{company_name}' 기업의 예측을 수행할 수 없습니다.")
        return None, [], ""
    
    print(f"\n✓ XGBoost 예측 완료:")
    print(f"  성장 확률: {xgb_result['growth_probability']:.2%}")
    print(f"  예측: {'상위 30%' if xgb_result['predicted_growth'] == 1 else '하위 70%'}")
    
    # 3. SHAP 값 계산
    company_data = get_company_features(company_name, year)
    shap_contributions = calculate_shap_values(
        model,
        company_data,
        xgb_result['feature_cols'],
        top_n=top_n_features
    )
    
    if shap_contributions:
        print(f"\n✓ SHAP 분석 완료 (상위 {len(shap_contributions)}개 피처)")
        for i, contrib in enumerate(shap_contributions[:5], 1):
            print(f"  {i}. {contrib['feature']}: {contrib['impact']} (SHAP: {contrib['shap_value']:+.4f})")
    
    # 4. RAG 프롬프트 생성
    rag_prompt = create_integrated_prompt(company_name, xgb_result, shap_contributions)
    
    return xgb_result, shap_contributions, rag_prompt
