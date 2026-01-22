import pandas as pd
import numpy as np
import os
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, ParameterGrid, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix, 
                            classification_report, roc_curve, precision_recall_curve,
                            average_precision_score, brier_score_loss)
from sklearn.model_selection import GroupKFold
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import mlflow
import mlflow.sklearn
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def load_data(use_train_dataset=True):
    """데이터 로드
    
    Args:
        use_train_dataset: True이면 train_dataset_patent_feature_add.csv 사용, False이면 기존 방식 (growth_features_normalized.csv)
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    if use_train_dataset:
        # train_dataset_patent_feature_add.csv 사용
        print("[train_dataset_patent_feature_add.csv 사용 모드]")
        train_path = os.path.join(project_root, 'data', 'train_dataset_patent_feature_add.csv')
        features_df = pd.read_csv(train_path, encoding='utf-8-sig')
        print(f"train_dataset_patent_feature_add.csv 로드: {features_df.shape}")
        
        # target_growth를 target_top30으로 변환 (호환성을 위해)
        if 'target_growth' in features_df.columns:
            features_df['target_top30'] = features_df['target_growth']
        
        # financial_df는 None 반환 (더 이상 필요 없음)
        financial_df = None
    else:
        # 기존 방식: growth_features_normalized.csv 사용
        print("[기존 방식: growth_features_normalized.csv 사용 모드]")
        # 원본 재무정보 데이터 로드 (타겟 변수 생성용)
        financial_path = os.path.join(project_root, 'data', '재무정보_final_imputed.csv')
        financial_df = pd.read_csv(financial_path, encoding='cp949')
        
        features_path = os.path.join(project_root, 'data', 'growth_features_normalized.csv')
        features_df = pd.read_csv(features_path, encoding='utf-8-sig')
        print(f"정규화된 재무 피처 데이터 로드: {features_df.shape}")
    
    return features_df, financial_df


def load_test_dataset(project_root=None):
    """test_dataset_patent_feature_add.csv 파일 로드"""
    if project_root is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
    
    test_path = os.path.join(project_root, 'data', 'test_dataset_patent_feature_add.csv')
    
    if os.path.exists(test_path):
        test_df = pd.read_csv(test_path, encoding='utf-8-sig')
        print(f"test_dataset_patent_feature_add.csv 로드: {test_df.shape}")
        
        # target_growth를 target_top30으로 변환 (호환성을 위해)
        if 'target_growth' in test_df.columns:
            test_df['target_top30'] = test_df['target_growth']
        
        return test_df
    else:
        print(f"경고: {test_path} 파일을 찾을 수 없습니다.")
        return pd.DataFrame()


def load_actual_target_from_csv(project_root, target_year=2024):
    """
    연도별_성장여부_컬럼변환.csv 파일에서 실제 타겟 값을 로드
    
    Parameters:
    -----------
    project_root : str
        프로젝트 루트 디렉토리 경로
    target_year : int
        타겟 연도 (기본값: 2024)
    
    Returns:
    --------
    pd.DataFrame
        기업명과 실제 타겟 값(y)을 포함한 데이터프레임
    """
    actual_path = os.path.join(project_root, 'data', '연도별_성장여부_컬럼변환.csv')
    
    try:
        actual_df = pd.read_csv(actual_path, encoding='utf-8-sig')
        # 해당 연도의 데이터만 추출
        actual_target = actual_df[actual_df['연도'] == target_year].copy()
        # 기업명 정규화
        actual_target['기업명_정규화'] = actual_target['기업명'].str.strip()
        print(f"실제 타겟 데이터 로드 완료: {len(actual_target)}개 기업 ({target_year}년)")
        return actual_target[['기업명', '기업명_정규화', 'y']]
    except FileNotFoundError:
        print(f"경고: {actual_path} 파일을 찾을 수 없습니다.")
        return pd.DataFrame()
    except Exception as e:
        print(f"경고: 실제 타겟 데이터 로드 중 오류 발생: {e}")
        return pd.DataFrame()


def create_target_variable(features_df, financial_df):
    """
    타겟 변수 생성: 다음 연도의 매출액 성장률이 상위 30%에 속하는지 여부 (이진 분류)
    - 2019년 피처 → 2020년 타겟 (2020년 성장률 상위 30% 여부)
    - 2020년 피처 → 2021년 타겟 (2021년 성장률 상위 30% 여부)
    - ...
    - 2023년 피처 → 2024년 타겟 (2024년 성장률 상위 30% 여부) - 최종 테스트
    """
    # 매출액 컬럼 숫자 변환 (컬럼명 자동 감지)
    revenue_col_name = None
    for col in financial_df.columns:
        if '매출' in str(col):
            revenue_col_name = col
            break
    
    if revenue_col_name is None:
        raise ValueError("매출액 컬럼을 찾을 수 없습니다.")
    
    # 이미 숫자형이면 그대로 사용, 아니면 변환
    if financial_df[revenue_col_name].dtype == 'object':
        financial_df['매출액'] = financial_df[revenue_col_name].astype(str).str.replace(',', '').str.replace(' - ', '').str.replace('-', '')
        financial_df['매출액'] = pd.to_numeric(financial_df['매출액'], errors='coerce')
    else:
        financial_df['매출액'] = pd.to_numeric(financial_df[revenue_col_name], errors='coerce')
    
    # 기업별, 연도별 매출액 딕셔너리 생성 (빠른 조회를 위해)
    revenue_dict = {}
    for _, row in financial_df.iterrows():
        key = (row['기업명'], row['연도'])
        revenue_dict[key] = row['매출액']
    
    # 먼저 모든 성장률 계산
    growth_rates = []
    target_years = []
    
    for idx, row in features_df.iterrows():
        company_name = row['기업명']
        feature_year = row['연도']  # 피처 기준 연도 (예: 2019)
        target_year = feature_year + 1  # 예측 대상 연도 (예: 2020)
        
        # 매출액 조회
        rev_t = revenue_dict.get((company_name, feature_year), np.nan)
        rev_t1 = revenue_dict.get((company_name, target_year), np.nan)
        
        if pd.notna(rev_t) and pd.notna(rev_t1) and rev_t > 0:
            # 매출액 성장률 계산
            growth_rate = (rev_t1 / rev_t) - 1
            growth_rates.append(growth_rate)
            target_years.append(target_year)
        else:
            growth_rates.append(np.nan)
            target_years.append(target_year)
    
    features_df['growth_rate'] = growth_rates
    features_df['target_year'] = target_years
    
    # 각 타겟 연도별로 상위 30% 기준 계산
    target_list = []
    for target_year in sorted(features_df['target_year'].unique()):
        year_data = features_df[features_df['target_year'] == target_year].copy()
        year_growth_rates = year_data['growth_rate'].dropna()
        
        if len(year_growth_rates) > 0:
            # 상위 30% 기준 (70th percentile)
            threshold = year_growth_rates.quantile(0.7)
            
            # 해당 연도의 데이터에 대해 이진 타겟 생성
            for idx in year_data.index:
                growth_rate = year_data.loc[idx, 'growth_rate']
                if pd.notna(growth_rate):
                    # 상위 30%에 속하면 1, 아니면 0
                    target_list.append(1 if growth_rate >= threshold else 0)
                else:
                    target_list.append(np.nan)
        else:
            # 해당 연도에 데이터가 없으면 모두 NaN
            for idx in year_data.index:
                target_list.append(np.nan)
    
    # 인덱스 순서 맞추기
    features_df = features_df.sort_index()
    features_df['target_top30'] = target_list
    
    # 각 연도별 상위 30% 기준 출력
    print("\n연도별 상위 30% 기준 (70th percentile):")
    for target_year in sorted(features_df['target_year'].unique()):
        year_data = features_df[features_df['target_year'] == target_year]
        year_growth = year_data['growth_rate'].dropna()
        if len(year_growth) > 0:
            threshold = year_growth.quantile(0.7)
            top30_count = (year_data['target_top30'] == 1).sum()
            print(f"  {target_year}년: {threshold:.4f} (상위 30% 기업 수: {top30_count}개)")
    
    return features_df


def prepare_data(df):
    """모델링을 위한 데이터 준비"""
    # 기업명, 연도, target_year, target_growth, target_top30은 제외
    exclude_cols = ['기업명', '연도', 'target_year', 'target_growth', 'target_top30', 'growth_rate']
    
    # 임베딩 컬럼 제외 (patent_emb_0 ~ patent_emb_49)
    embedding_cols = [col for col in df.columns if 'patent_emb' in col.lower()]
    exclude_cols.extend(embedding_cols)
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    if len(embedding_cols) > 0:
        print(f"임베딩 컬럼 {len(embedding_cols)}개 제외: {embedding_cols[:5]}..." if len(embedding_cols) > 5 else f"임베딩 컬럼 {len(embedding_cols)}개 제외: {embedding_cols}")
    
    # target_top30 또는 target_growth 사용
    if 'target_top30' in df.columns:
        y = df['target_top30'].copy()
    elif 'target_growth' in df.columns:
        y = df['target_growth'].copy()
    else:
        raise ValueError("타겟 변수(target_top30 또는 target_growth)를 찾을 수 없습니다.")
    
    X = df[feature_cols].copy()
    
    # 결측치가 있는 행 제거
    mask = ~(X.isnull().any(axis=1) | y.isnull())
    X = X[mask]
    y = y[mask]
    
    print(f"최종 데이터 shape: {X.shape}")
    print(f"타겟 변수 통계 (이진 분류):")
    print(f"  클래스 분포:")
    print(f"    0 (하위 70%): {(y == 0).sum()}개 ({(y == 0).sum()/len(y)*100:.1f}%)")
    print(f"    1 (상위 30%): {(y == 1).sum()}개 ({(y == 1).sum()/len(y)*100:.1f}%)")
    
    return X, y, feature_cols


def load_xgboost_params_from_file(script_dir=None):
    """
    XGBoost의 best_params.txt 파일에서 파라미터를 읽어옴
    
    Parameters:
    -----------
    script_dir : str, optional
        XGBoost 모델 디렉토리 경로. None이면 자동 감지
    
    Returns:
    --------
    dict
        XGBoost 파라미터 딕셔너리 (필요한 파라미터만 포함, objective, eval_metric 등은 제외)
    """
    if script_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # XGBoost의 best_params.txt 경로
    xgb_params_path = os.path.join(script_dir, 'best_params.txt')
    
    if not os.path.exists(xgb_params_path):
        print(f"경고: {xgb_params_path} 파일을 찾을 수 없습니다.")
        return None
    
    # XGBoost 파라미터 읽기
    xgb_params = {}
    with open(xgb_params_path, 'r', encoding='utf-8') as f:
        for line in f:
            if ':' in line and not line.startswith('최적'):
                key, value = line.strip().split(':', 1)
                key = key.strip()
                value = value.strip()
                
                # XGBoost 모델 학습에 필요한 파라미터만 선택
                # objective, eval_metric, booster, random_state, verbosity는 나중에 설정
                if key in ['objective', 'eval_metric', 'booster', 'random_state', 'verbosity']:
                    continue
                
                # 숫자로 변환 시도
                try:
                    if '.' in value:
                        xgb_params[key] = float(value)
                    else:
                        xgb_params[key] = int(value)
                except ValueError:
                    xgb_params[key] = value
    
    print(f"\n[XGBoost 파라미터 로드 완료]")
    print(f"  파일 경로: {xgb_params_path}")
    print(f"  로드된 파라미터:")
    for key, value in xgb_params.items():
        print(f"    {key}: {value}")
    
    return xgb_params


def optimize_xgboost_hyperparameters(X_train, y_train, cv=5):
    """단계별 그리드 탐색을 사용한 XGBoost 하이퍼파라미터 최적화"""
    print("\n[하이퍼파라미터 튜닝 시작]")
    print(f"  방법: 단계별 그리드 탐색 (cv={cv})")
    
    # MLflow 실험 설정 (메인 run 컨텍스트가 있으면 그대로 사용, 없으면 새로 생성)
    # 메인 run 컨텍스트 내에서 호출되므로 별도 실험 설정하지 않음
    
    # 기본 파라미터 (단계별로 업데이트됨)
    best_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'booster': 'gbtree',
        'random_state': 42,
        'verbosity': 0,
        'n_estimators': 1000
    }
    
    # 단계 1: 트리 구조 파라미터 탐색
    print("\n[단계 1/4] 트리 구조 파라미터 탐색 (max_depth, min_child_weight)")
    step1_grid = {
        'max_depth': [5, 7, 10],
        'min_child_weight': [1, 3, 5]
    }
    
    best_score = -np.inf
    step1_params = {}
    param_list = list(ParameterGrid(step1_grid))
    
    # 메인 run 컨텍스트가 있으면 중첩 run으로, 없으면 새 run으로 시작
    try:
        # 현재 활성 run이 있는지 확인
        active_run = mlflow.active_run()
        if active_run:
            # 중첩 run으로 시작
            step1_context = mlflow.start_run(nested=True, run_name="Step1_Tree_Structure")
        else:
            # 새 run으로 시작
            mlflow.set_experiment("XGBoost_Hyperparameter_Tuning")
            step1_context = mlflow.start_run(run_name="Step1_Tree_Structure")
    except:
        # 실패하면 새 실험으로 시작
        mlflow.set_experiment("XGBoost_Hyperparameter_Tuning")
        step1_context = mlflow.start_run(run_name="Step1_Tree_Structure")
    
    with step1_context:
        for idx, params in enumerate(tqdm(param_list, desc="트리 구조 탐색", unit="조합")):
            model = XGBClassifier(**best_params, **params)
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
            mean_score = scores.mean()
            
            # 각 조합마다 별도의 child run 생성
            with mlflow.start_run(nested=True, run_name=f"Step1_Combination_{idx+1}"):
                mlflow.log_params(params)
                mlflow.log_metric("cv_roc_auc", mean_score)
            
            if mean_score > best_score:
                best_score = mean_score
                step1_params = params
        
        mlflow.log_metric("best_cv_roc_auc", best_score)
        mlflow.log_params(step1_params)
        print(f"  최적 트리 구조: {step1_params}, CV ROC-AUC: {best_score:.4f}")
    
    best_params.update(step1_params)
    
    # 단계 2: 학습률 탐색
    print("\n[단계 2/4] 학습률 탐색")
    step2_grid = {'learning_rate': [0.01, 0.05, 0.1]}
    
    best_score = -np.inf
    step2_params = {}
    param_list = list(ParameterGrid(step2_grid))
    
    with mlflow.start_run(nested=True, run_name="Step2_Learning_Rate"):
        for idx, params in enumerate(tqdm(param_list, desc="학습률 탐색", unit="조합")):
            model = XGBClassifier(**best_params, **params)
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
            mean_score = scores.mean()
            
            # 각 조합마다 별도의 child run 생성
            with mlflow.start_run(nested=True, run_name=f"Step2_Combination_{idx+1}"):
                mlflow.log_params(params)
                mlflow.log_metric("cv_roc_auc", mean_score)
            
            if mean_score > best_score:
                best_score = mean_score
                step2_params = params
        
        mlflow.log_metric("best_cv_roc_auc", best_score)
        mlflow.log_params(step2_params)
        print(f"  최적 학습률: {step2_params}, CV ROC-AUC: {best_score:.4f}")
    
    best_params.update(step2_params)
    
    # 단계 3: 정규화 파라미터 탐색
    print("\n[단계 3/4] 정규화 파라미터 탐색 (reg_alpha, reg_lambda, gamma)")
    step3_grid = {
        'reg_alpha': [0.1, 0.5, 1.0],
        'reg_lambda': [0.1, 0.5, 1.0],
        'gamma': [0, 0.1, 0.5]
    }
    
    best_score = -np.inf
    step3_params = {}
    param_list = list(ParameterGrid(step3_grid))
    
    with mlflow.start_run(nested=True, run_name="Step3_Regularization"):
        for idx, params in enumerate(tqdm(param_list, desc="정규화 탐색", unit="조합")):
            model = XGBClassifier(**best_params, **params)
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
            mean_score = scores.mean()
            
            # 각 조합마다 별도의 child run 생성
            with mlflow.start_run(nested=True, run_name=f"Step3_Combination_{idx+1}"):
                mlflow.log_params(params)
                mlflow.log_metric("cv_roc_auc", mean_score)
            
            if mean_score > best_score:
                best_score = mean_score
                step3_params = params
        
        mlflow.log_metric("best_cv_roc_auc", best_score)
        mlflow.log_params(step3_params)
        print(f"  최적 정규화: {step3_params}, CV ROC-AUC: {best_score:.4f}")
    
    best_params.update(step3_params)
    
    # 단계 4: 샘플링 파라미터 탐색
    print("\n[단계 4/4] 샘플링 파라미터 탐색 (subsample, colsample_bytree)")
    step4_grid = {
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9]
    }
    
    best_score = -np.inf
    step4_params = {}
    param_list = list(ParameterGrid(step4_grid))
    
    with mlflow.start_run(nested=True, run_name="Step4_Sampling"):
        for idx, params in enumerate(tqdm(param_list, desc="샘플링 탐색", unit="조합")):
            model = XGBClassifier(**best_params, **params)
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
            mean_score = scores.mean()
            
            # 각 조합마다 별도의 child run 생성
            with mlflow.start_run(nested=True, run_name=f"Step4_Combination_{idx+1}"):
                mlflow.log_params(params)
                mlflow.log_metric("cv_roc_auc", mean_score)
            
            if mean_score > best_score:
                best_score = mean_score
                step4_params = params
        
        mlflow.log_metric("best_cv_roc_auc", best_score)
        mlflow.log_params(step4_params)
        print(f"  최적 샘플링: {step4_params}, CV ROC-AUC: {best_score:.4f}")
    
    best_params.update(step4_params)
    
    # 최종 결과 출력 및 MLflow 로깅
    print(f"\n[최종 결과] 최적 하이퍼파라미터:")
    final_params = {k: v for k, v in best_params.items() if k not in ['objective', 'eval_metric', 'booster', 'random_state', 'verbosity', 'n_estimators']}
    for key, value in final_params.items():
        print(f"  {key}: {value}")
    print(f"\n최종 CV 점수 (ROC-AUC): {best_score:.4f}")
    
    # MLflow에 최종 파라미터 로깅
    mlflow.log_params(final_params)
    mlflow.log_metric("final_cv_roc_auc", best_score)
    
    return final_params


def train_xgboost(X_train, y_train, X_val, y_val, use_tuning=True, cv=5, use_saved_params=False):
    """
    XGBoost 모델 학습 (이진 분류)
    
    Parameters:
    -----------
    use_tuning : bool
        하이퍼파라미터 튜닝 수행 여부
    cv : int
        Cross-validation fold 수
    use_saved_params : bool
        True이면 XGBoost의 best_params.txt 파일에서 파라미터를 읽어서 사용
    """
    # 하이퍼파라미터 튜닝
    if use_tuning:
        # 학습+검증 데이터를 합쳐서 GridSearchCV에 사용
        if isinstance(X_train, pd.DataFrame):
            X_train_val = pd.concat([X_train, X_val], ignore_index=True)
        else:
            X_train_val = pd.concat([pd.DataFrame(X_train), pd.DataFrame(X_val)], ignore_index=True)
        if isinstance(y_train, pd.Series):
            y_train_val = pd.concat([y_train, y_val], ignore_index=True)
        else:
            y_train_val = pd.concat([pd.Series(y_train), pd.Series(y_val)], ignore_index=True)
        best_params = optimize_xgboost_hyperparameters(X_train_val, y_train_val, cv=cv)
        
        # 최적 파라미터로 모델 생성
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'booster': 'gbtree',
            'random_state': 42,
            'verbosity': 0,
            **best_params
        }
    elif use_saved_params:
        # XGBoost의 best_params.txt에서 파라미터 로드
        script_dir = os.path.dirname(os.path.abspath(__file__))
        saved_params = load_xgboost_params_from_file(script_dir)
        
        if saved_params is None:
            print("경고: 저장된 파라미터 로드 실패. 기본 파라미터를 사용합니다.")
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'booster': 'gbtree',
                'max_depth': 6,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.9,
                'min_child_weight': 1,
                'gamma': 0,
                'random_state': 42,
                'verbosity': 0
            }
        else:
            # 저장된 파라미터를 사용하여 모델 생성
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'booster': 'gbtree',
                'random_state': 42,
                'verbosity': 0,
                **saved_params
            }
    else:
        # 기본 하이퍼파라미터 설정
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'booster': 'gbtree',
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.9,
            'min_child_weight': 1,
            'gamma': 0,
            'random_state': 42,
            'verbosity': 0
        }
    
    # DMatrix 생성
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    # 모델 학습
    print("\n최종 모델 학습 중...")
    evals = [(dtrain, 'train'), (dval, 'eval')]
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=evals,
        early_stopping_rounds=50,
        verbose_eval=100
    )
    
    return model, params


def calculate_top_k_precision(y_true, y_pred_proba, k=20):
    """Top-K Precision 계산"""
    if len(y_true) < k:
        k = len(y_true)
    
    # 확률이 높은 상위 K개 선택
    top_k_indices = np.argsort(y_pred_proba)[::-1][:k]
    top_k_precision = y_true.iloc[top_k_indices].sum() / k if isinstance(y_true, pd.Series) else y_true[top_k_indices].sum() / k
    return top_k_precision


def evaluate_model(model, X, y, split_name='', threshold=0.3, company_names=None):
    """
    모델 평가 (이진 분류) - 새로운 지표 포함
    - PR-AUC (Precision-Recall AUC)
    - Top-K Precision (K=20, 50)
    - F1 Score
    - Brier Score
    """
    dtest = xgb.DMatrix(X)
    # 확률 예측
    y_pred_proba = model.predict(dtest)
    # 이진 예측 (임계값 기준, 기본값 0.4로 Recall 향상)
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # 기본 지표 계산
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y, y_pred_proba) if len(np.unique(y)) > 1 else 0.0
    
    # 새로운 지표 계산
    # PR-AUC (Precision-Recall AUC)
    pr_auc = average_precision_score(y, y_pred_proba) if len(np.unique(y)) > 1 else 0.0
    
    # Brier Score
    brier = brier_score_loss(y, y_pred_proba)
    
    # Top-K Precision
    y_series = y if isinstance(y, pd.Series) else pd.Series(y, index=X.index if hasattr(X, 'index') else range(len(y)))
    top20_precision = calculate_top_k_precision(y_series, y_pred_proba, k=20)
    top50_precision = calculate_top_k_precision(y_series, y_pred_proba, k=50)
    
    print(f"\n{split_name} 평가 결과:")
    print(f"  PR-AUC: {pr_auc:.4f}")
    print(f"  Brier Score: {brier:.4f}")
    print(f"  Top-20 Precision: {top20_precision:.4f}")
    print(f"  Top-50 Precision: {top50_precision:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  ROC-AUC: {roc_auc:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y, y_pred)
    print(f"\n  Confusion Matrix:")
    print(f"    True Negative: {cm[0,0]}, False Positive: {cm[0,1]}")
    print(f"    False Negative: {cm[1,0]}, True Positive: {cm[1,1]}")
    
    return y_pred, y_pred_proba, {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'brier': brier,
        'top20_precision': top20_precision,
        'top50_precision': top50_precision
    }


def plot_feature_importance(model, feature_cols, top_n=15):
    """피처 중요도 시각화"""
    # XGBoost 피처 중요도 추출
    # 여러 방법 시도
    importance_dict = {}
    
    # 방법 1: get_score() 사용
    try:
        importance_dict = model.get_score(importance_type='gain')
        if len(importance_dict) == 0:
            importance_dict = model.get_score(importance_type='weight')
    except:
        pass
    
    # 방법 2: get_booster() 사용
    if len(importance_dict) == 0:
        try:
            booster = model.get_booster()
            importance_dict = booster.get_score(importance_type='gain')
            if len(importance_dict) == 0:
                importance_dict = booster.get_score(importance_type='weight')
        except:
            pass
    
    # 피처명과 중요도 매핑
    # XGBoost는 실제 피처명을 키로 사용하거나 f0, f1 형식을 사용할 수 있음
    feature_importance_list = []
    for feature in feature_cols:
        # 먼저 실제 피처명으로 시도
        importance = importance_dict.get(feature, 0)
        
        # f0, f1 형식으로도 시도
        if importance == 0:
            feature_idx = feature_cols.index(feature)
            feature_key = f'f{feature_idx}'
            importance = importance_dict.get(feature_key, 0)
        
        # float로 변환
        try:
            importance = float(importance)
        except (ValueError, TypeError):
            importance = 0.0
        
        feature_importance_list.append({
            'feature': feature,
            'importance': importance
        })
    
    feature_importance_df = pd.DataFrame(feature_importance_list).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(data=feature_importance_df.head(top_n), x='importance', y='feature')
    plt.title(f'Top {top_n} Feature Importance (XGBoost)')
    plt.xlabel('Importance (Gain)')
    plt.tight_layout()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, 'feature_importance.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n피처 중요도 그래프 저장: {output_path}")
    plt.close()
    
    return feature_importance_df


def rolling_validation(X, y, features_df_filtered, model_params, years=[2021, 2022, 2023, 2024]):
    """
    롤링 검증: 연도 단위로 rolling/expanding validation 수행
    2021→2022→2023→2024 순서로 각 연도를 평가
    """
    results = []
    
    for i, val_year in enumerate(years):
        print(f"\n[롤링 검증] {val_year}년 평가")
        print("-" * 70)
        
        # Train: val_year 이전의 모든 데이터
        train_years = [yr for yr in features_df_filtered['연도'].unique() if yr < val_year]
        # Validation: val_year 데이터
        val_mask = features_df_filtered['연도'] == val_year
        train_mask = features_df_filtered['연도'].isin(train_years)
        
        if not train_mask.any() or not val_mask.any():
            print(f"  건너뜀: Train={train_mask.sum()}개, Val={val_mask.sum()}개")
            continue
        
        X_train_roll = X[train_mask]
        y_train_roll = y[train_mask]
        X_val_roll = X[val_mask]
        y_val_roll = y[val_mask]
        
        print(f"  Train: {len(X_train_roll)}개 (연도: {sorted(train_years)})")
        print(f"  Validation: {len(X_val_roll)}개 (연도: {val_year})")
        
        # 모델 학습
        dtrain = xgb.DMatrix(X_train_roll, label=y_train_roll)
        dval = xgb.DMatrix(X_val_roll, label=y_val_roll)
        
        model = xgb.train(
            model_params,
            dtrain,
            num_boost_round=1000,
            evals=[(dtrain, 'train'), (dval, 'eval')],
            early_stopping_rounds=50,
            verbose_eval=0
        )
        
        # 평가
        y_pred_proba = model.predict(dval)
        y_series = y_val_roll if isinstance(y_val_roll, pd.Series) else pd.Series(y_val_roll, index=X_val_roll.index if hasattr(X_val_roll, 'index') else range(len(y_val_roll)))
        
        pr_auc = average_precision_score(y_val_roll, y_pred_proba) if len(np.unique(y_val_roll)) > 1 else 0.0
        brier = brier_score_loss(y_val_roll, y_pred_proba)
        roc_auc = roc_auc_score(y_val_roll, y_pred_proba) if len(np.unique(y_val_roll)) > 1 else 0.0
        f1 = f1_score(y_val_roll, (y_pred_proba >= 0.4).astype(int), zero_division=0)
        top20_precision = calculate_top_k_precision(y_series, y_pred_proba, k=20)
        top50_precision = calculate_top_k_precision(y_series, y_pred_proba, k=50)
        
        results.append({
            'year': val_year,
            'pr_auc': pr_auc,
            'brier': brier,
            'roc_auc': roc_auc,
            'f1': f1,
            'top20_precision': top20_precision,
            'top50_precision': top50_precision
        })
        
        print(f"  PR-AUC: {pr_auc:.4f}, Brier: {brier:.4f}, ROC-AUC: {roc_auc:.4f}, F1: {f1:.4f}, Top-20: {top20_precision:.4f}, Top-50: {top50_precision:.4f}")
    
    return results


def groupkfold_validation(X, y, features_df_filtered, model_params, n_splits=5):
    """
    GroupKFold 검증: 기업 단위로 분할하여 미관측 기업 일반화 강건성 점검
    """
    print(f"\n[GroupKFold 검증] 기업 단위 {n_splits}-fold")
    print("-" * 70)
    
    # 기업명 추출 (X의 인덱스로 features_df_filtered와 매칭)
    company_names = features_df_filtered.loc[X.index, '기업명'].values
    
    # GroupKFold 수행
    gkf = GroupKFold(n_splits=n_splits)
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=company_names)):
        print(f"\n  Fold {fold+1}/{n_splits}")
        
        X_train_fold = X.iloc[train_idx] if isinstance(X, pd.DataFrame) else X[train_idx]
        y_train_fold = y.iloc[train_idx] if isinstance(y, pd.Series) else y[train_idx]
        X_val_fold = X.iloc[val_idx] if isinstance(X, pd.DataFrame) else X[val_idx]
        y_val_fold = y.iloc[val_idx] if isinstance(y, pd.Series) else y[val_idx]
        
        train_companies = len(np.unique(company_names[train_idx]))
        val_companies = len(np.unique(company_names[val_idx]))
        print(f"    Train 기업: {train_companies}개, Validation 기업: {val_companies}개")
        
        # 모델 학습
        dtrain = xgb.DMatrix(X_train_fold, label=y_train_fold)
        dval = xgb.DMatrix(X_val_fold, label=y_val_fold)
        
        model = xgb.train(
            model_params,
            dtrain,
            num_boost_round=1000,
            evals=[(dtrain, 'train'), (dval, 'eval')],
            early_stopping_rounds=50,
            verbose_eval=0
        )
        
        # 평가
        y_pred_proba = model.predict(dval)
        y_series_fold = y_val_fold if isinstance(y_val_fold, pd.Series) else pd.Series(y_val_fold, index=X_val_fold.index if hasattr(X_val_fold, 'index') else range(len(y_val_fold)))
        
        pr_auc = average_precision_score(y_val_fold, y_pred_proba) if len(np.unique(y_val_fold)) > 1 else 0.0
        brier = brier_score_loss(y_val_fold, y_pred_proba)
        roc_auc = roc_auc_score(y_val_fold, y_pred_proba) if len(np.unique(y_val_fold)) > 1 else 0.0
        f1 = f1_score(y_val_fold, (y_pred_proba >= 0.4).astype(int), zero_division=0)
        top20_precision = calculate_top_k_precision(y_series_fold, y_pred_proba, k=20)
        top50_precision = calculate_top_k_precision(y_series_fold, y_pred_proba, k=50)
        
        fold_results.append({
            'fold': fold + 1,
            'pr_auc': pr_auc,
            'brier': brier,
            'roc_auc': roc_auc,
            'f1': f1,
            'top20_precision': top20_precision,
            'top50_precision': top50_precision
        })
        
        print(f"    PR-AUC: {pr_auc:.4f}, Brier: {brier:.4f}, ROC-AUC: {roc_auc:.4f}, F1: {f1:.4f}, Top-20: {top20_precision:.4f}, Top-50: {top50_precision:.4f}")
    
    # 평균 계산
    avg_results = {
        'pr_auc': np.mean([r['pr_auc'] for r in fold_results]),
        'brier': np.mean([r['brier'] for r in fold_results]),
        'roc_auc': np.mean([r['roc_auc'] for r in fold_results]),
        'f1': np.mean([r['f1'] for r in fold_results]),
        'top20_precision': np.mean([r['top20_precision'] for r in fold_results]),
        'top50_precision': np.mean([r['top50_precision'] for r in fold_results])
    }
    
    return avg_results


def print_evaluation_table(holdout_results, rolling_results, groupkfold_results):
    """
    평가 결과를 표 형태로 출력 (이미지 스타일)
    모든 지표 포함: PR-AUC, Top-20 Precision, Top-50 Precision, F1 Score, Brier Score, ROC-AUC
    """
    print("\n" + "="*80)
    print("모델 평가 결과 종합 표")
    print("="*80)
    
    # Holdout (최종 테스트) 결과
    holdout_pr_auc = holdout_results.get('pr_auc', 0.0)
    holdout_top20 = holdout_results.get('top20_precision', 0.0)
    holdout_top50 = holdout_results.get('top50_precision', 0.0)
    holdout_f1 = holdout_results.get('f1', 0.0)
    holdout_brier = holdout_results.get('brier', 0.0)
    holdout_roc_auc = holdout_results.get('roc_auc', 0.0)
    
    # Rolling 검증 평균
    if rolling_results:
        rolling_pr_auc = np.mean([r['pr_auc'] for r in rolling_results])
        rolling_top20 = np.mean([r.get('top20_precision', 0.0) for r in rolling_results])
        rolling_top50 = np.mean([r.get('top50_precision', 0.0) for r in rolling_results])
        rolling_f1 = np.mean([r.get('f1', 0.0) for r in rolling_results])
        rolling_brier = np.mean([r['brier'] for r in rolling_results])
        rolling_roc_auc = np.mean([r['roc_auc'] for r in rolling_results])
    else:
        rolling_pr_auc = rolling_top20 = rolling_top50 = rolling_f1 = rolling_brier = rolling_roc_auc = 0.0
    
    # GroupKFold 결과
    if groupkfold_results:
        gkf_pr_auc = groupkfold_results.get('pr_auc', 0.0)
        gkf_top20 = groupkfold_results.get('top20_precision', 0.0)
        gkf_top50 = groupkfold_results.get('top50_precision', 0.0)
        gkf_f1 = groupkfold_results.get('f1', 0.0)
        gkf_brier = groupkfold_results.get('brier', 0.0)
        gkf_roc_auc = groupkfold_results.get('roc_auc', 0.0)
    else:
        gkf_pr_auc = gkf_top20 = gkf_top50 = gkf_f1 = gkf_brier = gkf_roc_auc = 0.0
    
    # 표 출력
    print("\n모델: XGB")
    print("\n" + "-"*80)
    print(f"{'지표':<25} {'groupkfold':<15} {'holdout':<15} {'rolling':<15}")
    print("-"*80)
    print(f"{'pr_auc':<25} {gkf_pr_auc:<15.3f} {holdout_pr_auc:<15.3f} {rolling_pr_auc:<15.3f}")
    print(f"{'top-20-precision':<25} {gkf_top20:<15.3f} {holdout_top20:<15.3f} {rolling_top20:<15.3f}")
    print(f"{'top-50-precision':<25} {gkf_top50:<15.3f} {holdout_top50:<15.3f} {rolling_top50:<15.3f}")
    print(f"{'f1_score':<25} {gkf_f1:<15.3f} {holdout_f1:<15.3f} {rolling_f1:<15.3f}")
    print(f"{'brier_score':<25} {gkf_brier:<15.3f} {holdout_brier:<15.3f} {rolling_brier:<15.3f}")
    print(f"{'roc_auc':<25} {gkf_roc_auc:<15.3f} {holdout_roc_auc:<15.3f} {rolling_roc_auc:<15.3f}")
    print("-"*80)
    
    # 결과 딕셔너리 반환 (CSV 저장용)
    results_dict = {
        'model': 'XGB',
        'groupkfold_pr_auc': gkf_pr_auc,
        'groupkfold_top20_precision': gkf_top20,
        'groupkfold_top50_precision': gkf_top50,
        'groupkfold_f1_score': gkf_f1,
        'groupkfold_brier_score': gkf_brier,
        'groupkfold_roc_auc': gkf_roc_auc,
        'holdout_pr_auc': holdout_pr_auc,
        'holdout_top20_precision': holdout_top20,
        'holdout_top50_precision': holdout_top50,
        'holdout_f1_score': holdout_f1,
        'holdout_brier_score': holdout_brier,
        'holdout_roc_auc': holdout_roc_auc,
        'rolling_pr_auc': rolling_pr_auc,
        'rolling_top20_precision': rolling_top20,
        'rolling_top50_precision': rolling_top50,
        'rolling_f1_score': rolling_f1,
        'rolling_brier_score': rolling_brier,
        'rolling_roc_auc': rolling_roc_auc
    }
    
    return results_dict


def save_performance_report(train_metrics, val_metrics, test_metrics, best_params, 
                           feature_importance_df, script_dir, model_name='XGBoost'):
    """모델 성능 지표를 보고서 파일로 저장"""
    report_path = os.path.join(script_dir, f'{model_name}_PERFORMANCE_REPORT.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# {model_name} 모델 성능 보고서\n\n")
        f.write(f"**생성 일시**: {datetime.now().strftime('%Y년 %m월 %d일 %H:%M:%S')}\n\n")
        f.write("---\n\n")
        
        # 하이퍼파라미터
        f.write("## 하이퍼파라미터\n\n")
        f.write("| 파라미터 | 값 |\n")
        f.write("|---------|-----|\n")
        for key, value in best_params.items():
            f.write(f"| {key} | {value} |\n")
        f.write(f"| threshold | 0.3 |\n")
        f.write(f"| cv_folds | 5 |\n")
        f.write(f"| n_estimators | 1000 |\n")
        f.write("\n---\n\n")
        
        # 성능 지표
        f.write("## 모델 성능 지표\n\n")
        
        # Train 성능
        f.write("### Train 데이터 성능\n\n")
        f.write("| 지표 | 값 |\n")
        f.write("|------|-----|\n")
        f.write(f"| Accuracy | {train_metrics['accuracy']:.4f} |\n")
        f.write(f"| Precision | {train_metrics['precision']:.4f} |\n")
        f.write(f"| Recall | {train_metrics['recall']:.4f} |\n")
        f.write(f"| F1-Score | {train_metrics['f1']:.4f} |\n")
        f.write(f"| ROC-AUC | {train_metrics['roc_auc']:.4f} |\n")
        f.write("\n")
        
        # Validation 성능
        f.write("### Validation 데이터 성능\n\n")
        f.write("| 지표 | 값 |\n")
        f.write("|------|-----|\n")
        f.write(f"| Accuracy | {val_metrics['accuracy']:.4f} |\n")
        f.write(f"| Precision | {val_metrics['precision']:.4f} |\n")
        f.write(f"| Recall | {val_metrics['recall']:.4f} |\n")
        f.write(f"| F1-Score | {val_metrics['f1']:.4f} |\n")
        f.write(f"| ROC-AUC | {val_metrics['roc_auc']:.4f} |\n")
        f.write("\n")
        
        # Test 성능
        if test_metrics is not None:
            f.write("### Test 데이터 성능\n\n")
            f.write("| 지표 | 값 |\n")
            f.write("|------|-----|\n")
            f.write(f"| Accuracy | {test_metrics['accuracy']:.4f} |\n")
            f.write(f"| Precision | {test_metrics['precision']:.4f} |\n")
            f.write(f"| Recall | {test_metrics['recall']:.4f} |\n")
            f.write(f"| F1-Score | {test_metrics['f1']:.4f} |\n")
            f.write(f"| ROC-AUC | {test_metrics['roc_auc']:.4f} |\n")
            f.write("\n")
        
        f.write("---\n\n")
        
        # 피처 중요도
        f.write("## Top 15 피처 중요도\n\n")
        f.write("| 순위 | 피처명 | 중요도 |\n")
        f.write("|------|--------|--------|\n")
        top_features = feature_importance_df.head(15)
        for idx, (_, row) in enumerate(top_features.iterrows(), 1):
            f.write(f"| {idx} | {row['feature']} | {row['importance']:.4f} |\n")
        f.write("\n")
        
        f.write("---\n\n")
        
        # 성능 지표 해석
        f.write("## 성능 지표 해석\n\n")
        f.write("### Accuracy (정확도)\n")
        f.write("- 전체 예측 중 정확하게 예측한 비율\n")
        f.write(f"- 현재 모델: {val_metrics['accuracy']:.2%}\n\n")
        
        f.write("### Precision (정밀도)\n")
        f.write("- 상위 30%로 예측한 기업 중 실제로 상위 30%인 비율\n")
        f.write(f"- 현재 모델: {val_metrics['precision']:.2%}\n\n")
        
        f.write("### Recall (재현율)\n")
        f.write("- 실제 상위 30% 기업 중 모델이 올바르게 예측한 비율\n")
        f.write(f"- 현재 모델: {val_metrics['recall']:.2%}\n\n")
        
        f.write("### F1-Score\n")
        f.write("- Precision과 Recall의 조화평균\n")
        f.write(f"- 현재 모델: {val_metrics['f1']:.4f}\n\n")
        
        f.write("### ROC-AUC\n")
        f.write("- ROC 곡선 아래 면적 (0~1, 높을수록 좋음)\n")
        f.write(f"- 현재 모델: {val_metrics['roc_auc']:.4f}\n\n")
        
    print(f"\n성능 보고서 저장: {report_path}")
    return report_path


def plot_predictions(y_true, y_pred, y_pred_proba, split_name=''):
    """예측 결과 시각화 (이진 분류)"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
    axes[0, 0].set_xlabel('Predicted')
    axes[0, 0].set_ylabel('Actual')
    axes[0, 0].set_title(f'{split_name} - Confusion Matrix (XGBoost)')
    
    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    axes[0, 1].plot(fpr, tpr, label=f'ROC (AUC = {roc_auc_score(y_true, y_pred_proba):.3f})')
    axes[0, 1].plot([0, 1], [0, 1], 'r--', label='Random')
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].set_title(f'{split_name} - ROC Curve (XGBoost)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 예측 확률 분포
    axes[1, 0].hist(y_pred_proba[y_true == 0], bins=30, alpha=0.5, label='Class 0 (하위 70%)', color='blue')
    axes[1, 0].hist(y_pred_proba[y_true == 1], bins=30, alpha=0.5, label='Class 1 (상위 30%)', color='red')
    axes[1, 0].axvline(x=0.3, color='green', linestyle='--', label='Threshold (0.3)')
    axes[1, 0].set_xlabel('Predicted Probability')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title(f'{split_name} - Probability Distribution (XGBoost)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    axes[1, 1].plot(recall, precision)
    axes[1, 1].set_xlabel('Recall')
    axes[1, 1].set_ylabel('Precision')
    axes[1, 1].set_title(f'{split_name} - Precision-Recall Curve (XGBoost)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, f'predictions_{split_name.lower()}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"예측 결과 그래프 저장: {output_path}")
    plt.close()


def main():
    """메인 실행 함수"""
    print("="*70)
    print("XGBoost 모델링 시작")
    print("="*70)
    
    # 1. 데이터 로드 (train_dataset_patent_feature_add.csv 사용)
    print("\n[1단계] 데이터 로드")
    use_train_dataset = True  # train_dataset_patent_feature_add.csv 사용
    features_df, financial_df = load_data(use_train_dataset=use_train_dataset)
    print(f"피처 데이터 shape: {features_df.shape}")
    
    # 2. 타겟 변수 확인/생성
    if use_train_dataset:
        # train_dataset_patent_feature_add.csv를 사용하는 경우, target_growth가 이미 있음
        print("\n[2단계] 타겟 변수 확인 (train_dataset_patent_feature_add.csv 사용)")
        if 'target_growth' in features_df.columns:
            features_df['target_top30'] = features_df['target_growth']
            print(f"target_growth를 target_top30으로 변환 완료")
        if 'target_top30' in features_df.columns:
            print(f"타겟 변수 확인 완료. 결측치: {features_df['target_top30'].isnull().sum()}개")
        else:
            raise ValueError("타겟 변수를 찾을 수 없습니다.")
    else:
        # 기존 방식: 타겟 변수 생성
        print("\n[2단계] 타겟 변수 생성 (상위 30% 이진 분류)")
        if financial_df is None:
            raise ValueError("기존 방식을 사용하려면 financial_df가 필요합니다.")
        features_df = create_target_variable(features_df, financial_df)
        print(f"타겟 변수 생성 완료. 결측치: {features_df['target_top30'].isnull().sum()}개")
    
    # 3. 데이터 준비
    print("\n[3단계] 데이터 준비")
    X, y, feature_cols = prepare_data(features_df)
    
    # 4. Train/Test 분리 (연도 기준)
    # 최종 테스트(시간 홀드아웃): 2019~2023년 말 피처는 Train, 2024년 타겟을 예측
    print("\n[4단계] Train/Test 분리 (최종 테스트 - 시간 홀드아웃)")
    print("  - Train: 2019년 말, 2020년 말, 2021년 말, 2022년 말, 2023년 말 피처")
    print("  - Test: 2024년 타겟 예측")
    # 인덱스 매핑을 위해 features_df와 X의 인덱스 동기화
    features_df_filtered = features_df.loc[X.index].copy()
    # Train: 2019~2023년 피처 (전체 Train 데이터)
    train_mask = features_df_filtered['연도'] <= 2023  # 2019-2023년 말 피처
    
    # Train: 2019~2023년 피처 (전체 Train 데이터)
    X_train_all = X[train_mask]
    y_train_all = y[train_mask]
    
    print(f"Train 데이터 (2019~2023년): {len(X_train_all)}개")
    
    # 최종 테스트용: use_train_dataset가 False일 때만 사용 (기존 방식)
    if not use_train_dataset:
        test_mask_2023 = features_df_filtered['연도'] == 2023
        X_test_2024 = X[test_mask_2023] if test_mask_2023.sum() > 0 else pd.DataFrame()
    else:
        X_test_2024 = pd.DataFrame()  # test_dataset_patent_feature_add.csv를 사용하므로 빈 DataFrame
    
    # Train 데이터를 Train/Validation으로 분리 (모델 학습용)
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train_all, y_train_all, test_size=0.2, random_state=42
    )
    
    print(f"Train (학습용): {len(X_train_split)}개")
    print(f"Validation: {len(X_val)}개")
    print(f"최종 테스트 (2023년 피처 → 2024년 예측): {len(X_test_2024)}개")
    
    # 5. 모델 학습 (MLflow 실험 시작)
    print("\n[5단계] XGBoost 모델 학습")
    
    # 저장된 파라미터 사용 여부 설정 (튜닝 없이 바로 모델링)
    use_saved_params = True  # True로 설정하면 XGBoost의 best_params.txt 사용
    use_tuning = not use_saved_params  # use_saved_params가 True이면 튜닝 건너뜀
    
    experiment_name = f"XGBoost_Growth_Prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=f"XGBoost_Run_{datetime.now().strftime('%H%M%S')}"):
        if use_saved_params:
            # 저장된 파라미터를 사용하여 모델 학습 (튜닝 없이)
            print("[저장된 파라미터 사용 모드]")
            model, best_params = train_xgboost(X_train_split, y_train_split, X_val, y_val, 
                                              use_tuning=False, cv=5, use_saved_params=True)
        else:
            # 하이퍼파라미터 튜닝 실행 (메인 run 컨텍스트 내에서 중첩 run으로 저장됨)
            print("[하이퍼파라미터 튜닝 모드]")
            model, best_params = train_xgboost(X_train_split, y_train_split, X_val, y_val, use_tuning=True, cv=5)
        
        # MLflow에 모델 및 파라미터 로깅
        mlflow.log_params(best_params)
        mlflow.log_param("threshold", 0.3)
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("n_estimators", 1000)
        
        # 최적 하이퍼파라미터 저장
        script_dir = os.path.dirname(os.path.abspath(__file__))
        params_path = os.path.join(script_dir, 'best_params.txt')
        with open(params_path, 'w', encoding='utf-8') as f:
            f.write("최적 하이퍼파라미터:\n")
            for key, value in best_params.items():
                f.write(f"{key}: {value}\n")
        print(f"\n최적 하이퍼파라미터 저장: {params_path}")
        
        # 6. 모델 평가 (임계값 0.4로 Recall 향상)
        print("\n[6단계] 모델 평가")
        print("  임계값(Threshold): 0.3 (Recall 향상을 위해 추가 조정)")
        y_train_pred, y_train_pred_proba, train_metrics = evaluate_model(model, X_train_split, y_train_split, 'Train', threshold=0.3)
        y_val_pred, y_val_pred_proba, val_metrics = evaluate_model(model, X_val, y_val, 'Validation', threshold=0.3)
        
        # MLflow에 평가 지표 로깅
        mlflow.log_metrics({
            "train_accuracy": train_metrics['accuracy'],
            "train_precision": train_metrics['precision'],
            "train_recall": train_metrics['recall'],
            "train_f1": train_metrics['f1'],
            "train_roc_auc": train_metrics['roc_auc'],
            "val_accuracy": val_metrics['accuracy'],
            "val_precision": val_metrics['precision'],
            "val_recall": val_metrics['recall'],
            "val_f1": val_metrics['f1'],
            "val_roc_auc": val_metrics['roc_auc']
        })
    
    # 모델 파라미터 준비 (롤링 검증 및 GroupKFold용)
    model_params_for_validation = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'booster': 'gbtree',
        'random_state': 42,
        'verbosity': 0,
    }
    if best_params:
        model_params_for_validation.update(best_params)
    
    # 롤링 검증 (개발/튜닝): 2021→2022→2023→2024 순서로 연도 단위 rolling/expanding validation
    print("\n[롤링 검증] 연도 단위 rolling/expanding validation (2021→2022→2023→2024)")
    rolling_results = rolling_validation(X, y, features_df_filtered, model_params_for_validation, years=[2021, 2022, 2023, 2024])
    
    # GroupKFold 검증 (기업 일반화 점검): 기업 단위로 분할하여 미관측 기업 일반화 강건성 점검
    print("\n[GroupKFold 검증] 기업 단위 분할로 미관측 기업 일반화 강건성 점검")
    groupkfold_results = groupkfold_validation(X_train_all, y_train_all, features_df_filtered[train_mask], model_params_for_validation, n_splits=5)
    
    # 최종 테스트 (시간 홀드아웃): test_dataset_patent_feature_add.csv 사용
    holdout_results = {}
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    # test_dataset_patent_feature_add.csv 로드
    if use_train_dataset:
        print("\n[최종 테스트] test_dataset_patent_feature_add.csv 사용")
        test_df = load_test_dataset(project_root)
        
        # test가 비어있으면 train의 마지막 연도(2024)를 holdout으로 사용
        if len(test_df) == 0 or 'target_growth' not in test_df.columns:
            print("  → test_dataset.csv가 비어있음")
            print("  → train의 마지막 연도(2024년 타겟)를 최종 holdout으로 사용")
            
            # train에서 2024년 타겟 데이터만 추출
            holdout_mask = features_df_filtered['target_year'] == 2024
            X_holdout = X[holdout_mask]
            y_holdout = y[holdout_mask]
            
            if len(X_holdout) > 0:
                # 예측 수행
                dtest = xgb.DMatrix(X_holdout)
                y_test_pred_proba = model.predict(dtest)
                y_test_pred = (y_test_pred_proba >= 0.4).astype(int)
                y_test_actual = y_holdout.values
                y_test_actual_series = pd.Series(y_test_actual)
                
                # 모든 지표로 평가
                holdout_results = {
                    'pr_auc': average_precision_score(y_test_actual, y_test_pred_proba) if len(np.unique(y_test_actual)) > 1 else 0.0,
                    'top20_precision': calculate_top_k_precision(y_test_actual_series, y_test_pred_proba, k=20),
                    'top50_precision': calculate_top_k_precision(y_test_actual_series, y_test_pred_proba, k=50),
                    'f1': f1_score(y_test_actual, y_test_pred, zero_division=0),
                    'brier': brier_score_loss(y_test_actual, y_test_pred_proba),
                    'roc_auc': roc_auc_score(y_test_actual, y_test_pred_proba) if len(np.unique(y_test_actual)) > 1 else 0.0
                }
                
                print(f"  PR-AUC: {holdout_results['pr_auc']:.4f}")
                print(f"  Top-20 Precision: {holdout_results['top20_precision']:.4f}")
                print(f"  Top-50 Precision: {holdout_results['top50_precision']:.4f}")
                print(f"  F1-Score: {holdout_results['f1']:.4f}")
                print(f"  Brier Score: {holdout_results['brier']:.4f}")
                print(f"  ROC-AUC: {holdout_results['roc_auc']:.4f}")
                
                # 결과 저장
                holdout_df = features_df_filtered[holdout_mask][['기업명', '연도']].copy()
                holdout_df['predicted_top30'] = y_test_pred
                holdout_df['predicted_probability'] = y_test_pred_proba
                holdout_df['actual_top30'] = y_test_actual
                
                test_output_path = os.path.join(script_dir, 'test_predictions_2024.csv')
                holdout_df.to_csv(test_output_path, index=False, encoding='utf-8-sig')
                print(f"  예측 결과 저장: {test_output_path}")
        
        elif len(test_df) > 0 and 'target_growth' in test_df.columns:
            # test_df에서 target_growth를 target_top30으로 변환
            test_df['target_top30'] = test_df['target_growth']
            
            # test 데이터 준비 (train과 동일한 피처 컬럼만 사용)
            X_test_df, y_test_df, _ = prepare_data(test_df)
            
            if len(X_test_df) > 0:
                # 예측 수행
                dtest = xgb.DMatrix(X_test_df)
                y_test_pred_proba = model.predict(dtest)
                y_test_pred = (y_test_pred_proba >= 0.4).astype(int)
                y_test_actual = y_test_df.values
                y_test_actual_series = pd.Series(y_test_actual)
                
                # 모든 지표로 평가
                holdout_results = {
                    'pr_auc': average_precision_score(y_test_actual, y_test_pred_proba) if len(np.unique(y_test_actual)) > 1 else 0.0,
                    'top20_precision': calculate_top_k_precision(y_test_actual_series, y_test_pred_proba, k=20),
                    'top50_precision': calculate_top_k_precision(y_test_actual_series, y_test_pred_proba, k=50),
                    'f1': f1_score(y_test_actual, y_test_pred, zero_division=0),
                    'brier': brier_score_loss(y_test_actual, y_test_pred_proba),
                    'roc_auc': roc_auc_score(y_test_actual, y_test_pred_proba) if len(np.unique(y_test_actual)) > 1 else 0.0
                }
                
                print(f"  PR-AUC: {holdout_results['pr_auc']:.4f}")
                print(f"  Top-20 Precision: {holdout_results['top20_precision']:.4f}")
                print(f"  Top-50 Precision: {holdout_results['top50_precision']:.4f}")
                print(f"  F1-Score: {holdout_results['f1']:.4f}")
                print(f"  Brier Score: {holdout_results['brier']:.4f}")
                print(f"  ROC-AUC: {holdout_results['roc_auc']:.4f}")
                
                # 결과 저장 (인덱스 매칭)
                # prepare_data에서 결측치가 제거되었으므로 원본 test_df의 해당 인덱스만 사용
                test_output_df = test_df.loc[X_test_df.index, ['기업명', '연도']].copy()
                test_output_df['predicted_top30'] = y_test_pred
                test_output_df['predicted_probability'] = y_test_pred_proba
                test_output_df['actual_top30'] = test_df.loc[X_test_df.index, 'target_top30'].values
                
                test_output_path = os.path.join(script_dir, 'test_predictions_2024.csv')
                test_output_df.to_csv(test_output_path, index=False, encoding='utf-8-sig')
                print(f"  예측 결과 저장: {test_output_path}")
    else:
        # 기존 방식: 2023년 피처로 2024년 타겟 예측
        if len(X_test_2024) > 0:
            print("\n[최종 테스트] 시간 홀드아웃 (2023년 피처 → 2024년 타겟)")
            actual_target_df = load_actual_target_from_csv(project_root, target_year=2024)
            
            if len(actual_target_df) > 0:
                # 2023년 피처로 예측
                dtest = xgb.DMatrix(X_test_2024)
                y_test_pred_proba = model.predict(dtest)
                
                # 기업명 매칭
                test_df_2023 = features_df_filtered[test_mask_2023].copy()
                test_df_2023['기업명_정규화'] = test_df_2023['기업명'].str.strip()
                test_df_2023 = test_df_2023.merge(
                    actual_target_df[['기업명_정규화', 'y']],
                    on='기업명_정규화',
                    how='left'
                )
                
                matched_mask = test_df_2023['y'].notna()
                if matched_mask.sum() > 0:
                    y_test_actual = test_df_2023.loc[matched_mask, 'y'].values
                    y_test_pred_proba_matched = y_test_pred_proba[matched_mask]
                    y_test_actual_series = pd.Series(y_test_actual)
                    
                    # 모든 지표로 평가
                    holdout_results = {
                        'pr_auc': average_precision_score(y_test_actual, y_test_pred_proba_matched) if len(np.unique(y_test_actual)) > 1 else 0.0,
                        'top20_precision': calculate_top_k_precision(y_test_actual_series, y_test_pred_proba_matched, k=20),
                        'top50_precision': calculate_top_k_precision(y_test_actual_series, y_test_pred_proba_matched, k=50),
                        'f1': f1_score(y_test_actual, (y_test_pred_proba_matched >= 0.4).astype(int), zero_division=0),
                        'brier': brier_score_loss(y_test_actual, y_test_pred_proba_matched),
                        'roc_auc': roc_auc_score(y_test_actual, y_test_pred_proba_matched) if len(np.unique(y_test_actual)) > 1 else 0.0
                    }
                    
                    print(f"  PR-AUC: {holdout_results['pr_auc']:.4f}")
                    print(f"  Top-20 Precision: {holdout_results['top20_precision']:.4f}")
                    print(f"  Top-50 Precision: {holdout_results['top50_precision']:.4f}")
                    print(f"  F1-Score: {holdout_results['f1']:.4f}")
                    print(f"  Brier Score: {holdout_results['brier']:.4f}")
                    print(f"  ROC-AUC: {holdout_results['roc_auc']:.4f}")
                    
                    # test_df 업데이트
                    test_df_2023.loc[matched_mask, 'predicted_probability'] = y_test_pred_proba_matched
                    test_df_2023.loc[matched_mask, 'predicted_top30'] = (y_test_pred_proba_matched >= 0.4).astype(int)
                    test_df_2023.loc[matched_mask, 'actual_top30'] = y_test_actual
                    
                    # 결과 저장
                    test_output_path = os.path.join(script_dir, 'test_predictions_2024.csv')
                    output_cols = ['기업명', '연도', 'predicted_top30', 'predicted_probability', 'actual_top30']
                    test_df_2023[output_cols].to_csv(test_output_path, index=False, encoding='utf-8-sig')
                    print(f"  예측 결과 저장: {test_output_path}")
    
    # 평가 결과 종합 표 출력 및 저장
    print("\n" + "="*80)
    print("[12단계] 평가 결과 종합 표")
    print("="*80)
    final_results = print_evaluation_table(holdout_results, rolling_results, groupkfold_results)
    
    # 평가 결과 표를 CSV 파일로 저장
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_table_path = os.path.join(script_dir, 'evaluation_results_table.csv')
    
    # 표 형식으로 저장 (모든 지표 포함)
    metrics_list = ['pr_auc', 'top-20-precision', 'top-50-precision', 'f1_score', 'brier_score', 'roc_auc']
    results_dict = {
        'model': ['XGB'] * len(metrics_list),
        'metric': metrics_list,
        'groupkfold': [
            final_results['groupkfold_pr_auc'],
            final_results['groupkfold_top20_precision'],
            final_results['groupkfold_top50_precision'],
            final_results['groupkfold_f1_score'],
            final_results['groupkfold_brier_score'],
            final_results['groupkfold_roc_auc']
        ],
        'holdout': [
            final_results['holdout_pr_auc'],
            final_results['holdout_top20_precision'],
            final_results['holdout_top50_precision'],
            final_results['holdout_f1_score'],
            final_results['holdout_brier_score'],
            final_results['holdout_roc_auc']
        ],
        'rolling': [
            final_results['rolling_pr_auc'],
            final_results['rolling_top20_precision'],
            final_results['rolling_top50_precision'],
            final_results['rolling_f1_score'],
            final_results['rolling_brier_score'],
            final_results['rolling_roc_auc']
        ]
    }
    
    results_df = pd.DataFrame(results_dict)
    results_df.to_csv(results_table_path, index=False, encoding='utf-8-sig')
    print(f"\n평가 결과 표가 저장되었습니다: {results_table_path}")
    
    # 7. 피처 중요도 시각화
    print("\n[7단계] 피처 중요도 분석")
    feature_importance_df = plot_feature_importance(model, feature_cols)
    print("\nTop 10 피처 중요도:")
    print(feature_importance_df.head(10))
    
    # 8. 예측 결과 시각화
    print("\n[8단계] 예측 결과 시각화")
    plot_predictions(y_train_split, y_train_pred, y_train_pred_proba, 'Train')
    plot_predictions(y_val, y_val_pred, y_val_pred_proba, 'Validation')
    
    # 9. 성능 보고서 저장
    print("\n[9단계] 성능 보고서 저장")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_performance_report(
        train_metrics, val_metrics, None,  # test_metrics는 None으로 전달 (최종 테스트는 별도로 처리)
        best_params, feature_importance_df, script_dir, model_name='XGBoost'
    )
    
    # 10. 모델 저장
    print("\n[10단계] 모델 저장")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'xgboost_model.json')
    model.save_model(model_path)
    print(f"모델 저장 완료: {model_path}")
    
    print("\n" + "="*70)
    print("모델링 완료!")
    print("="*70)
    
    return model, feature_importance_df, {
        'train': train_metrics,
        'validation': val_metrics,
        'holdout': holdout_results,
        'rolling': rolling_results,
        'groupkfold': groupkfold_results,
        'final_table': final_results
    }


if __name__ == "__main__":
    model, feature_importance, metrics = main()
