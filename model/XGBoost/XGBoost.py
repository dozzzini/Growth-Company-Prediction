import pandas as pd
import numpy as np
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix, 
                            classification_report, roc_curve, precision_recall_curve)
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
import warnings
warnings.filterwarnings('ignore')


def load_data():
    """데이터 로드"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    # 피처 데이터 로드
    features_path = os.path.join(project_root, 'data', 'growth_features_rolling.csv')
    features_df = pd.read_csv(features_path, encoding='utf-8-sig')
    
    # 원본 재무정보 데이터 로드 (타겟 변수 생성용)
    financial_path = os.path.join(project_root, 'data', '재무정보_final_imputed.csv')
    financial_df = pd.read_csv(financial_path, encoding='cp949')
    
    return features_df, financial_df


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
    # 기업명, 연도, growth_rate, target_year는 제외
    feature_cols = [col for col in df.columns if col not in ['기업명', '연도', 'growth_rate', 'target_year', 'target_top30']]
    
    X = df[feature_cols].copy()
    y = df['target_top30'].copy()
    
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


def optimize_xgboost_hyperparameters(X_train, y_train, X_val, y_val, n_trials=50):
    """Optuna를 사용한 XGBoost 하이퍼파라미터 최적화"""
    def objective(trial):
        # 하이퍼파라미터 탐색 공간 정의
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'booster': 'gbtree',
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_float('min_child_weight', 0.1, 10.0, log=True),
            'gamma': trial.suggest_float('gamma', 0.001, 10.0, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.001, 10.0, log=True),
            'random_state': 42,
            'verbosity': 0
        }
        
        # DMatrix 생성
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        # 모델 학습
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            evals=[(dval, 'eval')],
            early_stopping_rounds=50,
            verbose_eval=False
        )
        
        # 검증 데이터로 예측 및 평가
        y_pred_proba = model.predict(dval)
        roc_auc = roc_auc_score(y_val, y_pred_proba)
        
        return roc_auc
    
    print(f"\n하이퍼파라미터 최적화 시작 (n_trials={n_trials})...")
    study = optuna.create_study(direction='maximize', study_name='xgboost_optimization')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"\n최적화 완료!")
    print(f"Best ROC-AUC: {study.best_value:.4f}")
    print(f"\n최적 하이퍼파라미터:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    return study.best_params


def train_xgboost(X_train, y_train, X_val, y_val, use_tuning=True, n_trials=50):
    """XGBoost 모델 학습 (이진 분류)"""
    # 하이퍼파라미터 튜닝
    if use_tuning:
        best_params = optimize_xgboost_hyperparameters(X_train, y_train, X_val, y_val, n_trials)
        # 기본 파라미터와 최적 파라미터 병합
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'booster': 'gbtree',
            'random_state': 42,
            'verbosity': 0,
            **best_params
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


def evaluate_model(model, X, y, split_name='', threshold=0.4):
    """모델 평가 (이진 분류)"""
    dtest = xgb.DMatrix(X)
    # 확률 예측
    y_pred_proba = model.predict(dtest)
    # 이진 예측 (임계값 기준, 기본값 0.4로 Recall 향상)
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # 평가 지표 계산
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y, y_pred_proba)
    
    print(f"\n{split_name} 평가 결과:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  ROC-AUC: {roc_auc:.4f}")
    
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
        'roc_auc': roc_auc
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
    axes[1, 0].axvline(x=0.4, color='green', linestyle='--', label='Threshold (0.4)')
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
    
    # 1. 데이터 로드
    print("\n[1단계] 데이터 로드")
    features_df, financial_df = load_data()
    print(f"피처 데이터 shape: {features_df.shape}")
    print(f"재무정보 데이터 shape: {financial_df.shape}")
    
    # 2. 타겟 변수 생성
    print("\n[2단계] 타겟 변수 생성 (상위 30% 이진 분류)")
    features_df = create_target_variable(features_df, financial_df)
    print(f"타겟 변수 생성 완료. 결측치: {features_df['target_top30'].isnull().sum()}개")
    
    # 3. 데이터 준비
    print("\n[3단계] 데이터 준비")
    X, y, feature_cols = prepare_data(features_df)
    
    # 4. Train/Test 분리 (연도 기준)
    # 2019-2022년 말 피처는 Train, 2023년 말 피처는 Test (2024년 예측용)
    print("\n[4단계] Train/Test 분리")
    print("  - Train: 2019년 말, 2020년 말, 2021년 말, 2022년 말 피처")
    print("  - Test: 2023년 말 피처 → 2024년 타겟 예측")
    # 인덱스 매핑을 위해 features_df와 X의 인덱스 동기화
    features_df_filtered = features_df.loc[X.index].copy()
    train_mask = features_df_filtered['연도'] < 2023  # 2019-2022년 말 피처
    test_mask = features_df_filtered['연도'] == 2023    # 2023년 말 피처 → 2024년 예측
    
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask] if test_mask.sum() > 0 else pd.DataFrame()
    y_test = y[test_mask] if test_mask.sum() > 0 else pd.Series(dtype=float)
    
    print(f"Train 데이터: {len(X_train)}개")
    print(f"Test 데이터: {len(X_test)}개")
    
    # Train 데이터를 다시 Train/Validation으로 분리
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    print(f"Train (학습용): {len(X_train_split)}개")
    print(f"Validation: {len(X_val)}개")
    
    # 5. 모델 학습
    print("\n[5단계] XGBoost 모델 학습")
    model, best_params = train_xgboost(X_train_split, y_train_split, X_val, y_val, use_tuning=True, n_trials=50)
    
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
    print("  임계값(Threshold): 0.4 (Recall 향상을 위해 0.5에서 조정)")
    y_train_pred, y_train_pred_proba, train_metrics = evaluate_model(model, X_train_split, y_train_split, 'Train', threshold=0.4)
    y_val_pred, y_val_pred_proba, val_metrics = evaluate_model(model, X_val, y_val, 'Validation', threshold=0.4)
    
    # Test 데이터가 있는 경우에만 평가
    if len(X_test) > 0:
        y_test_pred, y_test_pred_proba, test_metrics = evaluate_model(model, X_test, y_test, 'Test', threshold=0.4)
    else:
        print("\nTest 데이터가 없어 Test 평가를 건너뜁니다.")
        test_metrics = None
        y_test_pred = None
        y_test_pred_proba = None
    
    # 7. 피처 중요도 시각화
    print("\n[7단계] 피처 중요도 분석")
    feature_importance_df = plot_feature_importance(model, feature_cols)
    print("\nTop 10 피처 중요도:")
    print(feature_importance_df.head(10))
    
    # 8. 예측 결과 시각화
    print("\n[8단계] 예측 결과 시각화")
    plot_predictions(y_train_split, y_train_pred, y_train_pred_proba, 'Train')
    plot_predictions(y_val, y_val_pred, y_val_pred_proba, 'Validation')
    if y_test_pred is not None:
        plot_predictions(y_test, y_test_pred, y_test_pred_proba, 'Test')
    
    # 9. 모델 저장
    print("\n[9단계] 모델 저장")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'xgboost_model.json')
    model.save_model(model_path)
    print(f"모델 저장 완료: {model_path}")
    
    # 10. 최종 테스트 결과 요약 (2023년 피처 → 2024년 예측)
    if len(X_test) > 0:
        print("\n" + "="*70)
        print("최종 테스트 결과 요약 (2023년 피처 → 2024년 예측)")
        print("="*70)
        test_df = features_df_filtered[test_mask].copy()
        test_df['predicted_top30'] = y_test_pred
        test_df['predicted_probability'] = y_test_pred_proba
        test_df['actual_top30'] = y_test
        
        print(f"\n예측 결과:")
        print(f"  상위 30%로 예측된 기업: {(y_test_pred == 1).sum()}개")
        print(f"  실제 상위 30% 기업: {(y_test == 1).sum()}개")
        print(f"  정확도: {test_metrics['accuracy']:.4f}")
        print(f"  F1-Score: {test_metrics['f1']:.4f}")
        
        # 상위 30%로 예측된 기업 목록 저장
        script_dir = os.path.dirname(os.path.abspath(__file__))
        test_output_path = os.path.join(script_dir, 'test_predictions_2024.csv')
        test_df[['기업명', '연도', 'predicted_top30', 'predicted_probability', 'actual_top30']].to_csv(
            test_output_path, index=False, encoding='utf-8-sig'
        )
        print(f"\n예측 결과 저장: {test_output_path}")
    
    print("\n" + "="*70)
    print("모델링 완료!")
    print("="*70)
    
    return model, feature_importance_df, {
        'train': train_metrics,
        'validation': val_metrics,
        'test': test_metrics
    }


if __name__ == "__main__":
    model, feature_importance, metrics = main()
