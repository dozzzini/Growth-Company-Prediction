import pandas as pd
import numpy as np
import os
import sys
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix, 
                            classification_report, average_precision_score,
                            brier_score_loss, precision_recall_curve)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# LightGBM과 XGBoost의 데이터 로딩 함수 재사용
# sys.path를 통해 상위 디렉토리의 모듈 import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from LightGBM.LightGBM import load_data, load_actual_target_from_csv, create_target_variable, prepare_data, calculate_top_k_precision, load_test_dataset


def load_both_models(use_pretrained=True):
    """LightGBM과 XGBoost 모델 로드 (사전 학습된 모델 또는 새로 학습)"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    if use_pretrained:
        # 사전 학습된 모델 로드
        print("[사전 학습된 모델 로드]")
        lgbm_path = os.path.join(project_root, 'model', 'LightGBM', 'lightgbm_model.txt')
        xgb_path = os.path.join(project_root, 'model', 'XGBoost', 'xgboost_model.json')
        
        if os.path.exists(lgbm_path):
            lgbm_model = lgb.Booster(model_file=lgbm_path)
            print(f"  LightGBM 모델 로드: {lgbm_path}")
        else:
            raise FileNotFoundError(f"LightGBM 모델 파일을 찾을 수 없습니다: {lgbm_path}")
        
        if os.path.exists(xgb_path):
            xgb_model = xgb.Booster(model_file=xgb_path)
            print(f"  XGBoost 모델 로드: {xgb_path}")
        else:
            raise FileNotFoundError(f"XGBoost 모델 파일을 찾을 수 없습니다: {xgb_path}")
        
        return lgbm_model, xgb_model
    else:
        # 새로 학습 (여기서는 구현하지 않음, 필요시 추가)
        raise NotImplementedError("새로운 모델 학습은 아직 구현되지 않았습니다. use_pretrained=True로 설정하세요.")


def load_model_params():
    """LightGBM과 XGBoost의 최적 하이퍼파라미터 로드"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    # LightGBM 파라미터 로드
    lgbm_params_path = os.path.join(project_root, 'model', 'LightGBM', 'best_params.txt')
    lgbm_params = {}
    if os.path.exists(lgbm_params_path):
        with open(lgbm_params_path, 'r', encoding='utf-8') as f:
            for line in f:
                if ':' in line and not line.startswith('최적'):
                    key, value = line.strip().split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    # 숫자 변환 시도
                    try:
                        if '.' in value:
                            value = float(value)
                        else:
                            value = int(value)
                    except ValueError:
                        pass
                    lgbm_params[key] = value
    
    # XGBoost 파라미터 로드
    xgb_params_path = os.path.join(project_root, 'model', 'XGBoost', 'best_params.txt')
    xgb_params = {}
    if os.path.exists(xgb_params_path):
        with open(xgb_params_path, 'r', encoding='utf-8') as f:
            for line in f:
                if ':' in line and not line.startswith('최적'):
                    key, value = line.strip().split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    # 숫자 변환 시도
                    try:
                        if '.' in value:
                            value = float(value)
                        else:
                            value = int(value)
                    except ValueError:
                        pass
                    xgb_params[key] = value
    
    # 기본 파라미터 설정
    lgbm_params_base = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'verbose': -1,
        'random_state': 42,
    }
    lgbm_params_base.update({k: v for k, v in lgbm_params.items() if k not in ['objective', 'metric', 'boosting_type', 'verbose']})
    
    xgb_params_base = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'booster': 'gbtree',
        'random_state': 42,
        'verbosity': 0,
    }
    xgb_params_base.update({k: v for k, v in xgb_params.items() if k not in ['objective', 'eval_metric', 'booster', 'verbosity']})
    
    return lgbm_params_base, xgb_params_base


def ensemble_predict(lgbm_model, xgb_model, X, xgb_weight=0.628, threshold=0.4, return_individual=False):
    """
    앙상블 예측 (XGBoost와 LightGBM 결합)
    
    Parameters:
    -----------
    lgbm_model : lightgbm.Booster
        LightGBM 모델
    xgb_model : xgboost.Booster
        XGBoost 모델
    X : pd.DataFrame or np.ndarray
        입력 데이터
    xgb_weight : float
        XGBoost 가중치 (기본값: 0.628, 검증 성능 기반)
    threshold : float
        이진 분류 임계값 (기본값: 0.4)
    return_individual : bool
        True이면 개별 모델 예측도 반환
    
    Returns:
    --------
    ensemble_pred : np.ndarray
        이진 예측 (0 또는 1)
    ensemble_pred_proba : np.ndarray
        확률 예측 (0~1)
    (optional) xgb_pred_proba, lgbm_pred_proba : np.ndarray
        개별 모델 확률 예측 (return_individual=True일 때)
    """
    # 빈 데이터 체크
    if isinstance(X, pd.DataFrame):
        if len(X) == 0 or X.empty:
            raise ValueError("입력 데이터가 비어있습니다.")
        dtest = xgb.DMatrix(X)
        X_for_lgbm = X
    else:
        if len(X) == 0:
            raise ValueError("입력 데이터가 비어있습니다.")
        X_df = pd.DataFrame(X)
        dtest = xgb.DMatrix(X_df)
        X_for_lgbm = X_df
    
    # XGBoost 예측
    xgb_pred_proba = xgb_model.predict(dtest)
    
    # LightGBM 예측
    lgbm_pred_proba = lgbm_model.predict(X_for_lgbm, num_iteration=lgbm_model.best_iteration)
    
    # 가중 평균
    lgbm_weight = 1 - xgb_weight
    ensemble_pred_proba = xgb_weight * xgb_pred_proba + lgbm_weight * lgbm_pred_proba
    
    # 이진 예측
    ensemble_pred = (ensemble_pred_proba >= threshold).astype(int)
    
    if return_individual:
        return ensemble_pred, ensemble_pred_proba, xgb_pred_proba, lgbm_pred_proba
    else:
        return ensemble_pred, ensemble_pred_proba


def calculate_optimal_weights(xgb_pred_proba, lgbm_pred_proba, y_true, metric='pr_auc'):
    """
    검증 데이터로 최적 가중치 계산
    
    Parameters:
    -----------
    xgb_pred_proba : np.ndarray
        XGBoost 확률 예측
    lgbm_pred_proba : np.ndarray
        LightGBM 확률 예측
    y_true : np.ndarray or pd.Series
        실제 타겟 값
    metric : str
        최적화 지표 ('pr_auc', 'roc_auc', 'f1')
    
    Returns:
    --------
    best_weight : float
        최적 XGBoost 가중치
    best_score : float
        최적 점수
    """
    best_weight = 0.5
    best_score = 0
    
    print(f"\n[최적 가중치 탐색] 지표: {metric}")
    for w in np.arange(0.0, 1.1, 0.1):
        ensemble_pred = w * xgb_pred_proba + (1 - w) * lgbm_pred_proba
        
        if metric == 'pr_auc':
            score = average_precision_score(y_true, ensemble_pred) if len(np.unique(y_true)) > 1 else 0.0
        elif metric == 'roc_auc':
            score = roc_auc_score(y_true, ensemble_pred) if len(np.unique(y_true)) > 1 else 0.0
        elif metric == 'f1':
            threshold = 0.4
            y_pred = (ensemble_pred >= threshold).astype(int)
            score = f1_score(y_true, y_pred, zero_division=0)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        print(f"  가중치 {w:.1f}: {metric} = {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_weight = w
    
    print(f"\n최적 가중치: XGBoost={best_weight:.2f}, LightGBM={1-best_weight:.2f}, {metric}={best_score:.4f}")
    return best_weight, best_score


def evaluate_ensemble(y_true, y_pred, y_pred_proba, model_name='Ensemble', threshold=0.4):
    """앙상블 모델 평가 (모든 지표 포함)"""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else 0.0,
        'pr_auc': average_precision_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else 0.0,
        'brier': brier_score_loss(y_true, y_pred_proba)
    }
    
    # Top-K Precision
    y_series = y_true if isinstance(y_true, pd.Series) else pd.Series(y_true)
    metrics['top20_precision'] = calculate_top_k_precision(y_series, y_pred_proba, k=20)
    metrics['top50_precision'] = calculate_top_k_precision(y_series, y_pred_proba, k=50)
    
    print(f"\n[{model_name} 평가 결과]")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1-Score: {metrics['f1']:.4f}")
    print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"  PR-AUC: {metrics['pr_auc']:.4f}")
    print(f"  Brier Score: {metrics['brier']:.4f}")
    print(f"  Top-20 Precision: {metrics['top20_precision']:.4f}")
    print(f"  Top-50 Precision: {metrics['top50_precision']:.4f}")
    
    return metrics


def rolling_validation_ensemble(X, y, features_df_filtered, lgbm_params, xgb_params, years=[2021, 2022, 2023, 2024], xgb_weight=0.628):
    """
    롤링 검증: 앙상블 모델로 연도 단위 rolling/expanding validation 수행
    """
    results = []
    
    for i, val_year in enumerate(years):
        print(f"\n[롤링 검증] {val_year}년 평가 (앙상블)")
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
        
        # LightGBM 모델 학습
        train_data_lgbm = lgb.Dataset(X_train_roll, label=y_train_roll)
        val_data_lgbm = lgb.Dataset(X_val_roll, label=y_val_roll, reference=train_data_lgbm)
        lgbm_model = lgb.train(
            lgbm_params,
            train_data_lgbm,
            valid_sets=[train_data_lgbm, val_data_lgbm],
            valid_names=['train', 'valid'],
            num_boost_round=1000,
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0)
            ]
        )
        
        # XGBoost 모델 학습
        dtrain_xgb = xgb.DMatrix(X_train_roll, label=y_train_roll)
        dval_xgb = xgb.DMatrix(X_val_roll, label=y_val_roll)
        xgb_model = xgb.train(
            xgb_params,
            dtrain_xgb,
            num_boost_round=1000,
            evals=[(dtrain_xgb, 'train'), (dval_xgb, 'eval')],
            early_stopping_rounds=50,
            verbose_eval=False
        )
        
        # 앙상블 예측
        _, ensemble_pred_proba = ensemble_predict(lgbm_model, xgb_model, X_val_roll, xgb_weight=xgb_weight)
        y_series = y_val_roll if isinstance(y_val_roll, pd.Series) else pd.Series(y_val_roll)
        
        pr_auc = average_precision_score(y_val_roll, ensemble_pred_proba) if len(np.unique(y_val_roll)) > 1 else 0.0
        brier = brier_score_loss(y_val_roll, ensemble_pred_proba)
        roc_auc = roc_auc_score(y_val_roll, ensemble_pred_proba) if len(np.unique(y_val_roll)) > 1 else 0.0
        f1 = f1_score(y_val_roll, (ensemble_pred_proba >= 0.4).astype(int), zero_division=0)
        top20_precision = calculate_top_k_precision(y_series, ensemble_pred_proba, k=20)
        top50_precision = calculate_top_k_precision(y_series, ensemble_pred_proba, k=50)
        
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


def groupkfold_validation_ensemble(X, y, features_df_filtered, lgbm_params, xgb_params, n_splits=5, xgb_weight=0.628):
    """
    GroupKFold 검증: 앙상블 모델로 기업 단위 분할하여 미관측 기업 일반화 강건성 점검
    """
    print(f"\n[GroupKFold 검증] 기업 단위 {n_splits}-fold (앙상블)")
    print("-" * 70)
    
    # 기업명 추출
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
        
        # LightGBM 모델 학습
        train_data_lgbm = lgb.Dataset(X_train_fold, label=y_train_fold)
        val_data_lgbm = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data_lgbm)
        lgbm_model = lgb.train(
            lgbm_params,
            train_data_lgbm,
            valid_sets=[train_data_lgbm, val_data_lgbm],
            valid_names=['train', 'valid'],
            num_boost_round=1000,
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0)
            ]
        )
        
        # XGBoost 모델 학습
        dtrain_xgb = xgb.DMatrix(X_train_fold, label=y_train_fold)
        dval_xgb = xgb.DMatrix(X_val_fold, label=y_val_fold)
        xgb_model = xgb.train(
            xgb_params,
            dtrain_xgb,
            num_boost_round=1000,
            evals=[(dtrain_xgb, 'train'), (dval_xgb, 'eval')],
            early_stopping_rounds=50,
            verbose_eval=False
        )
        
        # 앙상블 예측
        _, ensemble_pred_proba = ensemble_predict(lgbm_model, xgb_model, X_val_fold, xgb_weight=xgb_weight)
        y_series_fold = y_val_fold if isinstance(y_val_fold, pd.Series) else pd.Series(y_val_fold)
        
        pr_auc = average_precision_score(y_val_fold, ensemble_pred_proba) if len(np.unique(y_val_fold)) > 1 else 0.0
        brier = brier_score_loss(y_val_fold, ensemble_pred_proba)
        roc_auc = roc_auc_score(y_val_fold, ensemble_pred_proba) if len(np.unique(y_val_fold)) > 1 else 0.0
        f1 = f1_score(y_val_fold, (ensemble_pred_proba >= 0.4).astype(int), zero_division=0)
        top20_precision = calculate_top_k_precision(y_series_fold, ensemble_pred_proba, k=20)
        top50_precision = calculate_top_k_precision(y_series_fold, ensemble_pred_proba, k=50)
        
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
    """평가 결과를 표 형태로 출력"""
    print("\n" + "="*80)
    print("앙상블 모델 평가 결과 종합 표")
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
    print("\n모델: Ensemble (XGBoost + LightGBM)")
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
        'model': 'Ensemble',
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


def main():
    """메인 실행 함수"""
    print("="*70)
    print("XGBoost + LightGBM 앙상블 모델")
    print("="*70)
    
    # 1. 데이터 로드 (train_dataset.csv 사용)
    print("\n[1단계] 데이터 로드")
    use_train_dataset = True  # train_dataset.csv 사용
    features_df, financial_df = load_data(
        include_patent_features=False,  # 특허 피처 제외 (use_train_dataset=True일 때는 무시됨)
        use_patent_only=False,
        use_normalized_financial=True,  # 정규화된 재무 피처 사용 (use_train_dataset=True일 때는 무시됨)
        use_train_dataset=use_train_dataset
    )
    print(f"피처 데이터 shape: {features_df.shape}")
    if financial_df is not None:
        print(f"재무정보 데이터 shape: {financial_df.shape}")
    
    # 2. 타겟 변수 확인/생성
    if use_train_dataset:
        # train_dataset.csv를 사용하는 경우, target_growth가 이미 있음
        print("\n[2단계] 타겟 변수 확인 (train_dataset.csv 사용)")
        if 'target_growth' in features_df.columns:
            features_df['target_top30'] = features_df['target_growth']
            print(f"target_growth를 target_top30으로 변환 완료")
        if 'target_top30' in features_df.columns:
            print(f"타겟 변수 확인 완료. 결측치: {features_df['target_top30'].isnull().sum()}개")
        else:
            raise ValueError("타겟 변수를 찾을 수 없습니다.")
    else:
        # 기존 방식: 타겟 변수 생성
        print("\n[2단계] 타겟 변수 생성")
        if financial_df is None:
            raise ValueError("기존 방식을 사용하려면 financial_df가 필요합니다.")
        features_df = create_target_variable(features_df, financial_df)
        print(f"타겟 변수 생성 완료. 결측치: {features_df['target_top30'].isnull().sum()}개")
    
    # 3. 데이터 준비
    print("\n[3단계] 데이터 준비")
    X, y, feature_cols = prepare_data(features_df, use_patent_only=False)
    # prepare_data가 이미 결측치를 제거했으므로, X와 y의 인덱스를 사용하여 features_df 필터링
    features_df_filtered = features_df.loc[X.index].copy()
    
    # 4. Train/Test 분할 (2019-2023: Train, 최종 테스트는 test_dataset.csv 사용)
    print("\n[4단계] Train/Test 분할")
    train_mask = features_df_filtered['연도'] <= 2023  # 2019-2023년 피처
    val_mask = features_df_filtered['연도'] == 2023  # 2023년을 Validation으로도 사용
    
    X_train_all = X[train_mask]
    y_train_all = y[train_mask]
    X_val = X[val_mask]
    y_val = y[val_mask]
    
    # 최종 테스트용: use_train_dataset가 False일 때만 사용 (기존 방식)
    if not use_train_dataset:
        test_mask_2023 = features_df_filtered['연도'] == 2023  # 2023년 피처로 2024년 예측
        X_test_2024 = X[test_mask_2023]  # 2023년 피처를 사용하여 2024년 타겟 예측
    else:
        X_test_2024 = pd.DataFrame()  # test_dataset.csv를 사용하므로 빈 DataFrame
    
    print(f"  Train (2019-2023): {len(X_train_all)}개")
    print(f"  Validation (2023): {len(X_val)}개")
    if not use_train_dataset:
        print(f"  Test (2023 피처 → 2024 타겟): {len(X_test_2024)}개")
    
    # Train 내부 Train/Val 분할 (최종 모델 학습용)
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train_all, y_train_all, test_size=0.2, random_state=42, stratify=y_train_all
    )
    print(f"  Train Split: {len(X_train_split)}개, Val Split: {len(X_val_split)}개")
    
    # 5. 모델 파라미터 로드
    print("\n[5단계] 모델 파라미터 로드")
    lgbm_params, xgb_params = load_model_params()
    
    # 6. 최적 가중치 탐색 (검증 데이터 사용)
    print("\n[6단계] 최적 가중치 탐색 (검증 데이터)")
    
    # X_val이 비어있으면 X_val_split 사용
    if len(X_val) == 0:
        print("  경고: Validation 데이터가 비어있습니다. Train Split의 Validation 데이터를 사용합니다.")
        X_val = X_val_split
        y_val = y_val_split
    
    # 검증 데이터가 여전히 비어있는지 확인
    if len(X_val) == 0:
        print("  경고: 검증 데이터가 없습니다. 기본 가중치를 사용합니다.")
        optimal_xgb_weight = 0.628
        optimal_score = 0.0
    else:
        # 사전 학습된 모델로 검증 데이터 예측하여 가중치 탐색
        lgbm_model_val, xgb_model_val = load_both_models(use_pretrained=True)
        
        # 검증 데이터로 개별 모델 예측
        _, xgb_val_pred_proba = ensemble_predict(lgbm_model_val, xgb_model_val, X_val, xgb_weight=1.0, return_individual=False)
        # 재예측 (개별 예측 받기 위해)
        _, _, xgb_val_pred_proba_individual, lgbm_val_pred_proba_individual = ensemble_predict(
            lgbm_model_val, xgb_model_val, X_val, xgb_weight=1.0, return_individual=True
        )
        
        # 최적 가중치 탐색
        optimal_xgb_weight, optimal_score = calculate_optimal_weights(
            xgb_val_pred_proba_individual, lgbm_val_pred_proba_individual, y_val, metric='pr_auc'
        )
    
    # 7. 최종 모델 학습 (Train Split 사용)
    print("\n[7단계] 최종 모델 학습 (Train Split)")
    # 여기서는 사전 학습된 모델을 사용하므로 새로 학습하지 않음
    # 실제로는 Train Split으로 새로 학습할 수도 있음
    
    # 8. 검증 데이터 평가
    print("\n[8단계] 검증 데이터 평가 (앙상블)")
    if len(X_val) > 0:
        lgbm_model_final, xgb_model_final = load_both_models(use_pretrained=True)
        y_val_pred, y_val_pred_proba = ensemble_predict(lgbm_model_final, xgb_model_final, X_val, xgb_weight=optimal_xgb_weight)
        val_metrics = evaluate_ensemble(y_val, y_val_pred, y_val_pred_proba, model_name='Validation Ensemble')
    else:
        print("  경고: 검증 데이터가 없어 평가를 건너뜁니다.")
        val_metrics = {}
    
    # 9. 롤링 검증 및 GroupKFold 검증
    print("\n[9단계] 롤링 검증 및 GroupKFold 검증")
    rolling_results = rolling_validation_ensemble(
        X, y, features_df_filtered, lgbm_params, xgb_params, 
        years=[2021, 2022, 2023, 2024], xgb_weight=optimal_xgb_weight
    )
    
    groupkfold_results = groupkfold_validation_ensemble(
        X_train_all, y_train_all, features_df_filtered[train_mask], 
        lgbm_params, xgb_params, n_splits=5, xgb_weight=optimal_xgb_weight
    )
    
    # 10. 최종 테스트 (2024년 타겟 예측)
    holdout_results = {}
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    # test_dataset.csv 로드
    if use_train_dataset:
        print("\n[10단계] 최종 테스트 (test_dataset.csv 사용 - 2023년 피처 → 2024년 타겟)")
        test_df = load_test_dataset(project_root)
        
        if len(test_df) > 0 and 'target_growth' in test_df.columns:
            # test_df에서 target_growth를 target_top30으로 변환
            test_df['target_top30'] = test_df['target_growth']
            
            # test 데이터 준비 (train과 동일한 피처 컬럼만 사용)
            X_test_df, y_test_df, _ = prepare_data(test_df, use_patent_only=False)
            
            if len(X_test_df) > 0:
                # 앙상블 예측 수행
                y_test_pred, y_test_pred_proba = ensemble_predict(
                    lgbm_model_final, xgb_model_final, X_test_df, xgb_weight=optimal_xgb_weight
                )
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
                print(f"  Brier Score: {holdout_results['brier']:.4f}")
                print(f"  ROC-AUC: {holdout_results['roc_auc']:.4f}")
                print(f"  F1-Score: {holdout_results['f1']:.4f}")
                print(f"  Top-20 Precision: {holdout_results['top20_precision']:.4f}")
                print(f"  Top-50 Precision: {holdout_results['top50_precision']:.4f}")
                
                # 예측 결과 저장 (인덱스 매칭)
                # prepare_data에서 결측치가 제거되었으므로 원본 test_df의 해당 인덱스만 사용
                test_results_df = test_df.loc[X_test_df.index, ['기업명', '연도']].copy()
                test_results_df['predicted_proba'] = y_test_pred_proba
                test_results_df['predicted_top30'] = y_test_pred
                test_results_df['actual_top30'] = test_df.loc[X_test_df.index, 'target_top30'].values
                test_results_df['xgb_weight'] = optimal_xgb_weight
                test_results_df['lgbm_weight'] = 1 - optimal_xgb_weight
                
                output_path = os.path.join(script_dir, 'test_predictions_2024.csv')
                test_results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
                print(f"\n테스트 예측 결과 저장: {output_path}")
        else:
            print("  경고: test_dataset.csv를 로드할 수 없거나 target_growth 컬럼이 없습니다.")
    else:
        # 기존 방식: 2023년 피처로 2024년 타겟 예측
        if len(X_test_2024) > 0:
            print("\n[10단계] 최종 테스트 (2023년 피처 → 2024년 타겟)")
            actual_target_df = load_actual_target_from_csv(project_root, target_year=2024)
            
            if len(actual_target_df) > 0:
                # 2023년 피처로 앙상블 예측
                y_test_pred, y_test_pred_proba = ensemble_predict(
                    lgbm_model_final, xgb_model_final, X_test_2024, xgb_weight=optimal_xgb_weight
                )
                
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
                    print(f"  Brier Score: {holdout_results['brier']:.4f}")
                    print(f"  ROC-AUC: {holdout_results['roc_auc']:.4f}")
                    print(f"  F1-Score: {holdout_results['f1']:.4f}")
                    print(f"  Top-20 Precision: {holdout_results['top20_precision']:.4f}")
                    print(f"  Top-50 Precision: {holdout_results['top50_precision']:.4f}")
                    
                    # 예측 결과 저장
                    test_results_df = test_df_2023[['기업명', '연도']].copy()
                    test_results_df['predicted_proba'] = y_test_pred_proba
                    test_results_df['predicted_top30'] = y_test_pred
                    test_results_df['actual_top30'] = test_df_2023['y'].values
                    test_results_df['xgb_weight'] = optimal_xgb_weight
                    test_results_df['lgbm_weight'] = 1 - optimal_xgb_weight
                    
                    output_path = os.path.join(script_dir, 'test_predictions_2024.csv')
                    test_results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
                    print(f"\n테스트 예측 결과 저장: {output_path}")
    
    # 11. 평가 결과 표 출력 및 저장
    print("\n[11단계] 평가 결과 종합")
    final_results = print_evaluation_table(holdout_results, rolling_results, groupkfold_results)
    
    # CSV 저장
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_table_path = os.path.join(script_dir, 'evaluation_results_table.csv')
    results_df = pd.DataFrame([final_results])
    results_df.to_csv(results_table_path, index=False, encoding='utf-8-sig')
    print(f"\n평가 결과 표가 저장되었습니다: {results_table_path}")
    
    print("\n" + "="*70)
    print("앙상블 모델링 완료!")
    print("="*70)
    print(f"\n최적 가중치: XGBoost={optimal_xgb_weight:.3f}, LightGBM={1-optimal_xgb_weight:.3f}")
    
    return {
        'holdout': holdout_results,
        'rolling': rolling_results,
        'groupkfold': groupkfold_results,
        'final_table': final_results,
        'optimal_weight': optimal_xgb_weight
    }


if __name__ == "__main__":
    results = main()
