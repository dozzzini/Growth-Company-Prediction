# XGBoost 모델 성능 보고서

**생성 일시**: 2026년 01월 17일 17:46:10

---

## 하이퍼파라미터

| 파라미터 | 값 |
|---------|-----|
| objective | binary:logistic |
| eval_metric | logloss |
| booster | gbtree |
| random_state | 42 |
| verbosity | 0 |
| max_depth | 3 |
| min_child_weight | 10 |
| learning_rate | 0.01 |
| gamma | 1.0 |
| reg_alpha | 2.0 |
| reg_lambda | 2.0 |
| colsample_bytree | 0.8 |
| subsample | 0.8 |
| threshold | 0.4 |
| cv_folds | 5 |
| n_estimators | 1000 |

---

## 모델 성능 지표

### Train 데이터 성능

| 지표 | 값 |
|------|-----|
| Accuracy | 0.7921 |
| Precision | 0.6831 |
| Recall | 0.5390 |
| F1-Score | 0.6026 |
| ROC-AUC | 0.8260 |

### Validation 데이터 성능

| 지표 | 값 |
|------|-----|
| Accuracy | 0.7105 |
| Precision | 0.6092 |
| Recall | 0.4173 |
| F1-Score | 0.4953 |
| ROC-AUC | 0.7348 |

---

## Top 15 피처 중요도

| 순위 | 피처명 | 중요도 |
|------|--------|--------|
| 1 | growth_volatility_scaled | 11.9396 |
| 2 | growth_acceleration_scaled | 5.7530 |
| 3 | cagr_2y_scaled | 5.4970 |
| 4 | rnd_intensity_scaled | 5.4273 |
| 5 | profitable_years | 5.3119 |
| 6 | revenue_t1_scaled | 4.7348 |
| 7 | capex_intensity_scaled | 4.5393 |
| 8 | patent_growth_rate | 4.3027 |
| 9 | operating_margin_scaled | 4.2250 |
| 10 | growth_recent_scaled | 4.2043 |
| 11 | patent_citation_age_adj_total | 4.2009 |
| 12 | citation_per_patent_recent | 4.1501 |
| 13 | capex_trend_scaled | 3.9641 |
| 14 | new_ipc_ratio | 3.8554 |
| 15 | ipc_diversity | 3.8193 |

---

## 성능 지표 해석

### Accuracy (정확도)
- 전체 예측 중 정확하게 예측한 비율
- 현재 모델: 71.05%

### Precision (정밀도)
- 상위 30%로 예측한 기업 중 실제로 상위 30%인 비율
- 현재 모델: 60.92%

### Recall (재현율)
- 실제 상위 30% 기업 중 모델이 올바르게 예측한 비율
- 현재 모델: 41.73%

### F1-Score
- Precision과 Recall의 조화평균
- 현재 모델: 0.4953

### ROC-AUC
- ROC 곡선 아래 면적 (0~1, 높을수록 좋음)
- 현재 모델: 0.7348

