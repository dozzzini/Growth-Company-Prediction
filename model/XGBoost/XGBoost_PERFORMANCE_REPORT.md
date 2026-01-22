# XGBoost 모델 성능 보고서

**생성 일시**: 2026년 01월 21일 18:14:56

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
| threshold | 0.3 |
| cv_folds | 5 |
| n_estimators | 1000 |

---

## 모델 성능 지표

### Train 데이터 성능

| 지표 | 값 |
|------|-----|
| Accuracy | 0.6910 |
| Precision | 0.4906 |
| Recall | 0.7446 |
| F1-Score | 0.5915 |
| ROC-AUC | 0.7943 |

### Validation 데이터 성능

| 지표 | 값 |
|------|-----|
| Accuracy | 0.6245 |
| Precision | 0.4279 |
| Recall | 0.6389 |
| F1-Score | 0.5125 |
| ROC-AUC | 0.6738 |

---

## Top 15 피처 중요도

| 순위 | 피처명 | 중요도 |
|------|--------|--------|
| 1 | growth_volatility_scaled | 14.7417 |
| 2 | cagr_2y_scaled | 8.3714 |
| 3 | rnd_intensity_scaled | 7.3081 |
| 4 | operating_margin_scaled | 7.2620 |
| 5 | profitable_years | 6.8735 |
| 6 | growth_acceleration_scaled | 6.4229 |
| 7 | patent_recent_ratio | 5.4670 |
| 8 | growth_recent_scaled | 5.4508 |
| 9 | revenue_t1_scaled | 5.3126 |
| 10 | patent_citation_age_adj_avg | 5.1198 |
| 11 | capex_trend_scaled | 4.8698 |
| 12 | patent_citation_age_adj_total | 4.8018 |
| 13 | patent_count_recent | 4.7787 |
| 14 | citation_per_patent_recent | 4.5607 |
| 15 | capex_intensity_scaled | 4.5043 |

---

## 성능 지표 해석

### Accuracy (정확도)
- 전체 예측 중 정확하게 예측한 비율
- 현재 모델: 62.45%

### Precision (정밀도)
- 상위 30%로 예측한 기업 중 실제로 상위 30%인 비율
- 현재 모델: 42.79%

### Recall (재현율)
- 실제 상위 30% 기업 중 모델이 올바르게 예측한 비율
- 현재 모델: 63.89%

### F1-Score
- Precision과 Recall의 조화평균
- 현재 모델: 0.5125

### ROC-AUC
- ROC 곡선 아래 면적 (0~1, 높을수록 좋음)
- 현재 모델: 0.6738

