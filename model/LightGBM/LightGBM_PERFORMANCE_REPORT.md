# LightGBM 모델 성능 보고서

**생성 일시**: 2026년 01월 17일 17:27:19

---

## 하이퍼파라미터

| 파라미터 | 값 |
|---------|-----|
| objective | binary |
| metric | binary_logloss |
| boosting_type | gbdt |
| verbose | -1 |
| random_state | 42 |
| max_depth | 5 |
| min_child_samples | 50 |
| num_leaves | 15 |
| learning_rate | 0.01 |
| reg_alpha | 5.0 |
| reg_lambda | 2.0 |
| bagging_fraction | 0.7 |
| bagging_freq | 5 |
| feature_fraction | 0.6 |
| threshold | 0.4 |
| cv_folds | 5 |
| n_estimators | 1000 |

---

## 모델 성능 지표

### Train 데이터 성능

| 지표 | 값 |
|------|-----|
| Accuracy | 0.9209 |
| Precision | 0.8995 |
| Recall | 0.8211 |
| F1-Score | 0.8585 |
| ROC-AUC | 0.9704 |

### Validation 데이터 성능

| 지표 | 값 |
|------|-----|
| Accuracy | 0.7078 |
| Precision | 0.6047 |
| Recall | 0.4094 |
| F1-Score | 0.4883 |
| ROC-AUC | 0.7134 |

---

## Top 15 피처 중요도

| 순위 | 피처명 | 중요도 |
|------|--------|--------|
| 1 | growth_volatility_scaled | 908.0708 |
| 2 | rnd_intensity_scaled | 424.9684 |
| 3 | debt_ratio_scaled | 382.7915 |
| 4 | operating_margin_scaled | 340.3199 |
| 5 | cagr_2y_scaled | 319.0111 |
| 6 | growth_recent_scaled | 317.3631 |
| 7 | capex_intensity_scaled | 296.3399 |
| 8 | growth_acceleration_scaled | 263.1882 |
| 9 | revenue_t1_scaled | 199.6871 |
| 10 | patent_emb_3 | 178.7737 |
| 11 | capex_trend_scaled | 165.8837 |
| 12 | patent_emb_46 | 134.8342 |
| 13 | patent_emb_8 | 123.0425 |
| 14 | capex_vs_industry_scaled | 114.7387 |
| 15 | patent_emb_39 | 112.7033 |

---

## 성능 지표 해석

### Accuracy (정확도)
- 전체 예측 중 정확하게 예측한 비율
- 현재 모델: 70.78%

### Precision (정밀도)
- 상위 30%로 예측한 기업 중 실제로 상위 30%인 비율
- 현재 모델: 60.47%

### Recall (재현율)
- 실제 상위 30% 기업 중 모델이 올바르게 예측한 비율
- 현재 모델: 40.94%

### F1-Score
- Precision과 Recall의 조화평균
- 현재 모델: 0.4883

### ROC-AUC
- ROC 곡선 아래 면적 (0~1, 높을수록 좋음)
- 현재 모델: 0.7134

