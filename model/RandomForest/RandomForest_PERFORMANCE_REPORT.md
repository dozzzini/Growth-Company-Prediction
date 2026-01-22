# RandomForest 모델 성능 보고서

**생성 일시**: 2026년 01월 21일 17:26:59

---

## 하이퍼파라미터

| 파라미터 | 값 |
|---------|-----|
| max_depth | 6 |
| min_samples_split | 5 |
| min_samples_leaf | 5 |
| max_features | sqrt |
| bootstrap | True |
| max_samples | 0.8 |
| class_weight | balanced |
| threshold | 0.4 |
| cv_folds | 5 |

---

## 모델 성능 지표

### Train 데이터 성능

| 지표 | 값 |
|------|-----|
| Accuracy | 0.9799 |
| Precision | 0.9413 |
| Recall | 0.9931 |
| F1-Score | 0.9665 |
| ROC-AUC | 0.9992 |

### Validation 데이터 성능

| 지표 | 값 |
|------|-----|
| Accuracy | 0.6836 |
| Precision | 0.5360 |
| Recall | 0.5276 |
| F1-Score | 0.5317 |
| ROC-AUC | 0.7040 |

---

## Top 15 피처 중요도

| 순위 | 피처명 | 중요도 |
|------|--------|--------|
| 1 | growth_volatility_scaled | 0.0750 |
| 2 | growth_acceleration_scaled | 0.0418 |
| 3 | growth_recent_scaled | 0.0409 |
| 4 | rnd_intensity_scaled | 0.0392 |
| 5 | cagr_2y_scaled | 0.0391 |
| 6 | capex_intensity_scaled | 0.0355 |
| 7 | operating_margin_scaled | 0.0336 |
| 8 | revenue_t1_scaled | 0.0315 |
| 9 | debt_ratio_scaled | 0.0314 |
| 10 | capex_trend_scaled | 0.0270 |
| 11 | capex_vs_industry_scaled | 0.0255 |
| 12 | patent_emb_8 | 0.0158 |
| 13 | patent_emb_39 | 0.0139 |
| 14 | patent_emb_3 | 0.0126 |
| 15 | patent_emb_22 | 0.0125 |

---

## 성능 지표 해석

### Accuracy (정확도)
- 전체 예측 중 정확하게 예측한 비율
- 현재 모델: 68.36%

### Precision (정밀도)
- 상위 30%로 예측한 기업 중 실제로 상위 30%인 비율
- 현재 모델: 53.60%

### Recall (재현율)
- 실제 상위 30% 기업 중 모델이 올바르게 예측한 비율
- 현재 모델: 52.76%

### F1-Score
- Precision과 Recall의 조화평균
- 현재 모델: 0.5317

### ROC-AUC
- ROC 곡선 아래 면적 (0~1, 높을수록 좋음)
- 현재 모델: 0.7040

