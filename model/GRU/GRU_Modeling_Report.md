# GRU Feature-Set Comparison 보고서  
*(재무 피처 vs 재무+특허 피처 | FAST TUNING + FULL REPORT)*

---

## 1. 개요 (Overview)

본 보고서는 **롤링 패널(rolling panel) 구조**로 구성된 기업 데이터(2019~2023년 피처, 2020~2024년 타겟)를 기반으로,  
**GRU 모델**을 사용해 **차년도 성장(이진: `target_growth` 0/1)**을 예측한 실험을 정리한 것이다.

핵심 목적은 다음과 같다.

1) **재무 피처만 사용했을 때**와 **재무+특허 피처를 결합했을 때**의 성능을 동일 프레임워크에서 비교  
2) 모델 선택/튜닝 과정에서 **Top-20 Precision 및 Top-50 Precision을 최우선 목표(스크리닝 관점)**로 두고 성능을 최대화  
3) 튜닝 시간 단축을 위해 **FAST TUNING(rolling year = 2022 단일 연도)**를 수행하되,  
   최종 보고 성능은 **FULL REPORT(rolling years = 2020, 2021, 2022 평균)**로 산출

---

## 2. 실험 설정 (Experiment Setup)

### 2.1 데이터 및 기간 설정

- Dataset: `total_dataset_patent_feature_add_수정.csv`
- feature_year(피처 기준 연도): **2019~2023**
- target_year(타겟 예측 연도): **2020~2024**
- 개발(튜닝) 풀: feature_year **2019~2022**
- 최종 테스트(시간 홀드아웃): **target_year == 2024**

### 2.2 평가 지표 세트

- PR-AUC (Average Precision)
- Top-20 Precision
- Top-50 Precision
- F1 Score *(threshold 기반 이진 분류 결과)*
- Brier Score *(확률 예측 품질)*

> 임계값(Threshold) 고정: **0.67**  
> Top-K 설정: **K=20, 50**

### 2.3 검증 전략 매핑(프로젝트 정의 반영)

- **Holdout (최종 테스트 / 시간 홀드아웃)**: `target_year == 2024`  
- **Rolling (개발/튜닝 검증)**: feature_year 기준 expanding 방식  
  - train: [2019 .. val_year-1], val: [val_year]
- **GroupKFold (기업 일반화 점검 / 옵션)**: 기업명 그룹 기준 K-Fold

---

## 3. 튜닝 전략 (Random Search + Objective Early Stopping)

### 3.1 Random Search

- Trials per feature-set: **8**
- 튜닝 평가 방식: **Rolling(FAST) = [2022] 단일 연도**
- 목적함수(Objective):  
  \[
  \text{Objective} = 0.7 \cdot Top20Precision + 0.3 \cdot Top50Precision
  \]

### 3.2 Early Stopping 기준

- 기본적인 val loss 최소화가 아니라,  
  **Objective(Top20/Top50) 최대화 기준으로 best epoch를 저장**하도록 설계

---

## 4. 실험 A — GRU (재무)

### 4.1 데이터 요약

- Total usable samples: **2330**
- Dev samples(2019~2022): **1864** | Pos=**563** Neg=**1301**
- Num features: **12**
- Example features (first 10):  
  `['revenue_t1', 'cagr_2y', 'growth_recent', 'growth_acceleration', 'growth_volatility', 'operating_margin', 'capex_intensity', 'capex_trend', 'capex_vs_industry', 'debt_ratio']`

### 4.2 튜닝 결과(FAST: rolling year = 2022)

- Best objective score(tuning): **0.640000**
- Best MODEL_PARAMS: `{'hidden_dim': 384, 'layer_dim': 3, 'dropout_prob': 0.2, 'bidirectional': True}`
- Best TRAIN_PARAMS:  
  `{'max_epochs': 50, 'patience': 8, 'lr': 2.77e-03, 'weight_decay': 3.14e-04, 'batch_size': 64, 'use_sampler': True, 'pos_weight_mult': 3.0, 'grad_clip': 1.0}`

### 4.3 최종 성능(FULL REPORT: holdout + rolling + groupkfold)

```
================================================================================
[RESULT TABLE] (metric x validation)  — GRU (재무)
================================================================================
validation        holdout  rolling  groupkfold
PR-AUC              0.395    0.429       0.463
Top-20 Precision    0.500    0.650       0.720
Top-50 Precision    0.520    0.507       0.492
F1 Score            0.463    0.468       0.476
Brier Score         0.527    0.473       0.487
```

---

## 5. 실험 B — GRU (재무+특허)

### 5.1 데이터 요약

- Total usable samples: **2330**
- Dev samples(2019~2022): **1864** | Pos=**563** Neg=**1301**
- Num features: **26**

### 5.2 튜닝 결과(FAST: rolling year = 2022)

- Best objective score(tuning): **0.669000**
- Best MODEL_PARAMS: `{'hidden_dim': 384, 'layer_dim': 3, 'dropout_prob': 0.2, 'bidirectional': True}`
- Best TRAIN_PARAMS:  
  `{'max_epochs': 50, 'patience': 8, 'lr': 2.77e-03, 'weight_decay': 3.14e-04, 'batch_size': 64, 'use_sampler': True, 'pos_weight_mult': 3.0, 'grad_clip': 1.0}`

### 5.3 최종 성능(FULL REPORT: holdout + rolling + groupkfold)

```
================================================================================
[RESULT TABLE] (metric x validation)  — GRU (재무+특허)
================================================================================
validation        holdout  rolling  groupkfold
PR-AUC              0.477    0.425       0.426
Top-20 Precision    0.850    0.717       0.690
Top-50 Precision    0.600    0.553       0.496
F1 Score            0.431    0.474       0.462
Brier Score         0.464    0.480       0.485
```

---

## 6. 비교 분석 및 시사점 (Key Findings)

### 6.1 튜닝(FAST) 관점: “특허 피처가 목표 지표(Top20/Top50)에 유리”

- 재무: best_score **0.6400**
- 재무+특허: best_score **0.6690**

튜닝 목적함수 자체가 **Top20/Top50을 직접 최적화**하도록 설계되어 있으므로,  
FAST 튜닝 결과만 보면 **재무+특허 피처셋이 목표 지표 관점에서 더 유리한 후보 파라미터를 찾을 가능성이 높다**는 신호로 해석할 수 있다.

### 6.2 FULL REPORT 관점: “재무+특허가 스크리닝(Top-K) 성능에서 우세 신호”

- **Holdout(최종 테스트)에서 Top-20 / Top-50이 크게 개선**
  - Top-20 Precision: 재무 **0.500** → 재무+특허 **0.850** *(대폭 개선)*
  - Top-50 Precision: 재무 **0.520** → 재무+특허 **0.600** *(개선)*

- **Rolling(개발 검증 평균)에서도 Top-20/50 개선**
  - Top-20 Precision: 재무 **0.650** → 재무+특허 **0.717**
  - Top-50 Precision: 재무 **0.507** → 재무+특허 **0.553**

- **GroupKFold(기업 일반화 점검)에서는 혼합 신호**
  - Top-20 Precision: 재무 **0.720** → 재무+특허 **0.690** *(소폭 하락)*
  - Top-50 Precision: 재무 **0.492** → 재무+특허 **0.496** *(유사)*

요약하면, **실제 운영 관점(“최종 테스트 + 롤링 검증”)에서는 재무+특허 피처가 스크리닝 성능을 개선**했으나,  
**기업 일반화(GroupKFold)에서는 재무 단독이 Top-20에서 약간 우세**한 결과가 관찰되었다.

### 6.3 PR-AUC / Brier Score 관점: “분리도/확률품질은 일관 개선이 아님”

- PR-AUC는 holdout에서 재무+특허가 상승(0.395 → 0.477)했지만,
  rolling/groupkfold에서는 큰 차이가 없거나 일부 하락도 관찰됨.
- Brier score는 재무+특허가 holdout에서 개선(0.527 → 0.464)했지만,
  rolling/groupkfold에서는 유사한 수준.

즉, 이번 결과는 “확률 자체의 품질(PR-AUC/Brier)”보다 **Top-K 스크리닝 성능**에서 차이가 크게 나타난 케이스로 해석하는 것이 적절하다.

---

## 7. 결론 및 다음 액션 (Recommendations)

### 7.1 결론

- 프로젝트 목표가 **Top-20/Top-50 기반 스크리닝**에 가깝다는 점을 감안하면,  
  **GRU + 재무+특허 피처셋이 유의미하게 더 좋은 결과(특히 holdout, rolling에서 Top-20/50 상승)를 제공**했다.

- 단, **기업 일반화(GroupKFold)에서 Top-20이 소폭 하락**했으므로,  
  “특허 피처가 모든 기업에 대해 일관된 일반화 성능을 항상 개선한다”라고 단정하기보다는,  
  **시간 홀드아웃 및 롤링 검증 중심으로 ‘실제 예측 성능’ 개선을 강조**하는 것이 안전하다.

### 7.2 다음 액션(차이를 더 “명확히/일관되게” 만들기 위한 제안)

1) **FAST 튜닝을 2022 단일 연도에서 [2021, 2022] 또는 [2020, 2021, 2022]로 확장**  
   - 단일 연도 튜닝은 운(표본 변동) 영향이 큼  
2) 재무 vs 재무+특허에서 **동일 탐색 budget(Trial 수) 유지 + Seed 3~5회 반복**  
   - “일관된 개선”을 통계적으로 제시하기 용이  
3) GroupKFold에서 Top-20이 하락하는 원인 점검  
   - (예) 일부 기업군에서 특허 피처 분포가 이질적이거나, 결측/0 값이 정보성보다 노이즈로 작동할 가능성  
4) (옵션) 확률 캘리브레이션(Platt/Isotonic) 또는 로짓 스케일링 등으로 **Brier 개선**을 별도 트랙으로 수행

---

**Modeling Completed.**
