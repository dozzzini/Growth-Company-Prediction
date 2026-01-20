# LSTM Feature-Set Comparison 보고서  
*(재무 피처 vs 재무+특허 피처 | Enhanced TUNING + FULL REPORT)*

---

## 1. 개요 (Overview)

본 보고서는 **롤링 패널(rolling panel) 구조**로 구성된 기업 데이터(2019~2023년 피처, 2020~2024년 타겟)를 기반으로,  
**LSTM 모델**을 사용해 **차년도 성장(이진: `target_growth` 0/1)**을 예측한 실험을 정리한 것이다.

본 실험의 핵심 목적은 다음과 같다.

1) **재무 피처만 사용했을 때**와 **재무+특허 피처를 결합했을 때**의 성능을 동일 프레임워크에서 비교  
2) 스크리닝(선별) 목적에 맞춰 **Top-20 / Top-50 Precision 중심**으로 성능을 최적화하면서, **PR-AUC·F1(best threshold)·Brier**까지 함께 고려한 **다중지표 튜닝(Enhanced Tuning)** 적용  
3) 튜닝 시간 단축을 위해 **FAST TUNING(rolling year = 2022 단일 연도)**를 수행하되,  
   최종 보고 성능은 **FULL REPORT(rolling years = 2020, 2021, 2022 평균)**로 산출  
4) 특허 피처의 분포 특성을 반영하기 위해 **특허 변환(patent transform)** 및 **확률 보정(temperature scaling)**을 적용하여, 지표 개선뿐 아니라 확률 품질(Brier)까지 함께 점검

---

## 2. 실험 설정 (Experiment Setup)

### 2.1 데이터 및 기간 설정

- Dataset: `total_dataset_patent_feature_add_수정.csv`
- feature_year(피처 기준 연도): **2019~2023**
- target_year(타겟 예측 연도): **2020~2024**
- 개발(튜닝) 풀: feature_year **2019~2022**
- 최종 테스트(시간 홀드아웃): **target_year == 2024**
- Total usable samples: **2330**
- Dev samples(2019~2022): **1864** | Pos=**563** Neg=**1301**

### 2.2 피처셋 구성

- **재무 피처셋**
  - Num features(used): **12**
  - Example features(12 used):  
    `['revenue_t1', 'cagr_2y', 'growth_recent', 'growth_acceleration', 'growth_volatility', 'operating_margin', 'capex_intensity', 'capex_trend', 'capex_vs_industry', 'debt_ratio', 'rnd_intensity', 'profitable_years']`

- **재무+특허 피처셋**
  - Num features(used): **26**
  - (공통 재무 피처 포함 + 특허 관련 피처 추가)

### 2.3 평가 지표 세트

- PR-AUC (Average Precision)
- Top-20 Precision *(K=20)*
- Top-50 Precision *(K=50)*
- **F1@best(th)** *(각 평가에서 F1이 최대가 되는 임계값을 탐색하여 산출)*
- Brier Score *(확률 예측 품질)*

> 본 실험은 **임계값 고정이 아니라, F1을 위해 best threshold를 탐색**하는 설정이며,  
> 스크리닝 관점에서는 Top-K 성능을 우선하되, **PR-AUC·F1(best)·Brier까지 동시에 균형**을 보는 전략을 채택했다.

### 2.4 검증 전략 매핑

- **Holdout (최종 테스트 / 시간 홀드아웃)**: `target_year == 2024`
- **Rolling (개발/리포트 검증)**: feature_year 기준 expanding 방식  
  - train: [2019 .. val_year-1], val: [val_year]
- **GroupKFold (기업 일반화 점검 / 옵션)**: 기업명 그룹 기준 K-Fold

---

## 3. Enhanced 튜닝 전략 (Random Search + Multi-metric Objective)

### 3.1 Random Search (FAST)

- Trials per feature-set: **12**
- FAST 튜닝 평가 방식: **Rolling(FAST) = [2022] 단일 연도**
- 튜닝 목적: 단일 지표가 아니라 **스크리닝 성능 + 분류 품질 + 확률 품질**을 함께 반영하는 조합 점수 최대화

### 3.2 목적함수(Objective) 구성

튜닝 로그에서 사용된 objective는 다음 요소를 포함한다.

- **Top20 / Top50 Precision**
- **PR-AUC**
- **F1@best(th)**
- **Brier Score (감점 항목)**

즉, “Top-K만 높고 확률이 무너지는 모델” 또는 “PR-AUC만 높고 Top-K가 약한 모델”을 배제하고,  
**다중 목적에 대해 실무적으로 균형 잡힌 파라미터를 선택**하도록 설계했다.

---

## 4. 공통 전처리 및 보정 (Enhancements)

### 4.1 Patent Transform

특허 피처의 분포 왜곡(극단값, 긴 꼬리)을 완화하기 위해 다음 변환을 적용했다.

- `clip_q = 0.99` : 상위 1% 극단값 클리핑
- `log1p = True` : 로그 변환으로 스케일 안정화
- `add_has_patent = True` : 특허 보유 여부 신호(feature) 추가

### 4.2 Temperature Scaling (확률 보정)

- `Temperature scaling = True`
- 분류 성능뿐 아니라 **확률 예측의 신뢰도(Calibration)** 관점에서 Brier Score와 함께 관리하는 목적

---

## 5. 실험 A — LSTM (재무)

### 5.1 튜닝 요약 (FAST: rolling year = 2022)

- Trials: **12**
- Best score(tuning): **0.583043**
- Best MODEL_PARAMS: `{'hidden_dim': 64, 'layer_dim': 3, 'dropout_prob': 0.2, 'bidirectional': False}`
- Best TRAIN_PARAMS:  
  `{'max_epochs': 90, 'patience': 14, 'lr': '4.69e-04', 'weight_decay': '2.39e-06', 'batch_size': 64, 'use_sampler': True, 'pos_weight_mult': 2.0, 'grad_clip': 2.0}`

> 본 실험에서는 best params가 로컬 경로에 저장되었으며, 튜닝 결과를 최종 평가(holdout/rolling/groupkfold)에 그대로 사용했다.

### 5.2 최종 성능 (FULL REPORT)

```
==========================================================================================
[RESULT TABLE] (metric x validation)  — LSTM (재무)  [Enhanced: multi-metric tuning]
==========================================================================================
validation        holdout  rolling  groupkfold
PR-AUC              0.374    0.449       0.444
Top-20 Precision    0.550    0.717       0.660
Top-50 Precision    0.440    0.520       0.500
F1@best(th)         0.465    0.490       0.497
Brier Score         0.252    0.274       0.277
```

---

## 6. 실험 B — LSTM (재무+특허)

### 6.1 튜닝 요약 (FAST: rolling year = 2022)

- Trials: **12**
- Best score(tuning): **0.608627**
- Best MODEL_PARAMS: `{'hidden_dim': 384, 'layer_dim': 3, 'dropout_prob': 0.3, 'bidirectional': True}`
- Best TRAIN_PARAMS:  
  `{'max_epochs': 90, 'patience': 14, 'lr': '4.44e-03', 'weight_decay': '8.16e-04', 'batch_size': 64, 'use_sampler': False, 'pos_weight_mult': 2.5, 'grad_clip': 1.0}`

### 6.2 최종 성능 (FULL REPORT)

```
==========================================================================================
[RESULT TABLE] (metric x validation)  — LSTM (재무+특허)  [Enhanced: patent transform + temperature scaling]
==========================================================================================
validation        holdout  rolling  groupkfold
PR-AUC              0.414    0.451       0.462
Top-20 Precision    0.650    0.717       0.730
Top-50 Precision    0.480    0.567       0.528
F1@best(th)         0.474    0.481       0.486
Brier Score         0.280    0.257       0.265
```

---

## 7. 비교 분석 및 시사점 (Key Findings)

### 7.1 튜닝(FAST) 관점: 재무+특허가 더 높은 탐색 성과

- 재무: best_score **0.5830**
- 재무+특허: best_score **0.6086**

다중지표 목적함수(Top-K + PR-AUC + F1(best) − Brier)를 기준으로 했을 때,  
**재무+특허 피처셋이 더 높은 점수의 파라미터 조합을 탐색**했다.  
이는 특허 피처가 스크리닝 지표 및 전반적 분별력(PR-AUC) 측면에서 추가 신호를 제공할 가능성을 시사한다.

### 7.2 FULL REPORT 관점: “Top-K 및 PR-AUC에서 일관된 개선”

- **PR-AUC**: holdout **0.374 → 0.414**, rolling **0.449 → 0.451**, groupkfold **0.444 → 0.462**
- **Top-20 Precision**: holdout **0.550 → 0.650**, rolling **0.717 → 0.717(동일)**, groupkfold **0.660 → 0.730**
- **Top-50 Precision**: holdout **0.440 → 0.480**, rolling **0.520 → 0.567**, groupkfold **0.500 → 0.528**

즉, 재무+특허 피처셋은 특히 **holdout(2024)과 groupkfold(기업 일반화)에서 Top-20이 의미 있게 개선**되었고,  
Top-50 역시 전반적으로 개선 방향을 보였다.  
스크리닝 관점에서 “상위 후보를 뽑는” 목적에 보다 부합하는 결과로 해석 가능하다.

### 7.3 F1@best(th) 관점: 큰 차이는 없으나, 전반적 균형 확인

- holdout: **0.465 → 0.474** (소폭 개선)
- rolling: **0.490 → 0.481** (소폭 하락)
- groupkfold: **0.497 → 0.486** (소폭 하락)

본 실험에서는 F1을 고정 threshold가 아니라 best threshold로 평가했기 때문에,  
F1은 “운영 임계값 선택의 여지”를 반영한 보조 품질 지표로 해석하는 것이 타당하다.  
결과적으로 **Top-K에서의 개선이 더 분명하며**, F1(best)은 큰 폭의 악화 없이 유지되는 양상이다.

### 7.4 확률 품질(Brier) 관점: 보정 효과/트레이드오프 점검 필요

- holdout Brier: **0.252(재무) vs 0.280(재무+특허)** → 재무+특허가 불리
- rolling / groupkfold: **0.274→0.257**, **0.277→0.265** → 재무+특허가 유리

즉, 재무+특허는 **개발 구간(rolling)과 기업 일반화(groupkfold)**에서는 확률 품질이 개선되지만,  
최종 테스트(holdout)에서는 Brier가 상승한다.  
이는 (1) 2024년 분포 변화, (2) temperature scaling 적용 방식/학습 구간 차이, (3) 특허 피처의 연도별 신호 안정성 등의 요인을 추가 점검할 필요가 있음을 의미한다.

---

## 8. 결론 및 다음 액션 (Recommendations)

### 8.1 결론

- **스크리닝 성능(Top-20/50) 중심**으로 보면,  
  재무+특허 피처셋이 holdout 및 groupkfold에서 **Top-20을 개선**하고, Top-50도 전반적으로 개선되어  
  **실무적 후보 선별 목적에 더 유리할 가능성이 높다.**
- PR-AUC도 재무+특허가 전반적으로 우세하여, 분별력 측면의 보강 신호가 확인된다.
- 다만 확률 품질(Brier)은 holdout에서 악화되어, **최종 연도에 대한 calibration 안정성**은 추가 확인이 필요하다.

### 8.2 다음 액션 (명확한 차이/안정성 확보)

1) **튜닝 rolling year를 [2020, 2021, 2022]로 확장**해 “FAST 단일 연도 의존”을 완화  
2) 동일 budget(Trials=12)을 유지하되 **Seed 3~5회 반복 실험**으로 결과 일관성 확보  
3) temperature scaling의 학습 구간(예: dev 내부 split, rolling 기준)과 적용 구간을 명확히 분리해,  
   **holdout에서 Brier 악화 원인을 진단**  
4) 특허 transform 조합(clip_q, log1p, has_patent)의 ablation을 수행하여,  
   **Top-K 개선의 기여 요인(어떤 변환이 유효했는지)**을 분해

---

**Modeling Completed.**
