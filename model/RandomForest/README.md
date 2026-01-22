# RandomForest 모델

## 개요

Random Forest 알고리즘을 사용하여 기업의 성장 가능성을 예측하는 모델입니다. 다수의 결정 트리를 앙상블하여 과적합을 방지하고 안정적인 예측을 제공합니다.

## 파일 구조

```
RandomForest/
├── RandomForest.py                    # 메인 모델 학습 스크립트
├── best_params.txt                    # 최적 하이퍼파라미터
├── randomforest_model.pkl             # 학습된 모델 (실행 후 생성)
├── feature_importance.png             # 피처 중요도 그래프
├── predictions_train.png              # Train 데이터 예측 결과
├── predictions_validation.png         # Validation 데이터 예측 결과
├── test_predictions_2024.csv          # 2024년 예측 결과
├── evaluation_results_table.csv       # 평가 결과 종합표
└── RandomForest_PERFORMANCE_REPORT.md # 성능 보고서
```

## 실행 방법

### 1. 기본 실행 (기존 하이퍼파라미터 사용)

```bash
cd /Users/roychoi/Documents/Github/sesac_project/Growth-Company-Prediction/model/RandomForest
python RandomForest.py
```

### 2. 하이퍼파라미터 튜닝 실행

`RandomForest.py` 파일에서 `use_tuning` 변수를 `True`로 변경 후 실행:

```python
# 5. 모델 학습 (MLflow 실험 시작)
use_tuning = True  # False → True로 변경
```

## 모델 특징

### Random Forest의 장점

1. **높은 정확도**: 다수의 트리를 앙상블하여 단일 트리보다 높은 성능
2. **과적합 방지**: Bootstrap aggregating과 feature randomness로 과적합 완화
3. **피처 중요도**: 각 피처의 중요도를 직관적으로 파악 가능
4. **병렬 처리**: 각 트리를 독립적으로 학습하여 빠른 학습 가능
5. **결측치 처리**: 결측치에 강건한 성능

### 하이퍼파라미터

| 파라미터            | 설명                                   | 기본값               |
| ------------------- | -------------------------------------- | -------------------- |
| `n_estimators`      | 트리의 개수                            | 500                  |
| `max_depth`         | 트리의 최대 깊이                       | 20                   |
| `min_samples_split` | 내부 노드를 분할하기 위한 최소 샘플 수 | 5                    |
| `min_samples_leaf`  | 리프 노드의 최소 샘플 수               | 2                    |
| `max_features`      | 분할 시 고려할 최대 피처 수            | 'sqrt'               |
| `bootstrap`         | Bootstrap 샘플링 사용 여부             | True                 |
| `max_samples`       | Bootstrap 샘플링 비율                  | 0.8                  |
| `class_weight`      | 클래스 가중치                          | 'balanced_subsample' |

## 데이터 처리

### 입력 데이터
- **Train 데이터**: `data/train_dataset.csv` (2019-2023년 피처)
- **Test 데이터**: `data/test_dataset.csv` (2023년 피처 → 2024년 타겟 예측)

### 피처
- 재무 피처: 매출액, 영업이익, 자산 등의 정규화 및 성장률
- 특허 피처 (선택적): 특허 출원 건수, 인용 수, IPC 다양성 등

### 타겟 변수
- `target_growth`: 다음 연도 매출액 성장률이 상위 30%에 속하는지 여부 (이진 분류)
  - 1: 상위 30% (성장 기업)
  - 0: 하위 70%

## 평가 지표

모델은 다음 3가지 검증 방법으로 평가됩니다:

### 1. GroupKFold 검증 (기업 일반화)
- 기업 단위로 5-fold 교차 검증
- 미관측 기업에 대한 일반화 성능 평가

### 2. Rolling 검증 (시간 일반화)
- 2021→2022→2023→2024 순서로 평가
- 시계열 데이터에 대한 강건성 평가

### 3. Holdout 검증 (최종 테스트)
- 2023년 피처로 2024년 타겟 예측
- 실제 배포 환경 시뮬레이션

### 평가 지표
- **PR-AUC**: Precision-Recall 곡선 아래 면적 (불균형 데이터에 적합)
- **Top-20/50 Precision**: 상위 K개 예측의 정밀도 (실무 활용도)
- **F1 Score**: Precision과 Recall의 조화평균
- **Brier Score**: 확률 예측의 정확도 (낮을수록 좋음)
- **ROC-AUC**: ROC 곡선 아래 면적

## LightGBM/XGBoost와의 비교

### RandomForest vs LightGBM/XGBoost

| 특징           | RandomForest        | LightGBM/XGBoost      |
| -------------- | ------------------- | --------------------- |
| 학습 방식      | Bagging (병렬)      | Boosting (순차)       |
| 속도           | 빠름 (병렬 처리)    | 중간~느림 (순차 학습) |
| 과적합         | 적음                | 주의 필요             |
| 하이퍼파라미터 | 튜닝 용이           | 튜닝 복잡             |
| 피처 중요도    | 직관적              | 상세 분석 가능        |
| 추천 상황      | 안정적 성능 필요 시 | 최고 성능 필요 시     |

## MLflow 실험 추적

모든 학습 과정은 MLflow로 자동 기록됩니다:
- 하이퍼파라미터
- 평가 지표 (train/validation)
- 모델 아티팩트

### MLflow UI 실행

```bash
mlflow ui
# 브라우저에서 http://localhost:5000 접속
```

## 결과 파일

### 1. 모델 파일
- `randomforest_model.pkl`: 학습된 RandomForest 모델 (joblib 형식)

### 2. 평가 결과
- `evaluation_results_table.csv`: 모든 검증 방법의 평가 지표
- `RandomForest_PERFORMANCE_REPORT.md`: 상세 성능 보고서

### 3. 예측 결과
- `test_predictions_2024.csv`: 2024년 예측 결과
  - 컬럼: 기업명, 연도, predicted_top30, predicted_probability, actual_top30

### 4. 시각화
- `feature_importance.png`: 상위 15개 피처 중요도
- `predictions_train.png`: Train 데이터 예측 결과 (Confusion Matrix, ROC Curve 등)
- `predictions_validation.png`: Validation 데이터 예측 결과

## 의존성 패키지

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
mlflow>=1.20.0
tqdm>=4.62.0
joblib>=1.0.0
```

## 문제 해결

### 메모리 부족 오류
- `n_estimators` 값을 줄이기 (500 → 300)
- `n_jobs` 값을 줄이기 (-1 → 4)

### 학습 속도가 느린 경우
- `max_depth`를 줄이기 (20 → 15)
- `n_estimators`를 줄이기 (500 → 300)

### 과적합 문제
- `min_samples_leaf`를 증가 (2 → 4)
- `max_samples`를 감소 (0.8 → 0.7)
- `class_weight`를 'balanced'로 변경

## 참고 자료

- [scikit-learn RandomForest 문서](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [Random Forest 원리](https://en.wikipedia.org/wiki/Random_forest)
- [앙상블 학습 가이드](https://scikit-learn.org/stable/modules/ensemble.html)
