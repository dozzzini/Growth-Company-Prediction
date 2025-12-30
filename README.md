## 🌿 Branch Strategy

본 프로젝트는 **데이터 수집–모델링–GenAI–대시보드 개발이 병렬적으로 진행되는 분석·서비스 결합형 프로젝트**로,  
기능 단위 분리와 안정적인 병합을 위해 **Git Flow를 단순화한 브랜치 전략**을 사용합니다.

---

### 1. 브랜치 구조 개요


| 브랜치 | 역할 |
|---|---|
| `main` | 최종 결과물(발표·제출 기준), 항상 안정 상태 유지 |
| `dev` | 통합 개발 브랜치, 모든 기능 병합의 기준 |
| `feature/*` | 기능·모듈 단위 작업 브랜치 |
| `hotfix/*` | dev 또는 main 기준 긴급 수정 |

---

### 2. 브랜치별 역할 상세

#### 🔹 `main`
- 최종 발표·포트폴리오 기준 브랜치
- 직접 커밋 금지
- `dev` 브랜치에서 검증 완료된 결과만 병합

#### 🔹 `dev`
- 모든 기능 개발의 **통합 기준 브랜치**
- feature 브랜치에서 작업 완료 후 PR로 병합
- 최소한의 실행 가능 상태 유지 (EDA / 모델 / 대시보드)

---

#### 🔹 `feature/*`
기능 단위로 명확히 분리하여 병렬 작업을 가능하게 합니다.

| 브랜치 예시 | 작업 범위 |
|---|---|
| `feature/data` | DART·KIPRIS·뉴스 API 수집, 전처리, 롤링 패널 구성 |
| `feature/model` | 피처 엔지니어링, 모델 학습, 검증, 확률 보정 |
| `feature/genai` | SHAP 기반 프롬프트, 가드레일, 심사 의견 생성 |
| `feature/dashboard` | Streamlit UI, 시각화, 사용자 흐름 구현 |

**운영 원칙**
- 하나의 feature 브랜치는 하나의 책임만 가짐
- 기능 완료 후 `dev` 브랜치로 Pull Request 생성
- 병합 완료 후 feature 브랜치는 삭제

---

#### 🔹 `hotfix/*`
- 데이터 누수, 치명적 버그 등 긴급 수정용 브랜치
- 예: `hotfix/data-leakage-fix`
- 수정 후 `dev` → 필요 시 `main`까지 즉시 반영

---

### 3. 작업 흐름 예시

```bash
# dev 기준 feature 브랜치 생성
git switch dev
git pull origin dev
git switch -c feature/model

# 작업 후 커밋
git add .
git commit -m "feat(model): add rolling validation and calibration"

# 원격 브랜치 푸시
git push origin feature/model

# GitHub에서 PR 생성 → dev로 병합
