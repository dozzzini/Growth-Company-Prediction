import os
import random
import warnings
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, average_precision_score

warnings.filterwarnings("ignore")


# =========================================================
# 0) 설정
# =========================================================
CONFIG = {
    # 단일 데이터셋 경로
    "DATA_PATH": os.path.join("data", "total_dataset_patent_feature_add_수정.csv"),

    # 인코딩 자동 시도 (UTF-8 실패 시 cp949)
    "ENCODING_CANDIDATES": ["utf-8-sig", "cp949"],

    # 컬럼명
    "COL_COMPANY": "기업명",
    "COL_YEAR": "연도",                 # feature_year (t)
    "COL_TARGET_YEAR": "target_year",   # t+1
    "COL_TARGET": "target_growth",      # 0/1

    # ===========================
    # 평가 설계 매핑
    # ===========================
    # 최종 테스트(시간 홀드아웃): target_year=2024 고정
    "HOLDOUT_TARGET_YEAR": 2024,

    # Rolling/GroupKFold는 개발/튜닝용(시간누수 방지 위해 최종 테스트 연도 제외)
    # feature_year 기준으로 2019~2022를 dev 풀로 사용
    "DEV_YEARS_MIN": 2019,
    "DEV_YEARS_MAX": 2022,

    # 최종 리포트용 rolling(expanding): train [2019..y-1], val [y]
    "ROLLING_VAL_YEARS": [2020, 2021, 2022],

    # ===========================
    # 공통 평가 설정
    # ===========================
    "THRESHOLD": 0.67,  # F1 계산용 임계값(고정)

    # Top-K Precision을 Top-20/Top-50으로 분리
    "TOP20_K": 20,
    "TOP50_K": 50,

    "SEED": 42,

    # GroupKFold 설정(기업 일반화 점검)
    "GROUPKFOLD_SPLITS": 5,

    # ===========================
    # Random Search 튜닝 설정
    # ===========================
    "USE_RANDOM_SEARCH": True,

    # (속도 개선) 피처셋별 trial 수를 분배
    "N_TRIALS_PER_FEATURESET": 8,  # 예: 재무 8 + 재무+특허 8 (총 16 trials)

    # objective = 0.7*Top20 + 0.3*Top50
    "OBJECTIVE_WEIGHTS": {
        "top20_precision": 0.7,
        "top50_precision": 0.3
    },

    "SAVE_BEST_PARAMS": True,
    "BEST_PARAMS_FILENAME": "best_params.txt",

    # ===========================
    # Early Stopping 기준
    # ===========================
    # True: val loss가 아니라 objective(Top20/Top50) 최대화로 best epoch 저장
    "EARLYSTOP_ON_OBJECTIVE": True,

    # EPS_IMPROVE는 너무 작으면 노이즈에 민감해짐 → 현실적인 값으로 상향
    "EPS_IMPROVE": 1e-4,

    # ===========================
    # (속도 개선) 튜닝 단계 가속 설정
    # ===========================
    # 튜닝은 rolling 1개 연도만으로 후보를 빠르게 찾고,
    # 최종표는 ROLLING_VAL_YEARS(3개)로 계산.
    "TUNING_ROLLING_VAL_YEARS": [2022],

    # 튜닝 단계 epoch/patience 축소
    "TUNING_MAX_EPOCHS": 50,
    "TUNING_PATIENCE": 8,
}


# =========================================================
# (중요) 피처셋 정의
#  - FIN_FEATS_12: "데이터 컬럼명 그대로" 12개를 반드시 입력
#  - PATENT_FEATS_15: 특허 15개 (strip 대응 포함)
# =========================================================
FIN_FEATS_12 = [
    "revenue_t1", "cagr_2y", "growth_recent", "growth_acceleration", "growth_volatility",
    "operating_margin", "capex_intensity", "capex_trend", "capex_vs_industry",
    "debt_ratio", "rnd_intensity", "profitable_years"
]

PATENT_FEATS_15 = [
    "patent_count_5y",
    "patent_count_recent",
    "ipc_diversity",
    "patent_register_rate",
    "patent_citation_total",
    "patent_citation_avg",
    "no_patent_flag",
    "patent_recent_ratio",
    "citation_recent_ratio",
    "patent_growth_rate",
    "patent_growth_accel",
    "new_ipc_ratio",
    "recent_ipc_overlap",
    "citation_per_patent_recent",
]


# =========================================================
# 0-1) 재현성
# =========================================================
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================================================
# 1) GRU 모델  (LSTM -> GRU로만 교체)
# =========================================================
class GRUClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_dim=256, layer_dim=2, dropout_prob=0.2, bidirectional=True):
        super().__init__()
        self.num_dir = 2 if bidirectional else 1

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=layer_dim,
            batch_first=True,
            dropout=dropout_prob if layer_dim > 1 else 0.0,
            bidirectional=bidirectional
        )

        head_in = hidden_dim * self.num_dir
        self.fc = nn.Sequential(
            nn.Linear(head_in, head_in * 2),
            nn.BatchNorm1d(head_in * 2),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            nn.Linear(head_in * 2, head_in),
            nn.BatchNorm1d(head_in),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            nn.Linear(head_in, head_in // 2),
            nn.ReLU(),
            nn.Linear(head_in // 2, 1)  # logits
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, 1, F)
        out, _ = self.gru(x)
        last_out = out[:, -1, :]
        return self.fc(last_out)


# =========================================================
# 2) 데이터 로드/전처리
# =========================================================
def _read_csv_with_fallback(path: str, encodings: List[str]) -> pd.DataFrame:
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
    raise last_err


def load_dataset(project_root: str) -> pd.DataFrame:
    path = os.path.join(project_root, CONFIG["DATA_PATH"])
    df = _read_csv_with_fallback(path, CONFIG["ENCODING_CANDIDATES"])

    # 컬럼명 공백 제거(예: 'patent_recent_ratio ')
    df.columns = df.columns.astype(str).str.strip()

    required = [CONFIG["COL_COMPANY"], CONFIG["COL_YEAR"], CONFIG["COL_TARGET_YEAR"], CONFIG["COL_TARGET"]]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing}")

    # 숫자형으로 강제(방어)
    df[CONFIG["COL_YEAR"]] = pd.to_numeric(df[CONFIG["COL_YEAR"]], errors="coerce")
    df[CONFIG["COL_TARGET_YEAR"]] = pd.to_numeric(df[CONFIG["COL_TARGET_YEAR"]], errors="coerce")
    df[CONFIG["COL_TARGET"]] = pd.to_numeric(df[CONFIG["COL_TARGET"]], errors="coerce")

    return df


def prepare_Xy_with_feature_cols(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.DataFrame, pd.Series]:
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"선택한 feature_cols 중 데이터에 없는 컬럼이 있습니다: {missing}")

    X = df[feature_cols].copy()
    y = df[CONFIG["COL_TARGET"]].copy()

    X = X.apply(pd.to_numeric, errors="coerce")

    mask = ~(X.isnull().any(axis=1) | y.isnull())
    X = X[mask]
    y = y[mask].astype(int)

    uniq = set(y.unique().tolist())
    if not uniq.issubset({0, 1}):
        raise ValueError(f"{CONFIG['COL_TARGET']}는 0/1이어야 합니다. 현재 값: {sorted(list(uniq))}")

    return X, y


# =========================================================
# 3) 텐서/학습/예측/Objective 유틸
# =========================================================
def to_tensor_3d(X_2d: np.ndarray) -> torch.FloatTensor:
    return torch.FloatTensor(X_2d).unsqueeze(1)  # (N, 1, F)


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def topk_precision_np(y_true: np.ndarray, y_proba: np.ndarray, k: int) -> float:
    k = int(min(max(k, 1), len(y_proba)))
    idx = np.argsort(-y_proba)[:k]
    return float(y_true[idx].sum() / k)


def val_objective_from_logits(v_logits: torch.Tensor, y_val_t: torch.Tensor) -> float:
    proba = sigmoid_np(v_logits.detach().cpu().numpy().reshape(-1))
    y_true = y_val_t.detach().cpu().numpy().reshape(-1).astype(int)

    top20 = topk_precision_np(y_true, proba, CONFIG["TOP20_K"])
    top50 = topk_precision_np(y_true, proba, CONFIG["TOP50_K"])

    w20 = CONFIG["OBJECTIVE_WEIGHTS"]["top20_precision"]
    w50 = CONFIG["OBJECTIVE_WEIGHTS"]["top50_precision"]
    return float(w20 * top20 + w50 * top50)


@torch.no_grad()
def predict_proba(model: nn.Module, X_tensor: torch.Tensor, device: torch.device) -> np.ndarray:
    model.eval()
    logits = model(X_tensor.to(device)).detach().cpu().numpy()
    proba = sigmoid_np(logits)
    return proba.reshape(-1)


def build_train_loader(X_tr_t: torch.Tensor, y_tr_t: torch.Tensor, batch_size: int, use_sampler: bool) -> DataLoader:
    dataset = TensorDataset(X_tr_t, y_tr_t)

    if not use_sampler:
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    y_np = y_tr_t.view(-1).cpu().numpy().astype(int)
    class_counts = np.bincount(y_np, minlength=2)
    w0 = 1.0 / max(class_counts[0], 1)
    w1 = 1.0 / max(class_counts[1], 1)
    sample_weights = np.where(y_np == 1, w1, w0)

    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    )
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)


def make_pos_weight(y_train_tensor: torch.Tensor, device: torch.device, mult: float = 1.0) -> torch.Tensor:
    y = y_train_tensor.view(-1)
    n_pos = (y == 1).sum().item()
    n_neg = (y == 0).sum().item()
    if n_pos == 0:
        return torch.tensor([1.0], device=device)
    base = n_neg / max(n_pos, 1)
    return torch.tensor([base * mult], device=device)


# =========================================================
# 4) 학습 (Early stopping: objective 기반) — 함수명만 GRU로 변경
# =========================================================
def train_gru(
    model: nn.Module,
    train_loader: DataLoader,
    X_val_t: torch.Tensor,
    y_val_t: torch.Tensor,
    device: torch.device,
    max_epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    pos_weight: torch.Tensor,
    grad_clip: float,
) -> nn.Module:
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_state = None

    if CONFIG["EARLYSTOP_ON_OBJECTIVE"]:
        best_val_obj = -float("inf")
    else:
        best_val_loss = float("inf")

    no_improve = 0

    for _epoch in range(1, max_epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()

            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

        model.eval()
        with torch.no_grad():
            v_logits = model(X_val_t.to(device))

            if CONFIG["EARLYSTOP_ON_OBJECTIVE"]:
                v_obj = val_objective_from_logits(v_logits, y_val_t.to(device))
                if v_obj > best_val_obj + CONFIG["EPS_IMPROVE"]:
                    best_val_obj = v_obj
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                v_loss = criterion(v_logits, y_val_t.to(device)).item()
                if v_loss < best_val_loss - 1e-6:
                    best_val_loss = v_loss
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1

        if no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# =========================================================
# 5) 지표 계산
# =========================================================
def brier_score(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    return float(np.mean((y_proba - y_true) ** 2))


def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (y_proba >= threshold).astype(int)

    pr_auc = float(average_precision_score(y_true, y_proba))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    top20 = float(topk_precision_np(y_true, y_proba, CONFIG["TOP20_K"]))
    top50 = float(topk_precision_np(y_true, y_proba, CONFIG["TOP50_K"]))
    brier = float(brier_score(y_true, y_proba))

    return {"pr_auc": pr_auc, "top20_precision": top20, "top50_precision": top50, "f1": f1, "brier": brier}


def objective_score(metrics: Dict[str, float]) -> float:
    w = CONFIG["OBJECTIVE_WEIGHTS"]
    return float(w["top20_precision"] * metrics["top20_precision"] + w["top50_precision"] * metrics["top50_precision"])


# =========================================================
# 6) 학습+평가 1회 실행(스케일링 포함)
# =========================================================
def fit_and_eval_once(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    input_size: int,
    device: torch.device,
    model_params: Dict[str, Any],
    train_params: Dict[str, Any],
) -> Dict[str, float]:
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train.astype(np.float32))
    X_va_s = scaler.transform(X_val.astype(np.float32))

    X_tr_t = to_tensor_3d(X_tr_s.astype(np.float32))
    X_va_t = to_tensor_3d(X_va_s.astype(np.float32))
    y_tr_t = torch.FloatTensor(y_train.astype(np.float32)).view(-1, 1)
    y_va_t = torch.FloatTensor(y_val.astype(np.float32)).view(-1, 1)

    # (변경) LSTMClassifier -> GRUClassifier
    model = GRUClassifier(
        input_size=input_size,
        hidden_dim=model_params["hidden_dim"],
        layer_dim=model_params["layer_dim"],
        dropout_prob=model_params["dropout_prob"],
        bidirectional=model_params["bidirectional"],
    ).to(device)

    train_loader = build_train_loader(
        X_tr_t, y_tr_t,
        batch_size=train_params["batch_size"],
        use_sampler=train_params["use_sampler"]
    )
    pos_weight = make_pos_weight(y_tr_t, device=device, mult=train_params["pos_weight_mult"])

    # (변경) train_lstm -> train_gru
    model = train_gru(
        model=model,
        train_loader=train_loader,
        X_val_t=X_va_t,
        y_val_t=y_va_t,
        device=device,
        max_epochs=train_params["max_epochs"],
        patience=train_params["patience"],
        lr=train_params["lr"],
        weight_decay=train_params["weight_decay"],
        pos_weight=pos_weight,
        grad_clip=train_params["grad_clip"],
    )

    y_proba = predict_proba(model, X_va_t, device)
    return compute_metrics(y_true=y_val.astype(int), y_proba=y_proba, threshold=CONFIG["THRESHOLD"])


# =========================================================
# 7) 평가 전략별 평가
# =========================================================
def evaluate_holdout_time(
    df_all: pd.DataFrame, feature_cols: List[str],
    model_params: Dict[str, Any], train_params: Dict[str, Any]
) -> Dict[str, float]:
    holdout_ty = CONFIG["HOLDOUT_TARGET_YEAR"]

    test_df = df_all[df_all[CONFIG["COL_TARGET_YEAR"]] == holdout_ty].copy()
    train_df = df_all[df_all[CONFIG["COL_TARGET_YEAR"]] < holdout_ty].copy()

    if len(train_df) == 0 or len(test_df) == 0:
        raise ValueError(
            f"시간 홀드아웃 분리 실패: train={len(train_df)}, test={len(test_df)}. "
            f"{CONFIG['COL_TARGET_YEAR']} 분포를 확인하세요."
        )

    X_tr = train_df[feature_cols].values
    y_tr = train_df[CONFIG["COL_TARGET"]].values.astype(int)
    X_te = test_df[feature_cols].values
    y_te = test_df[CONFIG["COL_TARGET"]].values.astype(int)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return fit_and_eval_once(
        X_train=X_tr, y_train=y_tr,
        X_val=X_te, y_val=y_te,
        input_size=len(feature_cols),
        device=device,
        model_params=model_params,
        train_params=train_params
    )


def evaluate_rolling_dev(
    df_dev: pd.DataFrame, feature_cols: List[str],
    model_params: Dict[str, Any], train_params: Dict[str, Any],
    val_years: List[int]
) -> Dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    metrics_list: List[Dict[str, float]] = []
    for val_year in val_years:
        train_years = list(range(CONFIG["DEV_YEARS_MIN"], int(val_year)))

        train_df = df_dev[df_dev[CONFIG["COL_YEAR"]].isin(train_years)].copy()
        val_df = df_dev[df_dev[CONFIG["COL_YEAR"]] == val_year].copy()

        if len(train_df) == 0 or len(val_df) == 0:
            continue

        X_tr = train_df[feature_cols].values
        y_tr = train_df[CONFIG["COL_TARGET"]].values.astype(int)
        X_va = val_df[feature_cols].values
        y_va = val_df[CONFIG["COL_TARGET"]].values.astype(int)

        m = fit_and_eval_once(
            X_train=X_tr, y_train=y_tr,
            X_val=X_va, y_val=y_va,
            input_size=len(feature_cols),
            device=device,
            model_params=model_params,
            train_params=train_params
        )
        metrics_list.append(m)

    if len(metrics_list) == 0:
        return {"pr_auc": np.nan, "top20_precision": np.nan, "top50_precision": np.nan, "f1": np.nan, "brier": np.nan}

    return {k: float(np.mean([mm[k] for mm in metrics_list])) for k in metrics_list[0].keys()}


def evaluate_groupkfold_dev(
    df_dev: pd.DataFrame, feature_cols: List[str],
    model_params: Dict[str, Any], train_params: Dict[str, Any]
) -> Dict[str, float]:
    gkf = GroupKFold(n_splits=CONFIG["GROUPKFOLD_SPLITS"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_dev = df_dev[feature_cols].values
    y_dev = df_dev[CONFIG["COL_TARGET"]].values.astype(int)
    groups = df_dev[CONFIG["COL_COMPANY"]].values

    fold_metrics: List[Dict[str, float]] = []
    for tr_idx, va_idx in gkf.split(X_dev, y_dev, groups=groups):
        m = fit_and_eval_once(
            X_train=X_dev[tr_idx],
            y_train=y_dev[tr_idx],
            X_val=X_dev[va_idx],
            y_val=y_dev[va_idx],
            input_size=len(feature_cols),
            device=device,
            model_params=model_params,
            train_params=train_params
        )
        fold_metrics.append(m)

    return {k: float(np.mean([fm[k] for fm in fold_metrics])) for k in fold_metrics[0].keys()}


# =========================================================
# 8) Random Search
# =========================================================
def sample_params(rng: np.random.RandomState, fast_tuning: bool = True) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    model_params = {
        "hidden_dim": int(rng.choice([64, 128, 256, 384])),
        "layer_dim": int(rng.choice([1, 2, 3])),
        "dropout_prob": float(rng.choice([0.1, 0.2, 0.3, 0.4])),
        "bidirectional": bool(rng.choice([True, False])),
    }

    lr = float(10 ** rng.uniform(np.log10(2e-4), np.log10(3e-3)))
    wd = float(10 ** rng.uniform(np.log10(1e-6), np.log10(1e-3)))

    if fast_tuning:
        max_epochs = int(rng.choice([40, 50]))
        patience = int(rng.choice([6, 8]))
    else:
        max_epochs = int(rng.choice([60, 80, 100]))
        patience = int(rng.choice([8, 12, 16]))

    train_params = {
        "max_epochs": max_epochs,
        "patience": patience,
        "lr": lr,
        "weight_decay": wd,
        "batch_size": int(rng.choice([64, 128, 256])),
        "use_sampler": bool(rng.choice([True, False])),
        "pos_weight_mult": float(rng.choice([1.0, 1.5, 2.0, 3.0, 4.0])),
        "grad_clip": float(rng.choice([0.5, 1.0, 2.0])),
    }

    return model_params, train_params


def run_random_search(df_dev: pd.DataFrame, feature_cols: List[str], tag: str) -> Dict[str, Any]:
    rng = np.random.RandomState(CONFIG["SEED"])

    best_score = -1e18
    best_pack: Dict[str, Any] = {}

    tuning_years = CONFIG["TUNING_ROLLING_VAL_YEARS"]

    for t in range(1, CONFIG["N_TRIALS_PER_FEATURESET"] + 1):
        model_params, train_params = sample_params(rng, fast_tuning=True)

        m = evaluate_rolling_dev(
            df_dev=df_dev,
            feature_cols=feature_cols,
            model_params=model_params,
            train_params=train_params,
            val_years=tuning_years,
        )
        score = objective_score(m)

        print(
            f"[RandomSearch-{tag} {t:02d}/{CONFIG['N_TRIALS_PER_FEATURESET']}] "
            f"score={score:.4f} | Top20={m['top20_precision']:.3f} Top50={m['top50_precision']:.3f} "
            f"PR-AUC={m['pr_auc']:.3f} F1={m['f1']:.3f} Brier={m['brier']:.3f} "
            f"| mp={model_params} tp={{lr:{train_params['lr']:.2e}, wd:{train_params['weight_decay']:.2e}, "
            f"bs:{train_params['batch_size']}, sampler:{train_params['use_sampler']}, pos_mult:{train_params['pos_weight_mult']}}}"
        )

        if np.isnan(score):
            continue
        if score > best_score:
            best_score = score
            best_pack = {
                "best_score": best_score,
                "best_metrics_tuning": m,
                "best_model_params": model_params,
                "best_train_params": train_params,
            }

    if not best_pack:
        raise RuntimeError("Random search failed: best_pack is empty (all trials returned NaN).")

    return best_pack


def save_best_params(best_pack: Dict[str, Any], tag: str) -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base, ext = os.path.splitext(CONFIG["BEST_PARAMS_FILENAME"])
    out_path = os.path.join(script_dir, f"{base}_{tag}{ext}")

    lines = []
    lines.append(f"최적 하이퍼파라미터(Random Search) — {tag}\n\n")
    lines.append(f"Objective score = {best_pack['best_score']:.6f}\n")
    lines.append(f"Tuning Rolling Years = {CONFIG['TUNING_ROLLING_VAL_YEARS']}\n")
    lines.append(f"EarlyStop 기준 = {'objective(Top20/Top50)' if CONFIG['EARLYSTOP_ON_OBJECTIVE'] else 'val_loss'}\n")

    lines.append("\n[Best Tuning Metrics]\n")
    for k, v in best_pack["best_metrics_tuning"].items():
        lines.append(f"- {k}: {v:.6f}\n")

    lines.append("\n[Best MODEL_PARAMS]\n")
    for k, v in best_pack["best_model_params"].items():
        lines.append(f"- {k}: {v}\n")

    lines.append("\n[Best TRAIN_PARAMS]\n")
    for k, v in best_pack["best_train_params"].items():
        lines.append(f"- {k}: {v}\n")

    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"\n[INFO] Best params saved: {out_path}")
    return out_path


# =========================================================
# 9) 출력 테이블
# =========================================================
def build_terminal_table(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    scheme_order = ["holdout", "rolling", "groupkfold"]
    metric_order = ["pr_auc", "top20_precision", "top50_precision", "f1", "brier"]

    rows = []
    for scheme in scheme_order:
        rows.append([results[scheme][m] for m in metric_order])

    df = pd.DataFrame(rows, index=scheme_order, columns=metric_order)

    df = df.rename(columns={
        "pr_auc": "PR-AUC",
        "top20_precision": "Top-20 Precision",
        "top50_precision": "Top-50 Precision",
        "f1": "F1 Score",
        "brier": "Brier Score",
    })
    df.index.name = "validation"
    return df


# =========================================================
# 10) 단일 실험 실행(피처셋만 바꿔서 2번 호출)
# =========================================================
def run_experiment(df_raw: pd.DataFrame, tag: str, feature_cols: List[str]) -> None:
    X_all, y_all = prepare_Xy_with_feature_cols(df_raw, feature_cols)
    df_all = df_raw.loc[X_all.index].copy()

    dev_mask = (df_all[CONFIG["COL_YEAR"]] >= CONFIG["DEV_YEARS_MIN"]) & (df_all[CONFIG["COL_YEAR"]] <= CONFIG["DEV_YEARS_MAX"])
    df_dev = df_all[dev_mask].copy()

    print("\n" + "-" * 80)
    print(f"[EXPERIMENT] {tag}")
    print("-" * 80)
    print(f"[INFO] Total usable samples: {len(df_all)}")
    print(f"[INFO] Dev samples({CONFIG['DEV_YEARS_MIN']}~{CONFIG['DEV_YEARS_MAX']}): {len(df_dev)} | "
          f"Pos={(df_dev[CONFIG['COL_TARGET']]==1).sum()} Neg={(df_dev[CONFIG['COL_TARGET']]==0).sum()}")
    print(f"[INFO] Num features: {len(feature_cols)}")
    print(f"[INFO] Example features (first 10): {feature_cols[:10]}")

    if CONFIG["USE_RANDOM_SEARCH"]:
        print("\n" + "-" * 80)
        print(f"[TUNING] Random Search (FAST) — {tag}")
        print(f"  - tuning rolling years: {CONFIG['TUNING_ROLLING_VAL_YEARS']}")
        print(f"  - trials: {CONFIG['N_TRIALS_PER_FEATURESET']}")
        print("-" * 80)

        best_pack = run_random_search(df_dev, feature_cols, tag=tag)

        if CONFIG["SAVE_BEST_PARAMS"]:
            save_best_params(best_pack, tag=tag)

        best_model_params = best_pack["best_model_params"]
        best_train_params = best_pack["best_train_params"]

        print("\n[INFO] Best params selected for final evaluations.")
        print(f"  - best_score(tuning): {best_pack['best_score']:.6f}")
        print(f"  - MODEL_PARAMS: {best_model_params}")
        print(f"  - TRAIN_PARAMS: { {k: (f'{v:.2e}' if isinstance(v,float) and (k in ['lr','weight_decay']) else v) for k,v in best_train_params.items()} }")
    else:
        raise RuntimeError("이 비교 실험에서는 Random Search 사용을 권장합니다. (CONFIG['USE_RANDOM_SEARCH']=True)")

    print("\n[HOLDOUT] Time holdout final test...")
    res_holdout = evaluate_holdout_time(df_all, feature_cols, best_model_params, best_train_params)

    print("\n[ROLLING] Rolling(expanding) — FULL REPORT...")
    res_rolling = evaluate_rolling_dev(
        df_dev=df_dev,
        feature_cols=feature_cols,
        model_params=best_model_params,
        train_params=best_train_params,
        val_years=CONFIG["ROLLING_VAL_YEARS"],
    )

    print("\n[GROUPKFOLD] Company generalization check (optional)...")
    res_group = evaluate_groupkfold_dev(df_dev, feature_cols, best_model_params, best_train_params)

    results = {"holdout": res_holdout, "rolling": res_rolling, "groupkfold": res_group}

    table = build_terminal_table(results).T

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 80)
    pd.set_option("display.float_format", lambda x: f"{x:.3f}" if pd.notna(x) else "nan")

    print("\n" + "=" * 80)
    print(f"[RESULT TABLE] (metric x validation)  — GRU ({tag})")
    print("=" * 80)
    print(table)

    print("\n[DETAIL]")
    for scheme, m in results.items():
        print(
            f"- {scheme}: PR-AUC={m['pr_auc']:.3f}, "
            f"Top20={m['top20_precision']:.3f}, Top50={m['top50_precision']:.3f}, "
            f"F1(th={CONFIG['THRESHOLD']})={m['f1']:.3f}, Brier={m['brier']:.3f}"
        )


# =========================================================
# 11) main
# =========================================================
def main():
    print("=" * 80)
    print("GRU Feature-Set Comparison — (재무) vs (재무+특허) [FAST TUNING + FULL REPORT]")
    print(f"- Dataset: {os.path.basename(CONFIG['DATA_PATH'])}")
    print(f"- Threshold(F1): {CONFIG['THRESHOLD']}")
    print(f"- Top-20/50 Precision: K={CONFIG['TOP20_K']}/{CONFIG['TOP50_K']}")
    print(f"- Final Test(Time Holdout): target_year == {CONFIG['HOLDOUT_TARGET_YEAR']}")
    print(f"- Dev Pool(feature_year): {CONFIG['DEV_YEARS_MIN']}~{CONFIG['DEV_YEARS_MAX']}")
    print(f"- Trials per feature-set: {CONFIG['N_TRIALS_PER_FEATURESET']} (objective: 0.7*Top20 + 0.3*Top50)")
    print(f"- Tuning rolling years: {CONFIG['TUNING_ROLLING_VAL_YEARS']} (FAST)")
    print(f"- Final rolling years: {CONFIG['ROLLING_VAL_YEARS']} (REPORT)")
    print(f"- Early stopping: {'objective(Top20/Top50)' if CONFIG['EARLYSTOP_ON_OBJECTIVE'] else 'val loss'}")
    print("=" * 80)

    set_seed(CONFIG["SEED"])

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))

    df_raw = load_dataset(project_root)

    if not FIN_FEATS_12:
        raise ValueError(
            "FIN_FEATS_12(재무 12개 컬럼명 리스트)가 비어있습니다. "
            "데이터 컬럼명 그대로 12개를 입력한 뒤 실행하세요."
        )

    fin_only_cols = FIN_FEATS_12
    fin_patent_cols = FIN_FEATS_12 + PATENT_FEATS_15

    run_experiment(df_raw=df_raw, tag="재무", feature_cols=fin_only_cols)
    run_experiment(df_raw=df_raw, tag="재무+특허", feature_cols=fin_patent_cols)

    print("\nDONE.")


if __name__ == "__main__":
    main()

