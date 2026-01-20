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
    "DATA_PATH": os.path.join("data", "total_dataset_patent_feature_add_수정.csv"),
    "ENCODING_CANDIDATES": ["utf-8-sig", "cp949"],

    "COL_COMPANY": "기업명",
    "COL_YEAR": "연도",                 # feature_year (t)
    "COL_TARGET_YEAR": "target_year",   # t+1
    "COL_TARGET": "target_growth",      # 0/1

    "HOLDOUT_TARGET_YEAR": 2024,

    "DEV_YEARS_MIN": 2019,
    "DEV_YEARS_MAX": 2022,
    "ROLLING_VAL_YEARS": [2020, 2021, 2022],

    # ====== Threshold 관련 ======
    "THRESHOLD_FIXED": 0.67,         # (내부 기본값으로만 사용; 출력/평가 항목에서는 제외)
    "TUNE_THRESHOLD_FOR_F1": True,   # 검증셋에서 F1 best threshold도 계산 (best_th 자체는 출력 제외)
    "THRESH_GRID_MIN": 0.05,
    "THRESH_GRID_MAX": 0.95,
    "THRESH_GRID_N": 181,            # 0.05~0.95를 촘촘히

    "TOP20_K": 20,
    "TOP50_K": 50,

    "SEED": 42,
    "GROUPKFOLD_SPLITS": 5,

    # ===========================
    # Random Search 튜닝 설정
    # ===========================
    "USE_RANDOM_SEARCH": True,
    "N_TRIALS_PER_FEATURESET": 12,   # 8 -> 12 (특허셋은 탐색공간이 커서 약간 늘림)
    "SAVE_BEST_PARAMS": True,
    "BEST_PARAMS_FILENAME": "best_params.txt",

    # ===========================
    # 멀티메트릭 Objective (Top-K 편향 완화)
    #  - Top20/Top50 + PR-AUC + F1(best_th) - Brier
    # ===========================
    "OBJECTIVE_WEIGHTS": {
        "top20_precision": 0.35,
        "top50_precision": 0.15,
        "pr_auc": 0.35,
        "f1_best": 0.25,
        "brier": 0.10,   # score에서 "빼기"로 적용
    },

    # ===========================
    # Early Stopping 기준
    # ===========================
    "EARLYSTOP_ON_OBJECTIVE": True,
    "EPS_IMPROVE": 1e-4,

    # ===========================
    # 튜닝 단계 가속 설정
    # ===========================
    "TUNING_ROLLING_VAL_YEARS": [2022],
    "TUNING_MAX_EPOCHS": 60,     # 50 -> 60
    "TUNING_PATIENCE": 10,       # 8  -> 10

    # ===========================
    # 특허 피처 변환(설계 변경)
    # ===========================
    "PATENT_CLIP_Q": 0.99,       # train 기준 99% 상한으로 clip
    "PATENT_LOG1P": True,
    "ADD_HAS_PATENT": True,      # no_patent_flag가 없어도 has_patent 생성

    # ===========================
    # 확률 보정(Temperature scaling)
    # ===========================
    "USE_TEMPERATURE_SCALING": True,  # Brier 개선 목적
    "TEMP_LR": 0.05,
    "TEMP_STEPS": 150,
}


# =========================================================
# (중요) 피처셋 정의
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
# 1) LSTM 모델
# =========================================================
class LSTMClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_dim=256, layer_dim=2, dropout_prob=0.2, bidirectional=True):
        super().__init__()
        self.num_dir = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
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
        out, _ = self.lstm(x)
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

    df.columns = df.columns.astype(str).str.strip()

    required = [CONFIG["COL_COMPANY"], CONFIG["COL_YEAR"], CONFIG["COL_TARGET_YEAR"], CONFIG["COL_TARGET"]]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing}")

    df[CONFIG["COL_YEAR"]] = pd.to_numeric(df[CONFIG["COL_YEAR"]], errors="coerce")
    df[CONFIG["COL_TARGET_YEAR"]] = pd.to_numeric(df[CONFIG["COL_TARGET_YEAR"]], errors="coerce")
    df[CONFIG["COL_TARGET"]] = pd.to_numeric(df[CONFIG["COL_TARGET"]], errors="coerce")

    return df


def prepare_Xy_with_feature_cols(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    지정한 feature_cols만 사용해서 X, y를 구성
    - 누락 컬럼은 제외(경고) 후 진행
    """
    existing = [c for c in feature_cols if c in df.columns]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"[WARN] Missing feature columns excluded ({len(missing)}): {missing[:10]}{'...' if len(missing) > 10 else ''}")

    X = df[existing].copy()
    y = df[CONFIG["COL_TARGET"]].copy()

    X = X.apply(pd.to_numeric, errors="coerce")

    mask = ~(X.isnull().any(axis=1) | y.isnull())
    X = X[mask]
    y = y[mask].astype(int)

    uniq = set(y.unique().tolist())
    if not uniq.issubset({0, 1}):
        raise ValueError(f"{CONFIG['COL_TARGET']}는 0/1이어야 합니다. 현재 값: {sorted(list(uniq))}")

    return X, y, existing


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


def brier_score(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    return float(np.mean((y_proba - y_true) ** 2))


def find_best_threshold_for_f1(y_true: np.ndarray, y_proba: np.ndarray) -> Tuple[float, float]:
    """
    검증셋 기준 F1 최대 threshold 탐색
    (best_th 자체는 반환하되, 최종 출력/테이블에서는 제외)
    """
    thr_grid = np.linspace(CONFIG["THRESH_GRID_MIN"], CONFIG["THRESH_GRID_MAX"], CONFIG["THRESH_GRID_N"])
    best_f1 = -1.0
    best_t = CONFIG["THRESHOLD_FIXED"]
    for t in thr_grid:
        pred = (y_proba >= t).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t, float(best_f1)


def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    """
    요청 반영:
      - F1@0.67(f1_fixed) 제거
      - best_th(best_threshold) 제거
      - 남길 항목: PR-AUC, Top-20, Top-50, F1@best(th), Brier
    """
    pr_auc = float(average_precision_score(y_true, y_proba))
    top20 = float(topk_precision_np(y_true, y_proba, CONFIG["TOP20_K"]))
    top50 = float(topk_precision_np(y_true, y_proba, CONFIG["TOP50_K"]))
    brier = float(brier_score(y_true, y_proba))

    if CONFIG["TUNE_THRESHOLD_FOR_F1"]:
        _best_t, f1_best = find_best_threshold_for_f1(y_true, y_proba)
    else:
        # 튜닝을 끄면 내부적으로 fixed threshold로 계산하되, 출력은 F1@best(th)만 유지
        pred = (y_proba >= CONFIG["THRESHOLD_FIXED"]).astype(int)
        f1_best = float(f1_score(y_true, pred, zero_division=0))

    return {
        "pr_auc": pr_auc,
        "top20_precision": top20,
        "top50_precision": top50,
        "f1_best": float(f1_best),
        "brier": brier
    }


def objective_score(metrics: Dict[str, float]) -> float:
    """
    멀티메트릭 목적함수:
      + Top20 + Top50 + PR-AUC + F1(best) - Brier
    """
    w = CONFIG["OBJECTIVE_WEIGHTS"]
    score = (
        w["top20_precision"] * metrics["top20_precision"]
        + w["top50_precision"] * metrics["top50_precision"]
        + w["pr_auc"] * metrics["pr_auc"]
        + w["f1_best"] * metrics["f1_best"]
        - w["brier"] * metrics["brier"]
    )
    return float(score)


@torch.no_grad()
def predict_logits(model: nn.Module, X_tensor: torch.Tensor, device: torch.device) -> np.ndarray:
    model.eval()
    logits = model(X_tensor.to(device)).detach().cpu().numpy().reshape(-1)
    return logits


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
# 3-1) 설계 변경: 특허 피처 변환기 (Train 통계 기반)
# =========================================================
def _is_patent_col(col: str) -> bool:
    return col in PATENT_FEATS_15 or col.startswith("patent_") or col.startswith("ipc_") or "citation" in col


class PatentFeatureTransformer:
    """
    - train 기준 quantile clip -> log1p
    - no_patent_flag가 없으면 has_patent 생성(특허 관련 컬럼 합>0)
    """
    def __init__(self, feature_cols: List[str]):
        self.feature_cols = feature_cols
        self.patent_cols = [c for c in feature_cols if _is_patent_col(c)]
        self.clip_upper_: Dict[str, float] = {}
        self.has_no_patent_flag = ("no_patent_flag" in feature_cols)

    def fit(self, X_train_df: pd.DataFrame) -> "PatentFeatureTransformer":
        if len(self.patent_cols) == 0:
            return self

        for c in self.patent_cols:
            x = pd.to_numeric(X_train_df[c], errors="coerce").fillna(0.0)
            upper = float(np.nanquantile(x.values, CONFIG["PATENT_CLIP_Q"])) if len(x) > 0 else 0.0
            if not np.isfinite(upper):
                upper = 0.0
            self.clip_upper_[c] = upper
        return self

    def transform(self, X_df: pd.DataFrame) -> pd.DataFrame:
        X = X_df.copy()

        # 특허 컬럼 결측 -> 0 (희소성 반영)
        for c in self.patent_cols:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)

            # clip
            upper = self.clip_upper_.get(c, None)
            if upper is not None and upper > 0:
                X[c] = np.minimum(X[c].values, upper)

            # log1p
            if CONFIG["PATENT_LOG1P"]:
                X[c] = np.log1p(np.maximum(X[c].values, 0.0))

        # no_patent_flag가 없으면 has_patent 생성
        if CONFIG["ADD_HAS_PATENT"] and (not self.has_no_patent_flag):
            if len(self.patent_cols) > 0:
                s = np.zeros(len(X), dtype=np.float32)
                for c in self.patent_cols:
                    s += X[c].values.astype(np.float32)
                X["has_patent"] = (s > 0).astype(np.float32)
            else:
                X["has_patent"] = 0.0

        return X


# =========================================================
# 3-2) 설계 변경: Temperature scaling (간단 확률 보정)
# =========================================================
def temperature_scale_logits(logits: np.ndarray, y_true: np.ndarray, device: torch.device) -> float:
    """
    validation logits로 temperature T를 맞춤 (NLL 최소화).
    T>0.5~5.0 사이로 clamp.
    """
    logits_t = torch.tensor(logits, dtype=torch.float32, device=device).view(-1, 1)
    y_t = torch.tensor(y_true.astype(np.float32), dtype=torch.float32, device=device).view(-1, 1)

    T = torch.tensor([1.0], dtype=torch.float32, device=device, requires_grad=True)
    opt = optim.Adam([T], lr=CONFIG["TEMP_LR"])
    criterion = nn.BCEWithLogitsLoss()

    for _ in range(CONFIG["TEMP_STEPS"]):
        opt.zero_grad()
        scaled = logits_t / torch.clamp(T, 1e-3, 100.0)
        loss = criterion(scaled, y_t)
        loss.backward()
        opt.step()

        with torch.no_grad():
            T.clamp_(0.5, 5.0)

    return float(T.detach().cpu().numpy().reshape(-1)[0])


# =========================================================
# 4) 학습 (Early stopping: objective 기반)
# =========================================================
def train_lstm(
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

        # ---- validation ----
        model.eval()
        with torch.no_grad():
            v_logits = model(X_val_t.to(device)).detach().cpu().numpy().reshape(-1)
            y_true = y_val_t.detach().cpu().numpy().reshape(-1).astype(int)
            v_proba = sigmoid_np(v_logits)

            metrics = compute_metrics(y_true=y_true, y_proba=v_proba)

            if CONFIG["EARLYSTOP_ON_OBJECTIVE"]:
                v_obj = objective_score(metrics)
                if v_obj > best_val_obj + CONFIG["EPS_IMPROVE"]:
                    best_val_obj = v_obj
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                v_logits_t = torch.tensor(v_logits, dtype=torch.float32, device=device).view(-1, 1)
                v_loss = criterion(v_logits_t, y_val_t.to(device)).item()
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
# 6) 학습+평가 1회 실행(스케일링 + 특허 변환 + (옵션)보정)
# =========================================================
def fit_and_eval_once(
    X_train_df: pd.DataFrame,
    y_train: np.ndarray,
    X_val_df: pd.DataFrame,
    y_val: np.ndarray,
    feature_cols: List[str],
    device: torch.device,
    model_params: Dict[str, Any],
    train_params: Dict[str, Any],
) -> Dict[str, float]:
    transformer = PatentFeatureTransformer(feature_cols=feature_cols)
    transformer.fit(X_train_df)

    X_tr_f = transformer.transform(X_train_df)
    X_va_f = transformer.transform(X_val_df)

    used_cols = list(X_tr_f.columns)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr_f.values.astype(np.float32))
    X_va_s = scaler.transform(X_va_f.values.astype(np.float32))

    X_tr_t = to_tensor_3d(X_tr_s.astype(np.float32))
    X_va_t = to_tensor_3d(X_va_s.astype(np.float32))
    y_tr_t = torch.FloatTensor(y_train.astype(np.float32)).view(-1, 1)
    y_va_t = torch.FloatTensor(y_val.astype(np.float32)).view(-1, 1)

    model = LSTMClassifier(
        input_size=len(used_cols),
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

    model = train_lstm(
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

    # ---- 평가: logits -> (옵션) temperature scaling -> proba ----
    val_logits = predict_logits(model, X_va_t, device)
    val_proba = sigmoid_np(val_logits)

    if CONFIG["USE_TEMPERATURE_SCALING"]:
        T = temperature_scale_logits(val_logits, y_val.astype(int), device=device)
        val_proba = sigmoid_np(val_logits / max(T, 1e-6))

    metrics = compute_metrics(y_true=y_val.astype(int), y_proba=val_proba)
    return metrics


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

    X_tr_df = train_df[feature_cols].copy()
    y_tr = train_df[CONFIG["COL_TARGET"]].values.astype(int)
    X_te_df = test_df[feature_cols].copy()
    y_te = test_df[CONFIG["COL_TARGET"]].values.astype(int)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return fit_and_eval_once(
        X_train_df=X_tr_df, y_train=y_tr,
        X_val_df=X_te_df, y_val=y_te,
        feature_cols=feature_cols,
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

        X_tr_df = train_df[feature_cols].copy()
        y_tr = train_df[CONFIG["COL_TARGET"]].values.astype(int)
        X_va_df = val_df[feature_cols].copy()
        y_va = val_df[CONFIG["COL_TARGET"]].values.astype(int)

        m = fit_and_eval_once(
            X_train_df=X_tr_df, y_train=y_tr,
            X_val_df=X_va_df, y_val=y_va,
            feature_cols=feature_cols,
            device=device,
            model_params=model_params,
            train_params=train_params
        )
        metrics_list.append(m)

    if len(metrics_list) == 0:
        return {
            "pr_auc": np.nan,
            "top20_precision": np.nan,
            "top50_precision": np.nan,
            "f1_best": np.nan,
            "brier": np.nan
        }

    keys = metrics_list[0].keys()
    return {k: float(np.mean([mm[k] for mm in metrics_list])) for k in keys}


def evaluate_groupkfold_dev(
    df_dev: pd.DataFrame, feature_cols: List[str],
    model_params: Dict[str, Any], train_params: Dict[str, Any]
) -> Dict[str, float]:
    gkf = GroupKFold(n_splits=CONFIG["GROUPKFOLD_SPLITS"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_dev_df = df_dev[feature_cols].copy()
    y_dev = df_dev[CONFIG["COL_TARGET"]].values.astype(int)
    groups = df_dev[CONFIG["COL_COMPANY"]].values

    fold_metrics: List[Dict[str, float]] = []
    for tr_idx, va_idx in gkf.split(X_dev_df.values, y_dev, groups=groups):
        tr_df = X_dev_df.iloc[tr_idx].copy()
        va_df = X_dev_df.iloc[va_idx].copy()
        y_tr = y_dev[tr_idx]
        y_va = y_dev[va_idx]

        m = fit_and_eval_once(
            X_train_df=tr_df, y_train=y_tr,
            X_val_df=va_df, y_val=y_va,
            feature_cols=feature_cols,
            device=device,
            model_params=model_params,
            train_params=train_params
        )
        fold_metrics.append(m)

    keys = fold_metrics[0].keys()
    return {k: float(np.mean([fm[k] for fm in fold_metrics])) for k in keys}


# =========================================================
# 8) Random Search
# =========================================================
def sample_params(rng: np.random.RandomState, fast_tuning: bool = True) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    model_params = {
        "hidden_dim": int(rng.choice([64, 128, 256, 384])),
        "layer_dim": int(rng.choice([1, 2, 3])),
        "dropout_prob": float(rng.choice([0.2, 0.3, 0.4, 0.5])),
        "bidirectional": bool(rng.choice([True, False])),
    }

    lr = float(10 ** rng.uniform(np.log10(1e-4), np.log10(5e-3)))
    wd = float(10 ** rng.uniform(np.log10(1e-7), np.log10(5e-3)))

    if fast_tuning:
        max_epochs = int(rng.choice([50, 60, 70]))
        patience = int(rng.choice([8, 10, 12]))
    else:
        max_epochs = int(rng.choice([80, 100, 120]))
        patience = int(rng.choice([12, 16, 20]))

    use_sampler = bool(rng.choice([False, False, True]))

    train_params = {
        "max_epochs": max_epochs,
        "patience": patience,
        "lr": lr,
        "weight_decay": wd,
        "batch_size": int(rng.choice([64, 128, 256])),
        "use_sampler": use_sampler,
        "pos_weight_mult": float(rng.choice([1.0, 1.5, 2.0, 2.5, 3.0])),
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
            f"PR-AUC={m['pr_auc']:.3f} F1@best={m['f1_best']:.3f} "
            f"Brier={m['brier']:.3f} "
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
    lines.append(f"EarlyStop 기준 = {'objective(multi-metric)' if CONFIG['EARLYSTOP_ON_OBJECTIVE'] else 'val_loss'}\n")

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
    metric_order = ["pr_auc", "top20_precision", "top50_precision", "f1_best", "brier"]

    rows = []
    for scheme in scheme_order:
        rows.append([results[scheme][m] for m in metric_order])

    df = pd.DataFrame(rows, index=scheme_order, columns=metric_order)

    df = df.rename(columns={
        "pr_auc": "PR-AUC",
        "top20_precision": "Top-20 Precision",
        "top50_precision": "Top-50 Precision",
        "f1_best": "F1@best(th)",
        "brier": "Brier Score",
    })
    df.index.name = "validation"
    return df


# =========================================================
# 10) 단일 실험 실행
# =========================================================
def run_experiment(df_raw: pd.DataFrame, tag: str, feature_cols: List[str]) -> None:
    X_all, y_all, used_cols = prepare_Xy_with_feature_cols(df_raw, feature_cols)

    df_all = df_raw.loc[X_all.index].copy()

    dev_mask = (df_all[CONFIG["COL_YEAR"]] >= CONFIG["DEV_YEARS_MIN"]) & (df_all[CONFIG["COL_YEAR"]] <= CONFIG["DEV_YEARS_MAX"])
    df_dev = df_all[dev_mask].copy()

    print("\n" + "-" * 80)
    print(f"[EXPERIMENT] {tag}")
    print("-" * 80)
    print(f"[INFO] Total usable samples: {len(df_all)}")
    print(f"[INFO] Dev samples({CONFIG['DEV_YEARS_MIN']}~{CONFIG['DEV_YEARS_MAX']}): {len(df_dev)} | "
          f"Pos={(df_dev[CONFIG['COL_TARGET']]==1).sum()} Neg={(df_dev[CONFIG['COL_TARGET']]==0).sum()}")
    print(f"[INFO] Num features(requested): {len(feature_cols)} | Num features(used): {len(used_cols)}")
    print(f"[INFO] Example features (first 12 used): {used_cols[:12]}")

    if CONFIG["USE_RANDOM_SEARCH"]:
        print("\n" + "-" * 80)
        print(f"[TUNING] Random Search (FAST) — {tag}")
        print(f"  - tuning rolling years: {CONFIG['TUNING_ROLLING_VAL_YEARS']}")
        print(f"  - trials: {CONFIG['N_TRIALS_PER_FEATURESET']}")
        print(f"  - objective: Top20/Top50 + PR-AUC + F1(best) - Brier")
        print("-" * 80)

        best_pack = run_random_search(df_dev, used_cols, tag=tag)

        if CONFIG["SAVE_BEST_PARAMS"]:
            save_best_params(best_pack, tag=tag)

        best_model_params = best_pack["best_model_params"]
        best_train_params = best_pack["best_train_params"]

        best_train_params["max_epochs"] = max(best_train_params["max_epochs"], 90)
        best_train_params["patience"] = max(best_train_params["patience"], 14)

        print("\n[INFO] Best params selected for final evaluations.")
        print(f"  - best_score(tuning): {best_pack['best_score']:.6f}")
        print(f"  - MODEL_PARAMS: {best_model_params}")
        print(f"  - TRAIN_PARAMS: { {k: (f'{v:.2e}' if isinstance(v,float) and (k in ['lr','weight_decay']) else v) for k,v in best_train_params.items()} }")
    else:
        raise RuntimeError("이 비교 실험에서는 Random Search 사용을 권장합니다. (CONFIG['USE_RANDOM_SEARCH']=True)")

    print("\n[HOLDOUT] Time holdout final test...")
    res_holdout = evaluate_holdout_time(df_all, used_cols, best_model_params, best_train_params)

    print("\n[ROLLING] Rolling(expanding) — FULL REPORT...")
    res_rolling = evaluate_rolling_dev(
        df_dev=df_dev,
        feature_cols=used_cols,
        model_params=best_model_params,
        train_params=best_train_params,
        val_years=CONFIG["ROLLING_VAL_YEARS"],
    )

    print("\n[GROUPKFOLD] Company generalization check (optional)...")
    res_group = evaluate_groupkfold_dev(df_dev, used_cols, best_model_params, best_train_params)

    results = {"holdout": res_holdout, "rolling": res_rolling, "groupkfold": res_group}

    table = build_terminal_table(results).T

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 80)
    pd.set_option("display.float_format", lambda x: f"{x:.3f}" if pd.notna(x) else "nan")

    print("\n" + "=" * 90)
    print(f"[RESULT TABLE] (metric x validation)  — LSTM ({tag})  [Enhanced: patent transform + multi-metric tuning]")
    print("=" * 90)
    print(table)

    print("\n[DETAIL]")
    for scheme, m in results.items():
        print(
            f"- {scheme}: PR-AUC={m['pr_auc']:.3f}, "
            f"Top20={m['top20_precision']:.3f}, Top50={m['top50_precision']:.3f}, "
            f"F1@best={m['f1_best']:.3f}, "
            f"Brier={m['brier']:.3f}"
        )


# =========================================================
# 11) main
# =========================================================
def main():
    print("=" * 90)
    print("LSTM Feature-Set Comparison — (재무) vs (재무+특허)  [Enhanced TUNING + FULL REPORT]")
    print(f"- Dataset: {os.path.basename(CONFIG['DATA_PATH'])}")
    print(f"- Tune threshold for F1: {CONFIG['TUNE_THRESHOLD_FOR_F1']} (F1@best(th) only)")
    print(f"- Top-20/50 Precision: K={CONFIG['TOP20_K']}/{CONFIG['TOP50_K']}")
    print(f"- Final Test(Time Holdout): target_year == {CONFIG['HOLDOUT_TARGET_YEAR']}")
    print(f"- Dev Pool(feature_year): {CONFIG['DEV_YEARS_MIN']}~{CONFIG['DEV_YEARS_MAX']}")
    print(f"- Trials per feature-set: {CONFIG['N_TRIALS_PER_FEATURESET']}")
    print(f"- Tuning rolling years: {CONFIG['TUNING_ROLLING_VAL_YEARS']} (FAST)")
    print(f"- Final rolling years: {CONFIG['ROLLING_VAL_YEARS']} (REPORT)")
    print(f"- Patent transform: clip_q={CONFIG['PATENT_CLIP_Q']}, log1p={CONFIG['PATENT_LOG1P']}, add_has_patent={CONFIG['ADD_HAS_PATENT']}")
    print(f"- Temperature scaling: {CONFIG['USE_TEMPERATURE_SCALING']}")
    print("=" * 90)

    set_seed(CONFIG["SEED"])

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))

    df_raw = load_dataset(project_root)

    if not FIN_FEATS_12:
        raise ValueError("FIN_FEATS_12(재무 12개 컬럼명 리스트)가 비어있습니다.")

    fin_only_cols = FIN_FEATS_12
    fin_patent_cols = FIN_FEATS_12 + PATENT_FEATS_15

    run_experiment(df_raw=df_raw, tag="재무", feature_cols=fin_only_cols)
    run_experiment(df_raw=df_raw, tag="재무+특허", feature_cols=fin_patent_cols)

    print("\nDONE.")


if __name__ == "__main__":
    main()

