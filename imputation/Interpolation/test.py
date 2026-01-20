import re
import numpy as np
import pandas as pd
import traceback

# =========================================================
# 1. 통합 설정
# =========================================================
FILE_CONFIG = {
    "INPUT_PATH": "재무정보_final_v2(16~24)_수정.csv",
    "OUTPUT_PATH": "전체기업_재무데이터_선형보간_260114_v0.2.csv",
    "TARGET_YEAR_MIN": 2016,
    "TARGET_YEAR_MAX": 2024,
    "NUM_COLS": ["매출액", "영업이익", "당기순이익", "자산총계", "부채총계", "자본총계", "연구개발비", "CAPEX"],
    "NONNEG_COLS": ["매출액", "자산총계", "부채총계", "자본총계", "연구개발비", "CAPEX"],
    "FLAG_COLS": ["매출액", "영업이익", "당기순이익", "자산총계", "부채총계", "자본총계", "CAPEX"]
}

ZERO_RULE_YEARS = [2016, 2017, 2018]
ZERO_RULE_TARGET_COLS = ["매출액", "영업이익", "당기순이익", "자산총계", "부채총계", "자본총계", "연구개발비", "CAPEX"]

# =========================================================
# 2. 공통 유틸리티 함수
# =========================================================
def to_number(x, zero_as_nan=False):
    if pd.isna(x): return np.nan
    if isinstance(x, (int, float)):
        val = float(x)
        return np.nan if (zero_as_nan and val == 0) else val
    
    s = str(x).strip().replace(",", "")
    if s in ["", "nan", "-", "결측", "null", "none", "NaN", "NONE", "NULL", "0"]:
        if s == "0" and not zero_as_nan: return 0.0
        return np.nan
    
    try:
        val = float(s)
        return np.nan if (zero_as_nan and val == 0) else val
    except ValueError:
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", s)
        if nums:
            val = float(nums[0])
            return np.nan if (zero_as_nan and val == 0) else val
        return np.nan

def format_accounting(x):
    try:
        val = float(x)
        if pd.isna(val) or val == 0: return "0"
        return "{:,.0f}".format(val)
    except: return "0"

def linear_interp_and_extrapolate(s_raw: pd.Series, clamp_nonneg: bool) -> pd.Series:
    s_filled = s_raw.interpolate(method="index", limit_area="inside")
    valid_indices = s_raw.index[s_raw.notna()]
    if len(valid_indices) < 2: return s_filled

    # 좌측 외삽
    x1, x2 = valid_indices[0], valid_indices[1]
    y1, y2 = s_raw.loc[x1], s_raw.loc[x2]
    slope_l = (y2 - y1) / (x2 - x1) if x2 != x1 else 0
    l_mask = s_filled.index < x1
    if l_mask.any():
        val_l = y1 + slope_l * (s_filled.index[l_mask] - x1)
        s_filled.loc[l_mask] = np.maximum(0, val_l) if clamp_nonneg else val_l

    # 우측 외삽
    xn_1, xn = valid_indices[-2], valid_indices[-1]
    yn_1, yn = s_raw.loc[xn_1], s_raw.loc[xn]
    slope_r = (yn - yn_1) / (xn - xn_1) if xn != xn_1 else 0
    r_mask = s_filled.index > xn
    if r_mask.any():
        val_r = yn + slope_r * (s_filled.index[r_mask] - xn)
        s_filled.loc[r_mask] = np.maximum(0, val_r) if clamp_nonneg else val_r
    return s_filled

# =========================================================
# 3. 로직별 처리 함수 (1 -> 2 -> 3 단계)
# =========================================================

# [STEP 1] 선형 보간 및 외삽 로직
def logic_1_linear_processing(g: pd.DataFrame) -> pd.DataFrame:
    g_in_range = g[(g["연도"] >= FILE_CONFIG["TARGET_YEAR_MIN"]) & (g["연도"] <= FILE_CONFIG["TARGET_YEAR_MAX"])].copy()
    if g_in_range.empty: return pd.DataFrame()

    for col in FILE_CONFIG["NUM_COLS"]:
        if col in g_in_range.columns:
            g_in_range[col] = g_in_range[col].apply(lambda x: to_number(x, False))

    year_level_sum = g_in_range[FILE_CONFIG["NUM_COLS"]].fillna(0).abs().sum(axis=1)
    if len(g_in_range.loc[year_level_sum > 0, "연도"].dropna().unique()) <= 1:
        return pd.DataFrame()

    # 2016-18 Zero Rule 판정
    years_present = set(g_in_range["연도"].dropna().unique().tolist())
    zero_rule_hit = False
    if set(ZERO_RULE_YEARS).issubset(years_present):
        rev_series = g_in_range.set_index("연도")["매출액"].reindex(ZERO_RULE_YEARS).fillna(0)
        zero_rule_hit = (rev_series == 0).all()

    # Reindex
    fixed_cols = [c for c in g_in_range.columns if c not in FILE_CONFIG["NUM_COLS"] and c != "연도"]
    fixed_info = {col: (g_in_range[col].dropna().head(1).values[0] if g_in_range[col].notna().any() else np.nan) for col in fixed_cols}
    
    g_res = g_in_range.drop_duplicates(subset=["연도"]).set_index("연도")
    g_res = g_res.reindex(range(FILE_CONFIG["TARGET_YEAR_MIN"], FILE_CONFIG["TARGET_YEAR_MAX"] + 1))
    for col, val in fixed_info.items(): g_res[col] = val

    # CAPEX 보정 및 보간
    if "CAPEX" in g_res.columns and "유형자산_증감" in g_res.columns:
        g_res["유형자산_증감"] = g_res["유형자산_증감"].apply(lambda x: to_number(x, False))
        invalid_capex = g_res["CAPEX"].isna() | (g_res["CAPEX"].fillna(0) == 0)
        if zero_rule_hit: invalid_capex = invalid_capex & (~g_res.index.isin(ZERO_RULE_YEARS))
        g_res.loc[invalid_capex, "CAPEX"] = g_res.loc[invalid_capex, "유형자산_증감"]

    for col in FILE_CONFIG["NUM_COLS"]:
        if col in g_res.columns:
            g_res[col] = linear_interp_and_extrapolate(g_res[col], clamp_nonneg=(col in FILE_CONFIG["NONNEG_COLS"]))
            g_res[col] = g_res[col].fillna(0)
    
    if zero_rule_hit:
        for col in ZERO_RULE_TARGET_COLS:
            if col in g_res.columns: g_res.loc[ZERO_RULE_YEARS, col] = 0
            
    return g_res.reset_index().rename(columns={"index": "연도"})

# [STEP 2 & 3] 결측치 전후 복사, 연구개발비 특수처리, Flag 생성 및 항등식
def logic_2_3_final_refinement(g: pd.DataFrame) -> pd.DataFrame:
    if g.empty: return g
    
    # 0을 NaN으로 간주하여 다시 수치 변환 (코드 2, 3 로직)
    for col in FILE_CONFIG["NUM_COLS"]:
        if col in g.columns:
            g[col] = g[col].apply(lambda x: to_number(x, True))

    # 연구개발비 특수 처리
    if "연구개발비" in g.columns and g["연구개발비"].isna().all():
        g["연구개발비"] = 1.0

    # Flag 생성을 위한 백업
    g_res = g.set_index("연_도" if "연_도" in g.columns else "연도")
    before_fill = g_res[FILE_CONFIG["FLAG_COLS"]].copy()

    # ffill / bfill
    for col in FILE_CONFIG["NUM_COLS"]:
        if col in g_res.columns:
            if col == "연구개발비" and (g_res[col] == 1.0).all(): continue
            g_res[col] = g_res[col].ffill().bfill()

    after_fill = g_res[FILE_CONFIG["FLAG_COLS"]].fillna(0)
    g_res[FILE_CONFIG["NUM_COLS"]] = g_res[FILE_CONFIG["NUM_COLS"]].fillna(0)

    # Flag 생성
    was_all_empty = before_fill.isna().all(axis=1)
    is_now_filled = after_fill.notna().all(axis=1)
    g_res['flag'] = (was_all_empty & is_now_filled).astype(int)

    # 회계 항등식
    if all(c in g_res.columns for c in ["자산총계", "부채총계", "자본총계"]):
        g_res["자산총계"] = g_res["부채총계"] + g_res["자본총계"]

    return g_res.reset_index().rename(columns={"index": "연도"})

# =========================================================
# 4. 메인 실행 프로세스
# =========================================================
if __name__ == "__main__":
    try:
        # 1. 파일 읽기
        try:
            df = pd.read_csv(FILE_CONFIG["INPUT_PATH"], encoding="cp949")
        except:
            df = pd.read_csv(FILE_CONFIG["INPUT_PATH"], encoding="utf-8-sig")

        original_column_order = list(df.columns)
        df.columns = df.columns.str.strip()
        df = df.dropna(subset=["기업명", "연도"])
        df["연도"] = df["연도"].apply(lambda x: to_number(x, False)).astype(int)

        # 2. 파이프라인 실행
        print("Step 1: 선형 보간 및 외삽 진행 중...")
        df_step1 = df.groupby("기업명", group_keys=False).apply(logic_1_linear_processing)

        print("Step 2 & 3: 데이터 보정, Flag 생성 및 항등식 적용 중...")
        df_final = df_step1.groupby("기업명", group_keys=False).apply(logic_2_3_final_refinement)

        # 3. 사후 처리 및 저장
        if df_final is not None and not df_final.empty:
            df_final = df_final.dropna(subset=["기업명"])

            print("회계 서식 적용 중...")
            for col in FILE_CONFIG["NUM_COLS"]:
                if col in df_final.columns:
                    df_final[col] = df_final[col].apply(format_accounting)

            # 컬럼 순서 조정 ('유형자산_증감' 뒤에 'flag' 배치)
            cols = list(df_final.columns)
            if '유형자산_증감' in cols and 'flag' in cols:
                cols.remove('flag')
                target_idx = cols.index('유형자산_증감') + 1
                cols.insert(target_idx, 'flag')
                df_final = df_final[cols]
            else:
                final_cols = [c for c in original_column_order if c in df_final.columns]
                if 'flag' not in final_cols: final_cols.append('flag')
                df_final = df_final[final_cols + [c for c in df_final.columns if c not in final_cols]]

            df_final.to_csv(FILE_CONFIG["OUTPUT_PATH"], index=False, encoding="utf-8-sig")
            print(f"모든 작업 완료! 최종 파일: {FILE_CONFIG['OUTPUT_PATH']}")

    except Exception as e:
        print(f"오류 발생: {e}")
        traceback.print_exc()
        



    




