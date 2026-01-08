import re
import numpy as np
import pandas as pd

# -----------------------------
# 1) 설정 (경로 및 컬럼 정의)
# -----------------------------
INPUT_PATH = "재무정보_final_v3.csv"
OUTPUT_PATH = "재무정보_최종_회계서식_완료.csv"

# 수치 데이터 처리 대상 컬럼
NUM_COLS = ["매출액", "영업이익", "당기순이익", "자산", "부채", "자본", "연구개발비", "CAPEX"]
# 0 미만 방지(하한 0) 적용 컬럼
NONNEG_COLS = ["매출액", "자산", "부채", "자본", "연구개발비", "CAPEX"]

# -----------------------------
# 2) 유틸리티 함수 정의
# -----------------------------

def parse_year(x):
    """연도 텍스트에서 4자리 숫자 추출"""
    s = str(x).strip()
    m = re.search(r"\d{4}", s)
    return int(m.group()) if m else np.nan

def to_number(x):
    """다양한 형태의 숫자/결측치 텍스트를 float으로 변환"""
    if pd.isna(x):
        return np.nan
    s = str(x).strip().replace(",", "")
    if s == "" or s.lower() == "nan" or s in ["-", "결측", "null", "none"]:
        return np.nan
    try:
        return float(s)
    except:
        return np.nan

def linear_interp_and_extrapolate(y: pd.Series, x: pd.Series, clamp_nonneg: bool) -> pd.Series:
    """선형 보간 및 양끝 외삽 처리"""
    y = y.astype(float).copy()
    x = x.astype(int).copy()

    if y.notna().sum() < 2:
        return y

    s = pd.Series(y.values, index=x.values).sort_index()
    
    # 1. 내부 결측 선형보간
    s_filled = s.interpolate(method="index", limit_area="inside")

    valid_idx = np.where(s_filled.notna().values)[0]
    if len(valid_idx) < 2:
        return pd.Series(s_filled.loc[x.values].values, index=y.index)

    # 2. 앞단 외삽
    first, second = valid_idx[0], valid_idx[1]
    x1, x2 = s_filled.index[first], s_filled.index[second]
    y1, y2 = s_filled.iloc[first], s_filled.iloc[second]
    slope_left = (y2 - y1) / (x2 - x1) if x2 != x1 else 0.0

    for i in range(0, first):
        xi = s_filled.index[i]
        val = y1 + slope_left * (xi - x1)
        if clamp_nonneg: val = max(0, val)
        s_filled.iloc[i] = val

    # 3. 뒷단 외삽
    last, prev = valid_idx[-1], valid_idx[-2]
    x1, x2 = s_filled.index[prev], s_filled.index[last]
    y1, y2 = s_filled.iloc[prev], s_filled.iloc[last]
    slope_right = (y2 - y1) / (x2 - x1) if x2 != x1 else 0.0

    for i in range(last + 1, len(s_filled)):
        xi = s_filled.index[i]
        val = y2 + slope_right * (xi - x2)
        if clamp_nonneg: val = max(0, val)
        s_filled.iloc[i] = val

    return pd.Series(s_filled.loc[x.values].values, index=y.index)

def format_to_accounting(x):
    """회계 서식 적용 (천 단위 쉼표 문자열)"""
    if pd.isna(x):
        return "0"
    s = str(x).strip().replace(",", "")
    if s == "" or s.lower() == "nan" or s in ["-", "결측", "null", "none"]:
        return "0"
    try:
        num = float(s)
        return "{:,.0f}".format(num)
    except:
        return "0"

# -----------------------------
# 3) 메인 실행 로직
# -----------------------------
try:
    print("1단계: 데이터 로드 및 기초 전처리 중...")
    try:
        df = pd.read_csv(INPUT_PATH, encoding="cp949")
    except UnicodeDecodeError:
        df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig")

    # 연도 정제
    df["연도"] = df["연도"].apply(parse_year)
    df = df.dropna(subset=["연도"])
    df["연도"] = df["연도"].astype(int)
    
    # 기업명 정제
    df["기업명"] = df["기업명"].astype(str).str.strip()

    # 숫자 변환 (CAPEX 보정을 위해 유형자산_증감 포함)
    process_cols = NUM_COLS + ["유형자산_증감"]
    for col in process_cols:
        if col in df.columns:
            df[col] = df[col].apply(to_number)

    # CAPEX 보정 (CAPEX가 없고 유형자산_증감이 있는 경우)
    if "CAPEX" in df.columns and "유형자산_증감" in df.columns:
        mask = df["CAPEX"].isna() & df["유형자산_증감"].notna()
        df.loc[mask, "CAPEX"] = df.loc[mask, "유형자산_증감"]

    print("2단계: 기업별 연도 보간 및 외삽 처리 중...")
    def process_group(group: pd.DataFrame) -> pd.DataFrame:
        group = group.sort_values("연도").copy()
        
        # 연도 구간 완성
        year_min, year_max = group["연도"].min(), group["연도"].max()
        full_years = pd.Index(range(year_min, year_max + 1), name="연도")
        
        group2 = group.set_index("연도").reindex(full_years).reset_index()
        group2["기업명"] = group["기업명"].iloc[0]
        
        # 기타 식별 정보 채우기
        for id_col in ["bsnsr_reg_nocrno", "dart_corp_code", "sectors", "field"]:
            if id_col in group2.columns:
                group2[id_col] = group[id_col].dropna().iloc[0] if group[id_col].notna().any() else group2[id_col]

        # 보간 및 외삽 적용
        for col in NUM_COLS:
            if col in group2.columns:
                clamp = col in NONNEG_COLS
                group2[col] = linear_interp_and_extrapolate(group2[col], group2["연도"], clamp_nonneg=clamp)
        return group2

    df_final = df.groupby("기업명", group_keys=False).apply(process_group)

    print("3단계: 회계 서식 적용 및 최종 저장 중...")
    # 회계 서식 적용
    for col in NUM_COLS:
        if col in df_final.columns:
            df_final[col] = df_final[col].apply(format_to_accounting)

    # 최종 저장
    df_final.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    
    print("-" * 30)
    print("모든 작업이 완료되었습니다!")
    print(f"최종 결과 파일: {OUTPUT_PATH}")

except Exception as e:
    print(f"실행 중 오류 발생: {e}")



    




