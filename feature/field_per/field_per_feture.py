import pandas as pd
import numpy as np
import os
from typing import Optional


def calculate_growth_features(df: pd.DataFrame, 
                             company_col: str = '기업명',
                             year_col: str = '연도',
                             revenue_col: str = '  매출액  ',
                             operating_profit_col: str = '  영업이익  ',
                             capex_col: str = '  CAPEX  ',
                             fixed_asset_change_col: str = '  유형자산_증감  ',
                             debt_col: str = '  부채총계  ',
                             equity_col: str = '  자본총계  ',
                             rnd_col: str = '  연구개발비  ',
                             field_col: str = 'field',
                             feature_year: Optional[int] = None) -> pd.DataFrame:
    """
    성장 기업 예측을 위한 피처를 계산하는 함수
    
    Parameters:
    -----------
    df : pd.DataFrame
        재무정보 데이터프레임 (기업별, 연도별 데이터)
    company_col : str
        기업명 컬럼명
    year_col : str
        연도 컬럼명
    revenue_col : str
        매출액 컬럼명
    operating_profit_col : str
        영업이익 컬럼명
    feature_year : int, optional
        피처 기준 연도 (예: 2019년 말 피처 → feature_year=2019)
        이 연도까지의 데이터를 사용하여 피처 계산
    
    Returns:
    --------
    pd.DataFrame
        기업별 피처가 추가된 데이터프레임
    """
    
    # 데이터 복사
    df_processed = df.copy()
    
    # 컬럼명 자동 감지 (인코딩 문제 대비)
    def find_column(df, keywords, index_hint=None):
        """키워드가 포함된 컬럼명 찾기"""
        # 먼저 키워드로 찾기
        for col in df.columns:
            col_str = str(col).strip()
            if any(keyword in col_str for keyword in keywords):
                return col
        
        # 인덱스 힌트가 있으면 사용
        if index_hint is not None and index_hint < len(df.columns):
            return df.columns[index_hint]
        
        return None
    
    # 컬럼명이 없으면 자동으로 찾기 (인덱스 힌트 포함)
    if company_col not in df_processed.columns:
        company_col_found = find_column(df_processed, ['기업명', '기업', 'company', 'corp'], index_hint=0)
        if company_col_found:
            company_col = company_col_found
    
    if year_col not in df_processed.columns:
        year_col_found = find_column(df_processed, ['연도', 'year', '년도'], index_hint=6)
        if year_col_found:
            year_col = year_col_found
    
    if revenue_col not in df_processed.columns:
        revenue_col_found = find_column(df_processed, ['매출액', '매출', 'revenue', 'sales'], index_hint=7)
        if revenue_col_found:
            revenue_col = revenue_col_found
    
    if operating_profit_col not in df_processed.columns:
        operating_profit_col_found = find_column(df_processed, ['영업이익', '영업', 'operating', 'profit'], index_hint=8)
        if operating_profit_col_found:
            operating_profit_col = operating_profit_col_found
    
    # 필수 컬럼 확인
    if company_col not in df_processed.columns:
        print(f"경고: 기업명 컬럼을 찾을 수 없습니다. 사용 가능한 컬럼 인덱스:")
        for i, col in enumerate(df_processed.columns):
            print(f"  [{i}]: {repr(col)}")
        raise ValueError(f"기업명 컬럼을 찾을 수 없습니다. 컬럼 인덱스 0을 사용하세요.")
    if year_col not in df_processed.columns:
        print(f"경고: 연도 컬럼을 찾을 수 없습니다. 사용 가능한 컬럼 인덱스:")
        for i, col in enumerate(df_processed.columns):
            print(f"  [{i}]: {repr(col)}")
        raise ValueError(f"연도 컬럼을 찾을 수 없습니다. 컬럼 인덱스 6을 사용하세요.")
    if revenue_col not in df_processed.columns:
        print(f"경고: 매출액 컬럼을 찾을 수 없습니다. 사용 가능한 컬럼 인덱스:")
        for i, col in enumerate(df_processed.columns):
            print(f"  [{i}]: {repr(col)}")
        raise ValueError(f"매출액 컬럼을 찾을 수 없습니다. 컬럼 인덱스 7을 사용하세요.")
    if operating_profit_col not in df_processed.columns:
        print(f"경고: 영업이익 컬럼을 찾을 수 없습니다. 사용 가능한 컬럼 인덱스:")
        for i, col in enumerate(df_processed.columns):
            print(f"  [{i}]: {repr(col)}")
        raise ValueError(f"영업이익 컬럼을 찾을 수 없습니다. 컬럼 인덱스 8을 사용하세요.")
    
    # 매출액과 영업이익을 숫자형으로 변환 (쉼표 제거)
    if revenue_col in df_processed.columns:
        df_processed[revenue_col] = df_processed[revenue_col].astype(str).str.replace(',', '').replace(' - ', np.nan).replace('-', np.nan)
        df_processed[revenue_col] = pd.to_numeric(df_processed[revenue_col], errors='coerce')
    
    if operating_profit_col in df_processed.columns:
        df_processed[operating_profit_col] = df_processed[operating_profit_col].astype(str).str.replace(',', '').replace(' - ', np.nan).replace('-', np.nan)
        df_processed[operating_profit_col] = pd.to_numeric(df_processed[operating_profit_col], errors='coerce')
    
    # CAPEX 관련 컬럼 숫자형 변환
    if capex_col in df_processed.columns:
        df_processed[capex_col] = df_processed[capex_col].astype(str).str.replace(',', '').replace(' - ', np.nan).replace('-', np.nan)
        df_processed[capex_col] = pd.to_numeric(df_processed[capex_col], errors='coerce')
    
    if fixed_asset_change_col in df_processed.columns:
        # 유형자산_증감은 음수일 수 있으므로 괄호 처리
        df_processed[fixed_asset_change_col] = df_processed[fixed_asset_change_col].astype(str).str.replace(',', '').str.replace('(', '-').str.replace(')', '').replace(' - ', np.nan).replace('-', np.nan)
        df_processed[fixed_asset_change_col] = pd.to_numeric(df_processed[fixed_asset_change_col], errors='coerce')
    
    # 부채총계, 자본총계, 연구개발비 컬럼 숫자형 변환
    if debt_col in df_processed.columns:
        df_processed[debt_col] = df_processed[debt_col].astype(str).str.replace(',', '').replace(' - ', np.nan).replace('-', np.nan)
        df_processed[debt_col] = pd.to_numeric(df_processed[debt_col], errors='coerce')
    
    if equity_col in df_processed.columns:
        df_processed[equity_col] = df_processed[equity_col].astype(str).str.replace(',', '').replace(' - ', np.nan).replace('-', np.nan)
        df_processed[equity_col] = pd.to_numeric(df_processed[equity_col], errors='coerce')
    
    if rnd_col in df_processed.columns:
        df_processed[rnd_col] = df_processed[rnd_col].astype(str).str.replace(',', '').replace(' - ', np.nan).replace('-', np.nan)
        df_processed[rnd_col] = pd.to_numeric(df_processed[rnd_col], errors='coerce')
    
    # 기준 연도 설정
    if feature_year is None:
        feature_year = df_processed[year_col].max()
    
    # 업종별 CAPEX 평균 계산 (같은 연도, 같은 field의 평균)
    # feature_year 기준으로 업종 평균 계산
    industry_capex_avg = {}
    if field_col in df_processed.columns and capex_col in df_processed.columns:
        t_minus_1_year = feature_year
        t_minus_1_data = df_processed[df_processed[year_col] == t_minus_1_year].copy()
        
        if not t_minus_1_data.empty:
            # field별, 연도별 CAPEX 평균 계산
            for field_val in t_minus_1_data[field_col].unique():
                if pd.notna(field_val):
                    field_data = t_minus_1_data[
                        (t_minus_1_data[field_col] == field_val) & 
                        (t_minus_1_data[capex_col].notna())
                    ]
                    if len(field_data) > 0:
                        avg_capex = field_data[capex_col].mean()
                        industry_capex_avg[field_val] = avg_capex
    
    # 기업별로 피처 계산
    features_list = []
    
    for company in df_processed[company_col].unique():
        company_data = df_processed[df_processed[company_col] == company].copy()
        company_data = company_data.sort_values(year_col)
        
        # 기준 연도(feature_year)까지의 데이터만 필터링
        # 예: 2019년 말 피처 → 2019년까지의 데이터 사용
        company_data_target = company_data[company_data[year_col] <= feature_year].copy()
        
        if len(company_data_target) < 2:
            continue
        
        # t-1, t-2, t-3년 데이터 추출 (feature_year 기준)
        # 예: 2019년 말 피처 → 2018년(t-1), 2017년(t-2), 2016년(t-3) 데이터 사용
        t_minus_1 = company_data_target[company_data_target[year_col] == feature_year]
        t_minus_2 = company_data_target[company_data_target[year_col] == feature_year - 1]
        t_minus_3 = company_data_target[company_data_target[year_col] == feature_year - 2]
        
        # 피처 초기화 (연도는 feature_year로 저장)
        feature_dict = {
            company_col: company,
            year_col: feature_year
        }
        
        # 1. revenue_t1: t-1년 매출액 (현재 규모)
        if not t_minus_1.empty and pd.notna(t_minus_1[revenue_col].iloc[0]):
            feature_dict['revenue_t1'] = t_minus_1[revenue_col].iloc[0]
        else:
            feature_dict['revenue_t1'] = np.nan
        
        # 2. cagr_2y: (t-1/t-3)^(1/2) - 1 (중기 성장 속도)
        if (not t_minus_1.empty and not t_minus_3.empty and 
            pd.notna(t_minus_1[revenue_col].iloc[0]) and 
            pd.notna(t_minus_3[revenue_col].iloc[0]) and
            t_minus_3[revenue_col].iloc[0] > 0):
            revenue_t1 = t_minus_1[revenue_col].iloc[0]
            revenue_t3 = t_minus_3[revenue_col].iloc[0]
            feature_dict['cagr_2y'] = (revenue_t1 / revenue_t3) ** (1/2) - 1
        else:
            feature_dict['cagr_2y'] = np.nan
        
        # 3. growth_recent: (t-1/t-2) - 1 (최근 모멘텀)
        if (not t_minus_1.empty and not t_minus_2.empty and
            pd.notna(t_minus_1[revenue_col].iloc[0]) and
            pd.notna(t_minus_2[revenue_col].iloc[0]) and
            t_minus_2[revenue_col].iloc[0] > 0):
            revenue_t1 = t_minus_1[revenue_col].iloc[0]
            revenue_t2 = t_minus_2[revenue_col].iloc[0]
            feature_dict['growth_recent'] = (revenue_t1 / revenue_t2) - 1
        else:
            feature_dict['growth_recent'] = np.nan
        
        # 4. growth_acceleration: 최근성장률 - 초기성장률 (성장 가속/감속)
        # 최근성장률: (t-1/t-2) - 1
        # 초기성장률: (t-2/t-3) - 1
        if (not t_minus_1.empty and not t_minus_2.empty and not t_minus_3.empty and
            pd.notna(t_minus_1[revenue_col].iloc[0]) and
            pd.notna(t_minus_2[revenue_col].iloc[0]) and
            pd.notna(t_minus_3[revenue_col].iloc[0]) and
            t_minus_2[revenue_col].iloc[0] > 0 and
            t_minus_3[revenue_col].iloc[0] > 0):
            recent_growth = (t_minus_1[revenue_col].iloc[0] / t_minus_2[revenue_col].iloc[0]) - 1
            initial_growth = (t_minus_2[revenue_col].iloc[0] / t_minus_3[revenue_col].iloc[0]) - 1
            feature_dict['growth_acceleration'] = recent_growth - initial_growth
        else:
            feature_dict['growth_acceleration'] = np.nan
        
        # 5. growth_volatility: std(연도별 성장률) (안정성)
        # 연도별 성장률 계산
        company_data_target = company_data_target.sort_values(year_col)
        company_data_target = company_data_target[company_data_target[revenue_col].notna()]
        
        if len(company_data_target) >= 2:
            growth_rates = []
            for i in range(1, len(company_data_target)):
                prev_revenue = company_data_target.iloc[i-1][revenue_col]
                curr_revenue = company_data_target.iloc[i][revenue_col]
                if prev_revenue > 0 and pd.notna(prev_revenue) and pd.notna(curr_revenue):
                    growth_rate = (curr_revenue / prev_revenue) - 1
                    growth_rates.append(growth_rate)
            
            if len(growth_rates) > 0:
                feature_dict['growth_volatility'] = np.std(growth_rates)
            else:
                feature_dict['growth_volatility'] = np.nan
        else:
            feature_dict['growth_volatility'] = np.nan
        
        # 6. profitable_years: 최근 2년 중 영업이익>0 연도 수 (수익 안정성)
        profitable_count = 0
        if not t_minus_1.empty and pd.notna(t_minus_1[operating_profit_col].iloc[0]):
            if t_minus_1[operating_profit_col].iloc[0] > 0:
                profitable_count += 1
        
        if not t_minus_2.empty and pd.notna(t_minus_2[operating_profit_col].iloc[0]):
            if t_minus_2[operating_profit_col].iloc[0] > 0:
                profitable_count += 1
        
        feature_dict['profitable_years'] = profitable_count
        
        # 7. capex_intensity: CAPEX / 매출액 (당해 투자 강도)
        if (not t_minus_1.empty and 
            pd.notna(t_minus_1[revenue_col].iloc[0]) and 
            pd.notna(t_minus_1[capex_col].iloc[0]) and
            t_minus_1[revenue_col].iloc[0] > 0):
            capex = t_minus_1[capex_col].iloc[0]
            revenue = t_minus_1[revenue_col].iloc[0]
            feature_dict['capex_intensity'] = capex / revenue
        else:
            feature_dict['capex_intensity'] = np.nan
        
        # 8. capex_trend: (t-1 CAPEX / t-2 CAPEX) - 1 (투자 추세)
        if (not t_minus_1.empty and not t_minus_2.empty and
            pd.notna(t_minus_1[capex_col].iloc[0]) and
            pd.notna(t_minus_2[capex_col].iloc[0]) and
            t_minus_2[capex_col].iloc[0] > 0):
            capex_t1 = t_minus_1[capex_col].iloc[0]
            capex_t2 = t_minus_2[capex_col].iloc[0]
            feature_dict['capex_trend'] = (capex_t1 / capex_t2) - 1
        else:
            feature_dict['capex_trend'] = np.nan
        
        # 9. capex_vs_industry: 기업 CAPEX - 업종 평균 (업종 대비 수준)
        if (not t_minus_1.empty and 
            pd.notna(t_minus_1[capex_col].iloc[0])):
            company_capex = t_minus_1[capex_col].iloc[0]
            # 기업의 field 값 가져오기
            company_field = t_minus_1[field_col].iloc[0] if field_col in t_minus_1.columns else None
            
            if company_field in industry_capex_avg:
                industry_avg = industry_capex_avg[company_field]
                feature_dict['capex_vs_industry'] = company_capex - industry_avg
            else:
                feature_dict['capex_vs_industry'] = np.nan
        else:
            feature_dict['capex_vs_industry'] = np.nan
        
        # 10. operating_margin: 영업이익 / 매출액 (영업이익률)
        if (not t_minus_1.empty and
            pd.notna(t_minus_1[revenue_col].iloc[0]) and
            pd.notna(t_minus_1[operating_profit_col].iloc[0]) and
            t_minus_1[revenue_col].iloc[0] > 0):
            operating_profit = t_minus_1[operating_profit_col].iloc[0]
            revenue = t_minus_1[revenue_col].iloc[0]
            feature_dict['operating_margin'] = operating_profit / revenue
        else:
            feature_dict['operating_margin'] = np.nan
        
        # 11. debt_ratio: 부채총계 / 자본총계 (부채비율)
        if (not t_minus_1.empty and
            pd.notna(t_minus_1[debt_col].iloc[0]) and
            pd.notna(t_minus_1[equity_col].iloc[0]) and
            t_minus_1[equity_col].iloc[0] > 0):
            debt = t_minus_1[debt_col].iloc[0]
            equity = t_minus_1[equity_col].iloc[0]
            feature_dict['debt_ratio'] = debt / equity
        else:
            feature_dict['debt_ratio'] = np.nan
        
        # 12. rnd_intensity: 연구개발비 / 매출액 (R&D 집중도)
        if (not t_minus_1.empty and
            pd.notna(t_minus_1[revenue_col].iloc[0]) and
            pd.notna(t_minus_1[rnd_col].iloc[0]) and
            t_minus_1[revenue_col].iloc[0] > 0):
            rnd = t_minus_1[rnd_col].iloc[0]
            revenue = t_minus_1[revenue_col].iloc[0]
            feature_dict['rnd_intensity'] = rnd / revenue
        else:
            feature_dict['rnd_intensity'] = np.nan
        
        features_list.append(feature_dict)
    
    # 결과 데이터프레임 생성
    features_df = pd.DataFrame(features_list)
    
    return features_df


def main():
    """
    메인 실행 함수 - 롤링 패널 방식으로 여러 연도에 대해 피처 생성
    - 2019년 말 피처 → 2020 타겟 (Train)
    - 2020년 말 피처 → 2021 타겟 (Train)
    - 2021년 말 피처 → 2022 타겟 (Train)
    - 2022년 말 피처 → 2023 타겟 (Train)
    - 2023년 말 피처 → 2024 타겟 (Test)
    """
    # 데이터 로드 - 스크립트 위치 기준으로 경로 설정
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    data_path = os.path.join(project_root, 'data', '전체기업_재무데이터_선형보간.csv')
    
    # 여러 인코딩 시도
    encodings = ['utf-8-sig', 'utf-8', 'cp949', 'euc-kr', 'latin1']
    df = None
    used_encoding = None
    
    for enc in encodings:
        try:
            df = pd.read_csv(data_path, encoding=enc)
            used_encoding = enc
            print(f"파일 로드 성공: 인코딩={enc}")
            break
        except FileNotFoundError as e:
            print(f"파일을 찾을 수 없습니다: {data_path}")
            print(f"절대 경로: {os.path.abspath(data_path)}")
            raise
        except Exception as e:
            # 인코딩 오류는 무시하고 다음 인코딩 시도
            if 'UnicodeDecodeError' not in str(type(e).__name__):
                print(f"인코딩 {enc} 시도 중 오류: {type(e).__name__}: {str(e)[:100]}")
            continue
    
    if df is None:
        print(f"\n모든 인코딩 시도 실패. 파일 경로를 확인해주세요: {data_path}")
        print(f"절대 경로: {os.path.abspath(data_path)}")
        raise ValueError("파일을 읽을 수 없습니다. 인코딩을 확인해주세요.")
    
    # 컬럼명 확인 및 출력
    print(f"\n컬럼 목록 (총 {len(df.columns)}개):")
    for i, col in enumerate(df.columns):
        print(f"  [{i}]: {repr(col)}")
    
    # 컬럼 인덱스로 직접 지정 (인코딩 문제 대비)
    # 일반적인 구조: 0=기업명, 5=field, 6=연도, 7=매출액, 8=영업이익, 11=부채총계, 12=자본총계, 13=연구개발비, 14=CAPEX, 17=유형자산_증감
    if len(df.columns) >= 18:
        company_col = df.columns[0]
        field_col = df.columns[5]
        year_col = df.columns[6]
        revenue_col = df.columns[7]
        operating_profit_col = df.columns[8]
        debt_col = df.columns[11]
        equity_col = df.columns[12]
        rnd_col = df.columns[13]
        capex_col = df.columns[14]
        fixed_asset_change_col = df.columns[17]
        print(f"\n컬럼명 자동 지정:")
        print(f"  기업명: {repr(company_col)} (인덱스 0)")
        print(f"  field: {repr(field_col)} (인덱스 5)")
        print(f"  연도: {repr(year_col)} (인덱스 6)")
        print(f"  매출액: {repr(revenue_col)} (인덱스 7)")
        print(f"  영업이익: {repr(operating_profit_col)} (인덱스 8)")
        print(f"  부채총계: {repr(debt_col)} (인덱스 11)")
        print(f"  자본총계: {repr(equity_col)} (인덱스 12)")
        print(f"  연구개발비: {repr(rnd_col)} (인덱스 13)")
        print(f"  CAPEX: {repr(capex_col)} (인덱스 14)")
        print(f"  유형자산_증감: {repr(fixed_asset_change_col)} (인덱스 17)")
    else:
        raise ValueError(f"컬럼 수가 부족합니다. 예상: 18개 이상, 실제: {len(df.columns)}개")
    
    # 롤링 패널: 여러 연도에 대해 피처 생성
    feature_years = [2019, 2020, 2021, 2022, 2023]  # 피처 기준 연도
    all_features = []
    
    print(f"\n{'='*60}")
    print("롤링 패널 피처 생성 시작")
    print(f"{'='*60}")
    
    for feature_year in feature_years:
        target_year = feature_year + 1  # 타겟 연도 = 피처 연도 + 1
        split_type = 'test' if feature_year == 2023 else 'train'
        
        print(f"\n[{feature_year}년 말 피처 → {target_year}년 타겟] ({split_type.upper()})")
        print(f"-" * 60)
        print(f"피처 계산: {feature_year}년까지의 데이터 사용")
        print(f"타겟 예측: {target_year}년 성장률")
        
        # 피처 계산 (feature_year까지의 데이터 사용)
        features_df = calculate_growth_features(
            df, 
            company_col=company_col,
            year_col=year_col,
            revenue_col=revenue_col,
            operating_profit_col=operating_profit_col,
            capex_col=capex_col,
            fixed_asset_change_col=fixed_asset_change_col,
            debt_col=debt_col,
            equity_col=equity_col,
            rnd_col=rnd_col,
            field_col=field_col,
            feature_year=feature_year
        )
        
        # 타겟 연도 및 split 정보는 메모리에서만 사용 (CSV 저장 시 제외)
        # features_df['feature_year'] = feature_year  # 피처 기준 연도
        # features_df['target_year'] = target_year    # 예측 대상 연도
        # features_df['split'] = split_type          # train/test 구분
        
        print(f"  생성된 샘플 수: {len(features_df)}")
        all_features.append(features_df)
    
    # 모든 연도의 피처 합치기
    final_features_df = pd.concat(all_features, ignore_index=True)
    
    # 모델링에 불필요한 컬럼 제거 (feature_year, target_year, split)
    # 이 컬럼들은 메모리에서만 사용하고 CSV에는 저장하지 않음
    columns_to_drop = ['feature_year', 'target_year', 'split']
    for col in columns_to_drop:
        if col in final_features_df.columns:
            final_features_df = final_features_df.drop(columns=[col])
    
    # 결과 저장 - 스크립트 위치 기준으로 경로 설정
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    output_path = os.path.join(project_root, 'data', 'growth_features_rolling_주원.csv')
    final_features_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"\n{'='*60}")
    print("피처 계산 완료")
    print(f"{'='*60}")
    print(f"결과 저장: {output_path}")
    print(f"\n전체 생성된 피처 수: {len(final_features_df)}")
    print(f"\n피처 컬럼 목록:")
    print(list(final_features_df.columns))
    print(f"\n피처 통계:")
    print(final_features_df[['revenue_t1', 'cagr_2y', 'growth_recent', 
                             'growth_acceleration', 'growth_volatility', 
                             'profitable_years']].describe())
    
    return final_features_df


if __name__ == "__main__":
    main()
