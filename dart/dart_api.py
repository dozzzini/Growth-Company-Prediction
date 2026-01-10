import requests
import pandas as pd
import json
import os
import time
from tqdm import tqdm

class DartCollector:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://opendart.fss.or.kr/api"

    def get_company_info(self, corp_code):
        """기업정보: 설립일, 업종코드, 주소"""
        url = f"{self.base_url}/company.json"
        params = {'crtfc_key': self.api_key, 'corp_code': corp_code}
        res = requests.get(url, params=params).json()
        if res.get('status') == '000':
            return {
                '설립일': res.get('est_dt'),
                '업종코드': res.get('induty_code'),
                '주소': res.get('adr')
            }
        return {}

    def get_financial_data(self, corp_code, year):
        """재무제표 & CAPEX: 매출, 영업이익, 자산, 부채, 자본, 유형자산취득, 연구개발비"""
        url = f"{self.base_url}/fnlttSinglAcntAll.json"
        params = {
            'crtfc_key': self.api_key,
            'corp_code': corp_code,
            'bsns_year': str(year),
            'reprt_code': '11011',  # 사업보고서(결산)
            'fs_div': 'CFS'         # 연결재무제표 기준
        }
        res = requests.get(url, params=params).json()
        
        extracted = {}
        cis_data = {}
        tangible_asset_current = None  # 유형자산 당기
        tangible_asset_previous = None  # 유형자산 전기
        
        if res.get('status') == '000':
            for item in res.get('list', []):
                name = item.get('account_nm')
                val = item.get('thstrm_amount')  # 당기 금액
                prev_val = item.get('frmtrm_amount')  # 전기 금액
                sj_div = item.get('sj_div')  # 재무제표 구분
                
                # 재무상태표(BS)에서만 자산, 부채, 자본 추출
                if sj_div == 'BS':
                    if name == '자산총계' or name == '자 산 총 계':
                        extracted['자산'] = val
                    elif name == '자본총계' or name == '자 본 총 계':
                        extracted['자본'] = val
                    elif name == '부채총계' or name == '부 채 총 계':
                        extracted['부채'] = val
                    elif name == '유형자산':
                        # 유형자산 당기와 전기 값 저장
                        tangible_asset_current = val
                        tangible_asset_previous = prev_val
                
                # 손익계산서(IS)에서 영업이익, 매출액, 연구개발비, 당기순이익 추출
                elif sj_div == 'IS':
                    if name == '영업이익' or name == '영업이익(손실)':
                        extracted['영업이익'] = val
                    elif name == '매출액' or name == '수익' or name == '수익(매출액)':
                        extracted['매출액'] = val
                    elif name == '당기순이익' or name == '당기순이익(손실)' or name == '순이익' or name == '순이익(손실)':
                        extracted['당기순이익'] = val
                    elif '연구개발비' in name or '연구개발비용' in name or 'R&D' in name.upper():
                        # 연구개발비는 여러 항목이 있을 수 있으므로 합산
                        if '연구개발비' not in extracted:
                            extracted['연구개발비'] = 0
                        try:
                            extracted['연구개발비'] += int(val) if val else 0
                        except (ValueError, TypeError):
                            pass

                # 포괄손익계산서(CIS) 데이터 수집 (IS에 없을 경우 대비)
                elif sj_div == 'CIS':
                    if name == '수익(매출액)' or name == '매출액' or name == '수익':
                        cis_data['매출액'] = val
                    elif name == '영업이익(손실)' or name == '영업이익':
                        cis_data['영업이익'] = val
                    elif name == '당기순이익' or name == '당기순이익(손실)' or name == '순이익' or name == '순이익(손실)':
                        cis_data['당기순이익'] = val
                
                # 현금흐름표(CF)에서 CAPEX 추출
                elif sj_div == 'CF':
                    if '유형자산의 취득' in name:
                        extracted['CAPEX'] = val
        
        # IS에서 값을 찾지 못한 경우 CIS 값으로 대체
        if '매출액' not in extracted and '매출액' in cis_data:
            extracted['매출액'] = cis_data['매출액']
        
        if '영업이익' not in extracted and '영업이익' in cis_data:
            extracted['영업이익'] = cis_data['영업이익']
        
        if '당기순이익' not in extracted and '당기순이익' in cis_data:
            extracted['당기순이익'] = cis_data['당기순이익']

        # 연구개발비가 없으면 0으로 설정
        if '연구개발비' not in extracted:
            extracted['연구개발비'] = 0
        
        # CAPEX가 없고 유형자산 값이 있으면 계산
        if 'CAPEX' not in extracted or not extracted.get('CAPEX'):
            if tangible_asset_current and tangible_asset_previous:
                try:
                    current = int(tangible_asset_current) if tangible_asset_current else 0
                    previous = int(tangible_asset_previous) if tangible_asset_previous else 0
                    extracted['CAPEX'] = current - previous
                    # 유형자산 당기와 전기 값도 저장
                    extracted['유형자산_당기'] = tangible_asset_current
                    extracted['유형자산_전기'] = tangible_asset_previous
                except (ValueError, TypeError):
                    extracted['CAPEX'] = None
                    extracted['유형자산_당기'] = tangible_asset_current
                    extracted['유형자산_전기'] = tangible_asset_previous
            else:
                # 유형자산 값이 없어도 컬럼은 추가
                extracted['유형자산_당기'] = tangible_asset_current if tangible_asset_current else None
                extracted['유형자산_전기'] = tangible_asset_previous if tangible_asset_previous else None
        else:
            # CAPEX가 있으면 유형자산 값은 None으로 설정
            extracted['유형자산_당기'] = None
            extracted['유형자산_전기'] = None

        return extracted

    def get_employee_status(self, corp_code, year):
        """직원현황: 종업원 합계, 정규직 수, 계약직 수"""
        url = f"{self.base_url}/empSttus.json"
        params = {
            'crtfc_key': self.api_key,
            'corp_code': corp_code,
            'bsns_year': str(year),
            'reprt_code': '11011'
        }
        res = requests.get(url, params=params).json()
        
        total_emp = 0  # 종업원 합계 (sm)
        total_regular = 0  # 정규직 수 (rgllbr_co)
        total_contract = 0  # 계약직 수 (cnttk_co)
        
        if res.get('status') == '000':
            for item in res.get('list', []):
                # 종업원 합계 (sm)
                sm = item.get('sm', '').replace(',', '').replace('-', '')
                if sm and sm.isdigit():
                    total_emp += int(sm)
                
                # 정규직 수 (rgllbr_co)
                rgllbr = item.get('rgllbr_co', '').replace(',', '').replace('-', '')
                if rgllbr and rgllbr.isdigit():
                    total_regular += int(rgllbr)
                
                # 계약직 수 (cnttk_co)
                cnttk = item.get('cnttk_co', '').replace(',', '').replace('-', '')
                if cnttk and cnttk.isdigit():
                    total_contract += int(cnttk)
        
        return {
            '종업원_합계': total_emp,
            '정규직_수': total_regular,
            '계약직_수': total_contract
        }
    
    def collect_all_companies_data(self, csv_path, year=2023, output_json='data/기업정보_전체.json'):
        """
        CSV 파일의 dart_corp_code 목록을 읽어서 모든 기업의 정보를 수집하고 JSON으로 저장
        
        Args:
            csv_path: CSV 파일 경로
            year: 조회할 연도
            output_json: 출력 JSON 파일 경로
        """
        # 스크립트 디렉토리 기준으로 경로 해결
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        csv_file = os.path.join(script_dir, csv_path)
        json_file = os.path.join(script_dir, output_json)
        
        print(f"CSV 파일 읽는 중: {csv_file}")
        
        # CSV 파일 읽기
        try:
            df = pd.read_csv(csv_file, encoding='utf-8-sig')
        except UnicodeDecodeError:
            df = pd.read_csv(csv_file, encoding='cp949')
        
        # dart_corp_code가 있는 행만 필터링
        df = df[df['dart_corp_code'].notna() & (df['dart_corp_code'] != '')].copy()
        
        # dart_corp_code를 문자열로 변환하고 앞의 0 유지
        df['dart_corp_code'] = df['dart_corp_code'].astype(str).apply(
            lambda x: x.split('.')[0].zfill(8) if x and x != 'nan' else ''
        )
        
        total_companies = len(df)
        print(f"총 {total_companies}개 기업의 데이터를 수집합니다.")
        print(f"연도: {year}")
        print(f"출력 파일: {json_file}\n")
        
        results = []
        success_count = 0
        fail_count = 0
        
        # tqdm으로 진행률 표시
        for index, row in tqdm(df.iterrows(), total=total_companies, desc=f"{year}년 데이터 수집", unit="기업"):
            corp_code = row['dart_corp_code']
            company_name = row['기업명']
            
            try:
                # 기업 정보 수집
                company_data = self.get_company_info(corp_code)
                finance_data = self.get_financial_data(corp_code, year)
                employee_data = self.get_employee_status(corp_code, year)
                
                # 결과 통합
                company_result = {
                    '기업명': company_name,
                    'dart_corp_code': corp_code,
                    'crno': str(row['crno']) if pd.notna(row['crno']) else '',
                    '연도': year,
                    **company_data,
                    **finance_data,
                    **employee_data
                }
                
                results.append(company_result)
                success_count += 1
                
            except Exception as e:
                fail_count += 1
                
                # 실패한 경우에도 기본 정보는 저장
                results.append({
                    '기업명': company_name,
                    'dart_corp_code': corp_code,
                    'crno': str(row['crno']) if pd.notna(row['crno']) else '',
                    '연도': year,
                    '에러': str(e)[:100]
                })
            
            # API 제한을 피하기 위해 대기
            time.sleep(0.1)
        
        # # JSON 파일로 저장
        # print(f"\n결과 저장 중: {json_file}")
        # with open(json_file, 'w', encoding='utf-8') as f:
        #     json.dump(results, f, ensure_ascii=False, indent=2)
        
        # CSV 파일로도 저장
        csv_file = json_file.replace('.json', '.csv')
        print(f"CSV 파일 저장 중: {csv_file}")
        results_df = pd.DataFrame(results)
        results_df.to_csv(csv_file, index=False, encoding='cp949')
        
        print(f"\n===== 수집 완료 =====")
        print(f"총 기업 수: {total_companies}개")
        print(f"성공: {success_count}개")
        print(f"실패: {fail_count}개")
        print(f"결과 파일: {csv_file}")

# --- 실행부 ---
if __name__ == "__main__":
    # .env 파일에서 API 키 읽기
    # 프로젝트 루트 디렉토리 찾기 (dart_api.py는 Growth-Company-Prediction/dart/ 폴더에 있음)
    # 상위 디렉토리로 2단계 올라가면 프로젝트 루트 (sesac_project/)
    script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env_path = os.path.join(script_dir, '.env')
    MY_KEY = None
    
    if os.path.exists(env_path):
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # 주석이나 빈 줄 무시
                if line and not line.startswith('#'):
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        if key == 'DART_API_KEY':
                            MY_KEY = value
                            break
    
    if not MY_KEY:
        print("오류: .env 파일에서 DART_API_KEY를 찾을 수 없습니다.")
        print(f".env 파일 경로 확인: {env_path}")
        print(".env 파일에 다음 형식으로 추가해주세요: DART_API_KEY=your_api_key")
        exit(1)
    
    CSV_FILE = 'data/특허_기업명리스트.csv'
    
    collector = DartCollector(MY_KEY)
    
    # 2019년부터 2024년까지 각 연도별로 데이터 수집
    for year in range(2019, 2025):
        print(f"\n{'='*60}")
        print(f"  {year}년도 데이터 수집 시작")
        print(f"{'='*60}\n")
        
        output_json = f'data/기업정보_{year}.json'
        collector.collect_all_companies_data(CSV_FILE, year, output_json)
        
        print(f"\n{year}년도 데이터 수집 완료!\n")
    
    print(f"\n{'='*60}")
    print("모든 연도 데이터 수집이 완료되었습니다!")
    print(f"{'='*60}")
    # collector.collect_all_companies_data(CSV_FILE, 2023, 'data/기업정보_2023.csv')