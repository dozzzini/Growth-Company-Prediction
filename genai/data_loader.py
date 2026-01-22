"""
CSV 데이터 로더 모듈
특허 정보 및 재무 정보 CSV 파일을 읽고 텍스트로 변환합니다.
"""
import pandas as pd
from typing import List, Dict, Optional
import os
from tqdm import tqdm


def load_patent_data(file_path: str) -> pd.DataFrame:
    """
    통합된 특허 정보 CSV 파일을 로드합니다 (특허정보_final_v2.csv).
    
    Args:
        file_path: CSV 파일 경로
    
    Returns:
        특허 정보 DataFrame
        
    컬럼:
        - company_name: 회사명
        - applicantName_출원인명: 출원인명
        - inventionTitle_발명의명칭: 발명의 명칭
        - applicationNumber_출원번호: 출원번호
        - applicationDate_출원일자: 출원일자
        - 출원년도: 출원 연도
        - registerDate_등록일자: 등록일자
        - 등록년도: 등록 연도
        - registerStatus_등록상태: 등록 상태
        - ipcNumber_IPC코드: IPC 코드 (복수개는 | 로 구분)
        - astrtCont_초록: 특허 초록
        - 피인용횟수: 피인용 횟수 (기술 영향력 지표)
        - indexNo_일련번호: 일련번호
    """
    return pd.read_csv(file_path, encoding='utf-8')


def load_financial_data(file_path: str, encoding: str = 'utf-8') -> pd.DataFrame:
    """
    재무 정보 CSV 파일을 로드합니다.
    인코딩 문제가 있을 경우 cp949나 euc-kr을 시도합니다.
    
    Args:
        file_path: CSV 파일 경로
        encoding: 파일 인코딩 (기본값: utf-8)
    
    Returns:
        재무 정보 DataFrame
    """
    try:
        return pd.read_csv(file_path, encoding=encoding)
    except UnicodeDecodeError:
        # UTF-8로 안 되면 다른 인코딩 시도
        for enc in ['cp949', 'euc-kr', 'latin-1']:
            try:
                return pd.read_csv(file_path, encoding=enc)
            except:
                continue
        raise ValueError(f"파일을 읽을 수 없습니다: {file_path}")


def patent_to_text(row: pd.Series) -> str:
    """
    통합된 특허 정보 DataFrame의 행을 텍스트로 변환합니다.
    
    Args:
        row: 특허 정보 행
    
    Returns:
        포맷팅된 텍스트 문자열
    """
    text_parts = []
    
    # 회사명
    if pd.notna(row.get('company_name')):
        text_parts.append(f"회사명: {row['company_name']}")
    
    # 출원인명
    if pd.notna(row.get('applicantName_출원인명')):
        text_parts.append(f"출원인: {row['applicantName_출원인명']}")
    
    # 발명의 명칭
    if pd.notna(row.get('inventionTitle_발명의명칭')):
        text_parts.append(f"발명명칭: {row['inventionTitle_발명의명칭']}")
    
    # 출원번호
    if pd.notna(row.get('applicationNumber_출원번호')):
        text_parts.append(f"출원번호: {row['applicationNumber_출원번호']}")
    
    # 출원일자 및 연도
    if pd.notna(row.get('applicationDate_출원일자')):
        text_parts.append(f"출원일자: {row['applicationDate_출원일자']}")
    if pd.notna(row.get('출원년도')):
        text_parts.append(f"출원년도: {int(row['출원년도'])}년")
    
    # 등록일자 및 연도
    if pd.notna(row.get('registerDate_등록일자')):
        text_parts.append(f"등록일자: {row['registerDate_등록일자']}")
    if pd.notna(row.get('등록년도')):
        text_parts.append(f"등록년도: {int(row['등록년도'])}년")
    
    # 등록상태
    if pd.notna(row.get('registerStatus_등록상태')):
        text_parts.append(f"등록상태: {row['registerStatus_등록상태']}")
    
    # IPC 코드
    if pd.notna(row.get('ipcNumber_IPC코드')):
        ipc_codes = row['ipcNumber_IPC코드']
        text_parts.append(f"IPC코드: {ipc_codes}")
    
    # 피인용횟수 (기술 영향력 지표) - 강조 표시
    if pd.notna(row.get('피인용횟수')):
        citation_count = int(row['피인용횟수'])
        if citation_count > 0:
            if citation_count >= 10:
                text_parts.append(f"⭐ 피인용횟수: {citation_count}회 (매우 높은 기술 영향력 - 다른 특허에서 {citation_count}번 인용됨)")
            elif citation_count >= 5:
                text_parts.append(f"✓ 피인용횟수: {citation_count}회 (높은 기술 영향력 - 다른 특허에서 {citation_count}번 인용됨)")
            else:
                text_parts.append(f"피인용횟수: {citation_count}회 (다른 특허에서 {citation_count}번 인용됨)")
    
    # 초록
    if pd.notna(row.get('astrtCont_초록')):
        abstract = str(row['astrtCont_초록']).strip()
        if abstract:
            text_parts.append(f"초록: {abstract}")
    
    return "\n".join(text_parts)


def patent_to_metadata(row: pd.Series) -> Dict:
    """
    특허 정보에서 메타데이터를 추출합니다.
    
    Args:
        row: 특허 정보 행
    
    Returns:
        메타데이터 딕셔너리
    """
    metadata = {
        "type": "patent",
        "company_name": str(row.get('company_name', '')),
        "citation_count": int(row.get('피인용횟수', 0)),
        "register_status": str(row.get('registerStatus_등록상태', '')),
        "application_year": int(row.get('출원년도', 0)) if pd.notna(row.get('출원년도')) else 0,
    }
    return metadata


def financial_to_text(row: pd.Series) -> str:
    """
    재무 정보 DataFrame의 행을 텍스트로 변환합니다.
    
    Args:
        row: 재무 정보 행
    
    Returns:
        재무 정보를 설명하는 텍스트
    """
    text_parts = []
    
    # 회사명 찾기 (컬럼명이 다를 수 있음)
    company_cols = [col for col in row.index if '회사명' in col or '기업명' in col or 'company' in col.lower()]
    if company_cols and pd.notna(row.get(company_cols[0])):
        text_parts.append(f"회사명: {row[company_cols[0]]}")
    
    # 연도 찾기
    year_cols = [col for col in row.index if '연도' in col or 'year' in col.lower()]
    if year_cols and pd.notna(row.get(year_cols[0])):
        text_parts.append(f"연도: {int(row[year_cols[0]])}")
    
    # 주요 재무 지표들
    financial_cols = [col for col in row.index if any(keyword in col for keyword in [
        '매출', '영업이익', '당기순이익', '자산', '부채', '자본', 
        'CAPEX', 'R&D', '연구개발'
    ])]
    
    for col in financial_cols:
        if pd.notna(row.get(col)):
            try:
                value = float(row[col])
                # 0이 아닌 값만 표시
                if value != 0:
                    text_parts.append(f"{col}: {value:,.0f}")
            except:
                pass  # 숫자로 변환 불가한 경우 무시
    
    return "\n".join(text_parts)


def load_all_data(
    patent_file: str,
    financial_file: str
) -> Dict[str, List]:
    """
    모든 데이터 파일을 로드하고 텍스트 및 메타데이터로 변환합니다.
    
    Args:
        patent_file: 통합된 특허 정보 CSV 파일 경로 (특허정보_final_v2.csv)
        financial_file: 재무 정보 CSV 파일 경로
    
    Returns:
        데이터 타입별 텍스트와 메타데이터를 담은 딕셔너리
        {
            "patent": {"texts": [...], "metadata": [...]},
            "financial": {"texts": [...], "metadata": [...]}
        }
    """
    print("[1단계] 데이터 파일 로드 중...")
    
    # 통합된 특허 정보 로드
    print(f"  - 특허 정보 로드: {os.path.basename(patent_file)}")
    patent_df = load_patent_data(patent_file)
    print(f"    총 {len(patent_df):,}건의 특허 데이터")
    
    # 특허 정보를 텍스트 및 메타데이터로 변환
    patent_texts = []
    patent_metadata = []
    for _, row in tqdm(patent_df.iterrows(), total=len(patent_df), desc="    특허 정보 변환", unit="건"):
        patent_texts.append(patent_to_text(row))
        patent_metadata.append(patent_to_metadata(row))
    print(f"    ✓ 특허 정보 {len(patent_texts):,}건 로드 완료")
    
    # 재무 정보 로드
    print(f"\n  - 재무 정보 로드: {os.path.basename(financial_file)}")
    financial_df = load_financial_data(financial_file)
    print(f"    총 {len(financial_df):,}건의 재무 데이터")
    
    # 재무 정보를 텍스트 및 메타데이터로 변환
    financial_texts = []
    financial_metadata = []
    for _, row in tqdm(financial_df.iterrows(), total=len(financial_df), desc="    재무 정보 변환", unit="건"):
        # 회사명 추출
        company_cols = [col for col in row.index if '회사명' in col or '기업명' in col or 'company' in col.lower()]
        company_name = str(row[company_cols[0]]) if company_cols and pd.notna(row.get(company_cols[0])) else ""
        
        financial_texts.append(financial_to_text(row))
        financial_metadata.append({
            "type": "financial",
            "company_name": company_name,
            "citation_count": 0,  # 재무 정보는 피인용 없음
            "register_status": "",
            "application_year": int(row.get('연도', 0)) if pd.notna(row.get('연도')) else 0
        })
    print(f"    ✓ 재무 정보 {len(financial_texts):,}건 로드 완료")
    
    return {
        "patent": {"texts": patent_texts, "metadata": patent_metadata},
        "financial": {"texts": financial_texts, "metadata": financial_metadata}
    }


def combine_all_texts(data_dict: Dict[str, Dict]) -> tuple[List[str], List[Dict]]:
    """
    모든 데이터를 하나의 텍스트 및 메타데이터 리스트로 결합합니다.
    
    Args:
        data_dict: 데이터 타입별 텍스트와 메타데이터 딕셔너리
    
    Returns:
        (결합된 텍스트 리스트, 결합된 메타데이터 리스트) 튜플
    """
    all_texts = []
    all_metadata = []
    
    # 각 데이터 타입별로 처리
    for data_type, data in data_dict.items():
        texts = data["texts"]
        metadata_list = data["metadata"]
        
        for text, metadata in zip(texts, metadata_list):
            if text.strip():  # 빈 텍스트 제외
                # 데이터 타입별 접두어 추가
                if data_type == "patent":
                    prefix = "[특허정보] "
                else:  # financial
                    prefix = "[재무정보] "
                
                all_texts.append(prefix + text)
                all_metadata.append(metadata)
    
    return all_texts, all_metadata
