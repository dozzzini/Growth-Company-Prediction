"""
회사명 기반 검색 유틸리티
특정 회사의 특허 및 재무 정보를 종합적으로 검색
"""
from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma


def search_by_company(vector_db: Chroma, company_name: str, k: int = 30) -> tuple[List[Document], List[Document]]:
    """
    회사명으로 특허와 재무 정보를 검색합니다.
    
    Args:
        vector_db: ChromaDB 벡터 데이터베이스
        company_name: 회사명
        k: 검색할 문서 수
    
    Returns:
        (특허 문서 리스트, 재무 문서 리스트)
    """
    # 특허 검색
    patent_query = f"{company_name} 특허 기술 발명"
    patent_results = vector_db.similarity_search(patent_query, k=k)
    patent_docs = [
        doc for doc in patent_results
        if doc.metadata.get('type') == 'patent' and company_name in doc.metadata.get('company_name', '')
    ]
    
    # 재무 검색
    financial_query = f"{company_name} 매출 영업이익 재무"
    financial_results = vector_db.similarity_search(financial_query, k=k)
    financial_docs = [
        doc for doc in financial_results
        if doc.metadata.get('type') == 'financial' and company_name in doc.metadata.get('company_name', '')
    ]
    
    return patent_docs, financial_docs


def format_company_context(patent_docs: List[Document], financial_docs: List[Document]) -> str:
    """
    회사의 특허 및 재무 정보를 포맷팅합니다.
    
    Args:
        patent_docs: 특허 문서 리스트
        financial_docs: 재무 문서 리스트
    
    Returns:
        포맷팅된 컨텍스트 문자열
    """
    parts = []
    
    # 통계
    parts.append(f"📊 검색 결과 통계: 특허 {len(patent_docs)}건, 재무 {len(financial_docs)}건")
    parts.append("=" * 70)
    parts.append("")
    
    # 특허 정보 (피인용 높은 순)
    if patent_docs:
        sorted_patents = sorted(
            patent_docs,
            key=lambda doc: doc.metadata.get('citation_count', 0),
            reverse=True
        )
        
        for i, doc in enumerate(sorted_patents[:10], 1):  # 상위 10개
            citation = doc.metadata.get('citation_count', 0)
            if citation >= 10:
                header = f"📌 [중요 특허 #{i}] (피인용 {citation}회 - 매우 높은 기술 영향력)"
            elif citation >= 5:
                header = f"✓ [주요 특허 #{i}] (피인용 {citation}회 - 높은 기술 영향력)"
            else:
                header = f"[특허 #{i}]"
            
            parts.append(header)
            parts.append(doc.page_content)
            parts.append("")
    
    # 재무 정보 (최신순)
    if financial_docs:
        sorted_financial = sorted(
            financial_docs,
            key=lambda doc: doc.metadata.get('application_year', 0),
            reverse=True
        )
        
        for i, doc in enumerate(sorted_financial[:5], 1):  # 최근 5개
            year = doc.metadata.get('application_year', '')
            header = f"[재무정보 #{i}]" + (f" ({year}년)" if year else "")
            parts.append(header)
            parts.append(doc.page_content)
            parts.append("")
    
    return "\n".join(parts)
