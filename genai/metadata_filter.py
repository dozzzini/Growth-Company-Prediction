"""
메타데이터 기반 검색 필터링 유틸리티
특허 정보 우선순위 및 피인용횟수 기반 정렬 기능 제공
"""
from typing import List, Dict, Any
from langchain_core.documents import Document


def rerank_by_citation(documents: List[Document]) -> List[Document]:
    """
    검색된 문서를 피인용횟수 기반으로 재정렬합니다.
    특허 정보를 우선적으로 배치하고, 피인용횟수가 높은 순서로 정렬합니다.
    
    Args:
        documents: 검색된 문서 리스트
    
    Returns:
        재정렬된 문서 리스트
    """
    # 특허와 재무 정보 분리
    patent_docs = []
    financial_docs = []
    
    for doc in documents:
        metadata = doc.metadata if hasattr(doc, 'metadata') else {}
        if metadata.get('type') == 'patent':
            patent_docs.append(doc)
        else:
            financial_docs.append(doc)
    
    # 특허 문서를 피인용횟수 기준으로 내림차순 정렬
    patent_docs.sort(
        key=lambda doc: doc.metadata.get('citation_count', 0) if hasattr(doc, 'metadata') else 0,
        reverse=True
    )
    
    # 특허 우선, 그 다음 재무 정보
    return patent_docs + financial_docs


def get_high_citation_patents(documents: List[Document], min_citations: int = 5) -> List[Document]:
    """
    피인용횟수가 높은 특허만 필터링합니다.
    
    Args:
        documents: 문서 리스트
        min_citations: 최소 피인용횟수
    
    Returns:
        필터링된 특허 문서 리스트
    """
    return [
        doc for doc in documents
        if hasattr(doc, 'metadata') 
        and doc.metadata.get('type') == 'patent'
        and doc.metadata.get('citation_count', 0) >= min_citations
    ]


def format_docs_with_priority(docs: List[Document]) -> str:
    """
    문서를 우선순위에 따라 포맷팅합니다.
    피인용횟수가 높은 특허를 강조 표시합니다.
    
    Args:
        docs: 문서 리스트
    
    Returns:
        포맷팅된 텍스트
    """
    # 재정렬
    reranked_docs = rerank_by_citation(docs)
    
    formatted_parts = []
    patent_count = 0
    financial_count = 0
    
    for i, doc in enumerate(reranked_docs, 1):
        metadata = doc.metadata if hasattr(doc, 'metadata') else {}
        doc_type = metadata.get('type', 'unknown')
        
        if doc_type == 'patent':
            patent_count += 1
            citation_count = metadata.get('citation_count', 0)
            
            if citation_count >= 10:
                header = f"📌 [중요 특허 #{patent_count}] (피인용 {citation_count}회 - 매우 높은 기술 영향력)"
            elif citation_count >= 5:
                header = f"✓ [주요 특허 #{patent_count}] (피인용 {citation_count}회 - 높은 기술 영향력)"
            else:
                header = f"[특허 #{patent_count}]"
                
            formatted_parts.append(f"{header}\n{doc.page_content}")
        else:
            financial_count += 1
            formatted_parts.append(f"[재무정보 #{financial_count}]\n{doc.page_content}")
    
    # 상단에 통계 추가
    stats = f"📊 검색 결과 통계: 특허 {patent_count}건, 재무 {financial_count}건\n" + "="*70 + "\n\n"
    
    return stats + "\n\n".join(formatted_parts)


def create_metadata_aware_retriever(vector_db, search_kwargs: Dict[str, Any] = None):
    """
    메타데이터 기반 검색을 수행하는 retriever를 생성합니다.
    
    Args:
        vector_db: ChromaDB 벡터 데이터베이스
        search_kwargs: 검색 파라미터
    
    Returns:
        설정된 retriever
    """
    if search_kwargs is None:
        search_kwargs = {}
    
    # 기본 검색 설정
    default_kwargs = {
        "search_type": "mmr",  # Maximal Marginal Relevance (다양성 확보)
        "search_kwargs": {
            "k": 10,  # 최종 반환 개수
            "fetch_k": 50,  # 초기 가져올 개수 (필터링 전)
            "lambda_mult": 0.7  # 다양성 vs 관련성 균형 (0: 다양성, 1: 관련성)
        }
    }
    
    # 사용자 설정 병합
    default_kwargs.update(search_kwargs)
    
    return vector_db.as_retriever(**default_kwargs)
