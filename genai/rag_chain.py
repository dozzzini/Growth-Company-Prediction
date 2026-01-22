"""
RAG 체인 구성 모듈
LangChain Expression Language (LCEL)를 사용하여 RAG 체인을 구축합니다.
"""
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from typing import Optional, Dict, Any
from .embedding import create_embeddings
from .prompt import create_reviewer_prompt
from .metadata_filter import format_docs_with_priority
import os
from dotenv import load_dotenv

load_dotenv()


def format_docs(docs) -> str:
    """
    문서 리스트를 하나의 문자열로 포맷팅
    (기본 포맷팅 - 우선순위 없이 단순 결합)
    """
    return "\n\n".join(doc.page_content for doc in docs)


def create_rag_chain(
    vector_db: Chroma,
    llm: Optional[ChatOpenAI] = None,
    prompt: Optional[PromptTemplate] = None,
    search_kwargs: Optional[Dict[str, Any]] = None,
    use_metadata_filter: bool = True
):
    """
    RAG 체인을 생성합니다.
    
    Args:
        vector_db: 벡터 데이터베이스
        llm: 언어 모델 (None이면 기본 모델 사용)
        prompt: 프롬프트 템플릿 (None이면 기본 템플릿 사용)
        search_kwargs: 검색 파라미터
        use_metadata_filter: 메타데이터 기반 필터링 및 재정렬 사용 여부
    
    Returns:
        RAG 체인
    """
    if llm is None:
        llm = create_llm()
    
    if prompt is None:
        prompt = create_reviewer_prompt()
    
    # Retriever 설정
    retriever_kwargs = {}
    if search_kwargs:
        retriever_kwargs["search_kwargs"] = search_kwargs
    
    # 메타데이터 필터링 사용 시 검색 파라미터 설정
    if use_metadata_filter:
        # similarity search 사용 (MMR은 재무정보를 제외하는 경향이 있음)
        if "search_kwargs" not in retriever_kwargs:
            retriever_kwargs["search_kwargs"] = {}
        # 더 많은 문서를 가져와서 특허와 재무를 모두 포함
        if "k" not in retriever_kwargs["search_kwargs"]:
            retriever_kwargs["search_kwargs"]["k"] = 20  # 15 → 20으로 증가
    
    retriever = vector_db.as_retriever(**retriever_kwargs)
    
    # 문서 포맷터 선택
    doc_formatter = format_docs_with_priority if use_metadata_filter else format_docs
    
    # RAG 체인 구축 (LCEL 방식)
    qa_chain = (
        {"context": retriever | doc_formatter, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return qa_chain


def run_rag_chain(
    qa_chain,
    question: str,
    input_dict: Optional[Dict[str, Any]] = None
) -> str:
    """
    RAG 체인을 실행합니다.
    
    Args:
        qa_chain: RAG 체인
        question: 질문 문자열
        input_dict: 추가 입력 딕셔너리 (선택사항)
    
    Returns:
        RAG 체인의 응답
    """
    if input_dict is None:
        result = qa_chain.invoke(question)
    else:
        result = qa_chain.invoke(input_dict)
    
    return result


def create_llm(
    model_name: str = "gpt-4o",
    temperature: float = 0,
    api_key: Optional[str] = None,
    **kwargs
) -> ChatOpenAI:
    """
    ChatOpenAI 인스턴스를 생성합니다.
    
    Args:
        model_name: 사용할 모델 이름 (기본값: gpt-4o)
        temperature: 생성 온도 (0: 결정적, 1: 창의적)
        api_key: OpenAI API 키 (None이면 환경변수에서 읽기)
        **kwargs: ChatOpenAI의 추가 파라미터
    
    Returns:
        ChatOpenAI 인스턴스
    """
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError(
            "OpenAI API 키가 설정되지 않았습니다. "
            "환경 변수 OPENAI_API_KEY를 설정하거나 api_key 파라미터를 제공하세요."
        )
    
    return ChatOpenAI(
        model_name=model_name,
        temperature=temperature,
        openai_api_key=api_key,
        **kwargs
    )
