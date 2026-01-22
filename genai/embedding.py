"""
임베딩 및 벡터 DB 모듈
텍스트를 벡터로 변환하고 ChromaDB 벡터 데이터베이스를 생성하는 기능을 제공합니다.
"""
import warnings
# ChromaDB의 pydantic v1 경고 억제
warnings.filterwarnings("ignore", category=UserWarning, module="chromadb")

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from typing import List, Optional, Dict
import os
import shutil
import time
from dotenv import load_dotenv
from tqdm import tqdm

# 환경 변수 로드
load_dotenv()


def create_embeddings(model: str = "text-embedding-3-small", api_key: Optional[str] = None) -> OpenAIEmbeddings:
    """
    OpenAI 임베딩 모델을 생성합니다.
    
    Args:
        model: 사용할 임베딩 모델명 (기본값: text-embedding-3-small)
        api_key: OpenAI API 키 (None이면 환경 변수에서 읽음)
    
    Returns:
        OpenAIEmbeddings 인스턴스
    """
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError(
            "OpenAI API 키가 설정되지 않았습니다. "
            "환경 변수 OPENAI_API_KEY를 설정하거나 api_key 파라미터를 제공하세요."
        )
    
    return OpenAIEmbeddings(model=model, openai_api_key=api_key)


def create_vector_db(
    texts: List[str],
    embeddings: Optional[OpenAIEmbeddings] = None,
    persist_directory: Optional[str] = None,
    collection_name: str = "rag_collection",
    force_recreate: bool = False,
    metadata_list: Optional[List[Dict]] = None
) -> Chroma:
    """
    텍스트 리스트로부터 ChromaDB 벡터 데이터베이스를 생성합니다.
    
    Args:
        texts: 임베딩할 텍스트 리스트
        embeddings: 사용할 임베딩 모델 (None이면 기본 모델 사용)
        persist_directory: 벡터 DB를 저장할 디렉토리 경로 (None이면 메모리에만 저장)
        collection_name: ChromaDB 컬렉션 이름
        force_recreate: True면 기존 DB를 삭제하고 새로 생성
    
    Returns:
        Chroma 벡터 데이터베이스 인스턴스
    """
    # 기존 벡터 DB 확인
    if persist_directory and os.path.exists(persist_directory) and not force_recreate:
        print(f"  ✓ 기존 벡터 DB 발견: {persist_directory}")
        print(f"  ✓ 기존 벡터 DB를 로드합니다 (임베딩 생성 스킵)")
        return load_vector_db(persist_directory, embeddings, collection_name)
    
    if force_recreate and persist_directory and os.path.exists(persist_directory):
        print(f"  ⚠️  기존 벡터 DB 삭제 중...")
        import shutil
        shutil.rmtree(persist_directory)
        print(f"  ✓ 기존 벡터 DB 삭제 완료")
    if embeddings is None:
        embeddings = create_embeddings()
    
    # 텍스트를 Document 객체로 변환 (메타데이터 포함)
    print(f"  Document 객체 변환 중...")
    if metadata_list and len(metadata_list) == len(texts):
        documents = [
            Document(page_content=text, metadata=metadata)
            for text, metadata in tqdm(zip(texts, metadata_list), total=len(texts), desc="  텍스트 변환", unit="문서")
        ]
        print(f"  ✓ 메타데이터 포함 변환 완료")
    else:
        documents = [Document(page_content=text) for text in tqdm(texts, desc="  텍스트 변환", unit="문서")]
    
    # ChromaDB 생성 (배치 처리로 진행 표시)
    print(f"  임베딩 생성 및 ChromaDB 저장 중...")
    print(f"  ⚠️  총 {len(documents)}개 문서 임베딩 - 예상 시간: {len(documents)//100} ~ {len(documents)//50}초")
    print(f"  ⚠️  OpenAI API 비용 예상: ${len(documents)*0.00002:.2f} ~ ${len(documents)*0.00004:.2f}")
    
    # 배치 크기 설정 (너무 크면 메모리 문제, 너무 작으면 느림)
    batch_size = 500
    
    if persist_directory:
        # 첫 배치로 초기화
        first_batch = documents[:batch_size]
        vector_db = Chroma.from_documents(
            documents=first_batch,
            embedding=embeddings,
            persist_directory=persist_directory,
            collection_name=collection_name
        )
        
        # 나머지 배치 처리 (Rate Limit 방지를 위해 딜레이 추가)
        for i in tqdm(range(batch_size, len(documents), batch_size), 
                     desc="  임베딩 진행", 
                     unit="배치"):
            batch = documents[i:i+batch_size]
            
            # Rate Limit 방지를 위한 재시도 로직
            max_retries = 3
            for retry in range(max_retries):
                try:
                    vector_db.add_documents(batch)
                    # 성공 시 다음 배치 전 짧은 딜레이 (Rate Limit 방지)
                    time.sleep(1)
                    break
                except Exception as e:
                    if "rate_limit" in str(e).lower() or "429" in str(e):
                        wait_time = 5 * (retry + 1)  # 5초, 10초, 15초
                        print(f"\n  ⚠️  Rate Limit 도달. {wait_time}초 대기 중...")
                        time.sleep(wait_time)
                        if retry == max_retries - 1:
                            raise
                    else:
                        raise
            
        # 명시적으로 저장
        vector_db.persist()
    else:
        # 메모리만 사용하는 경우
        first_batch = documents[:batch_size]
        vector_db = Chroma.from_documents(
            documents=first_batch,
            embedding=embeddings,
            collection_name=collection_name
        )
        
        for i in tqdm(range(batch_size, len(documents), batch_size), 
                     desc="  임베딩 진행", 
                     unit="배치"):
            batch = documents[i:i+batch_size]
            
            # Rate Limit 방지를 위한 재시도 로직
            max_retries = 3
            for retry in range(max_retries):
                try:
                    vector_db.add_documents(batch)
                    # 성공 시 다음 배치 전 짧은 딜레이
                    time.sleep(1)
                    break
                except Exception as e:
                    if "rate_limit" in str(e).lower() or "429" in str(e):
                        wait_time = 5 * (retry + 1)
                        print(f"\n  ⚠️  Rate Limit 도달. {wait_time}초 대기 중...")
                        time.sleep(wait_time)
                        if retry == max_retries - 1:
                            raise
                    else:
                        raise
    
    return vector_db


def load_vector_db(
    persist_directory: str,
    embeddings: Optional[OpenAIEmbeddings] = None,
    collection_name: str = "rag_collection"
) -> Chroma:
    """
    저장된 ChromaDB 벡터 데이터베이스를 로드합니다.
    
    Args:
        persist_directory: 벡터 DB가 저장된 디렉토리 경로
        embeddings: 사용할 임베딩 모델 (None이면 기본 모델 사용)
        collection_name: ChromaDB 컬렉션 이름
    
    Returns:
        Chroma 벡터 데이터베이스 인스턴스
    """
    if embeddings is None:
        embeddings = create_embeddings()
    
    vector_db = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name=collection_name
    )
    
    return vector_db
