"""
텍스트 청킹 모듈
문서를 의미 있는 단위로 분할하는 기능을 제공합니다.
"""
from langchain_text_splitters import CharacterTextSplitter
from typing import List


def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    separator: str = "\n\n"
) -> List[str]:
    """
    텍스트를 청크로 분할합니다.
    
    Args:
        text: 분할할 원본 텍스트
        chunk_size: 각 청크의 최대 크기
        chunk_overlap: 청크 간 겹치는 문자 수
        separator: 텍스트 분할 시 사용할 구분자
    
    Returns:
        분할된 텍스트 청크 리스트
    """
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator=separator
    )
    texts = text_splitter.split_text(text)
    return texts


def chunk_file(
    file_path: str,
    encoding: str = "utf-8",
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> List[str]:
    """
    파일을 읽어서 청크로 분할합니다.
    
    Args:
        file_path: 읽을 파일 경로
        encoding: 파일 인코딩 (기본값: utf-8)
        chunk_size: 각 청크의 최대 크기
        chunk_overlap: 청크 간 겹치는 문자 수
    
    Returns:
        분할된 텍스트 청크 리스트
    """
    with open(file_path, "r", encoding=encoding) as f:
        raw_text = f.read()
    
    return chunk_text(raw_text, chunk_size, chunk_overlap)
