"""
RAG 시스템 사용 예제
각 모듈을 개별적으로 사용하는 방법을 보여줍니다.
"""
import os
from .chunking import chunk_file, chunk_text
from .embedding import create_vector_db, create_embeddings, load_vector_db
from .prompt import create_reviewer_prompt, create_custom_prompt
from .rag_chain import create_rag_chain, run_rag_chain, create_llm


def example_basic_usage():
    """기본 사용 예제"""
    print("="*70)
    print("기본 사용 예제")
    print("="*70)
    
    # 1. 파일에서 텍스트 청킹
    print("\n[1] 텍스트 청킹")
    texts = chunk_file("company_news.txt", chunk_size=500, chunk_overlap=50)
    print(f"   생성된 청크 수: {len(texts)}개")
    
    # 2. 벡터 DB 생성
    print("\n[2] 벡터 DB 생성")
    vector_db = create_vector_db(texts)
    print("   벡터 DB 생성 완료")
    
    # 3. 프롬프트 생성
    print("\n[3] 프롬프트 생성")
    prompt = create_reviewer_prompt()
    print("   프롬프트 생성 완료")
    
    # 4. LLM 생성
    print("\n[4] LLM 생성")
    llm = create_llm(model_name="gpt-4o", temperature=0)
    print("   LLM 생성 완료")
    
    # 5. RAG 체인 생성 및 실행
    print("\n[5] RAG 체인 생성 및 실행")
    qa_chain = create_rag_chain(vector_db, llm=llm, prompt=prompt)
    
    question = "상위 30% 성장 확률 82%, 주요 긍정 인자: R&D 투자 효율성(rnd_intensity)"
    answer = run_rag_chain(qa_chain, question)
    
    print("\n질문:", question)
    print("\n답변:")
    print(answer)


def example_custom_prompt():
    """커스텀 프롬프트 사용 예제"""
    print("\n" + "="*70)
    print("커스텀 프롬프트 사용 예제")
    print("="*70)
    
    # 커스텀 프롬프트 생성
    custom_template = """
당신은 전문 분석가입니다.
다음 정보를 바탕으로 간결한 분석을 제공하세요.

컨텍스트: {context}
질문: {question}

분석:
"""
    prompt = create_custom_prompt(custom_template)
    print("커스텀 프롬프트 생성 완료")


def example_save_load_vector_db():
    """벡터 DB 저장 및 로드 예제"""
    print("\n" + "="*70)
    print("벡터 DB 저장 및 로드 예제")
    print("="*70)
    
    # 텍스트 청킹
    texts = chunk_file("company_news.txt")
    
    # 벡터 DB 생성 및 저장
    print("\n[1] 벡터 DB 생성 및 저장")
    vector_db = create_vector_db(texts, save_path="./vector_db")
    print("   벡터 DB 저장 완료: ./vector_db")
    
    # 벡터 DB 로드
    print("\n[2] 벡터 DB 로드")
    loaded_db = load_vector_db("./vector_db")
    print("   벡터 DB 로드 완료")


def example_different_llm():
    """다른 LLM 모델 사용 예제"""
    print("\n" + "="*70)
    print("다른 LLM 모델 사용 예제")
    print("="*70)
    
    # GPT-3.5 사용
    llm_gpt35 = create_llm(model_name="gpt-3.5-turbo", temperature=0.7)
    print("GPT-3.5-turbo 모델 생성 완료")
    
    # 다른 temperature 설정
    llm_creative = create_llm(model_name="gpt-4o", temperature=0.9)
    print("창의적 응답을 위한 높은 temperature 설정 완료")


if __name__ == "__main__":
    # 기본 사용 예제 실행
    example_basic_usage()
    
    # 다른 예제들 (주석 해제하여 실행)
    # example_custom_prompt()
    # example_save_load_vector_db()
    # example_different_llm()
