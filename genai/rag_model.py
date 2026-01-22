"""
RAG 모델 실행 예제
CSV 데이터를 로드하고 ChromaDB를 사용하여 RAG 시스템을 구축합니다.
"""
import os
import sys
from dotenv import load_dotenv
from tqdm import tqdm

# 환경 변수 로드 (.env 파일에서)
load_dotenv()

# OpenAI API 키 확인
if not os.getenv("OPENAI_API_KEY"):
    print("="*70)
    print("경고: OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
    print("="*70)
    print("\n다음 중 하나의 방법으로 API 키를 설정하세요:")
    print("1. 환경 변수 설정:")
    print("   export OPENAI_API_KEY='your-api-key-here'")
    print("\n2. .env 파일 생성 (프로젝트 루트에):")
    print("   OPENAI_API_KEY=your-api-key-here")
    print("="*70)
    sys.exit(1)

# 직접 실행 시 상대 import 오류 방지
try:
    from .data_loader import load_all_data, combine_all_texts
    from .chunking import chunk_text
    from .embedding import create_vector_db
    from .prompt import create_reviewer_prompt
    from .rag_chain import create_rag_chain, run_rag_chain, create_llm
except ImportError:
    # 직접 실행 시 (python rag_model.py)
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from genai.data_loader import load_all_data, combine_all_texts
    from genai.chunking import chunk_text
    from genai.embedding import create_vector_db
    from genai.prompt import create_reviewer_prompt
    from genai.rag_chain import create_rag_chain, run_rag_chain, create_llm


def main(sample_size=None, force_recreate=False):
    """
    메인 실행 함수
    
    Args:
        sample_size: 테스트용 샘플 크기 (None이면 전체 데이터 사용)
                    예: sample_size=1000 -> 처음 1000개 문서만 사용
        force_recreate: True면 기존 벡터 DB를 삭제하고 새로 생성
    """
    # 데이터 파일 경로 설정
    base_path = os.path.join(os.path.dirname(__file__), "..", "data")
    patent_file = os.path.join(base_path, "특허정보_final_v2.csv")
    financial_file = os.path.join(base_path, "재무정보_final_imputed.csv")
    
    # 1. CSV 데이터 로드 및 텍스트 변환
    print("="*70)
    print("[1단계] CSV 데이터 로드 및 텍스트 변환")
    print("="*70)
    data_dict = load_all_data(patent_file, financial_file)
    
    # 모든 텍스트 및 메타데이터 결합
    all_texts, all_metadata = combine_all_texts(data_dict)
    print(f"\n  총 {len(all_texts)}개의 문서 생성 완료")
    print(f"  - 특허 정보: {sum(1 for m in all_metadata if m['type'] == 'patent')}건")
    print(f"  - 재무 정보: {sum(1 for m in all_metadata if m['type'] == 'financial')}건")
    print(f"  - 피인용 5회 이상 특허: {sum(1 for m in all_metadata if m.get('citation_count', 0) >= 5)}건")
    
    # 샘플 크기 적용 (테스트용)
    if sample_size:
        print(f"\n  ⚠️  테스트 모드: {sample_size}개 문서만 사용합니다")
        all_texts = all_texts[:sample_size]
        all_metadata = all_metadata[:sample_size]
    
    # 2. 텍스트 청킹
    print("\n" + "="*70)
    print("[2단계] 텍스트 청킹")
    print("="*70)
    chunk_size = 1000
    chunk_overlap = 100
    
    # 각 텍스트를 청킹
    chunked_texts = []
    for text in tqdm(all_texts, desc="  텍스트 청킹 진행", unit="문서"):
        chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunked_texts.extend(chunks)
    
    print(f"  청크 크기: {chunk_size}, 겹침: {chunk_overlap}")
    print(f"  생성된 청크 수: {len(chunked_texts)}개")
    
    # 3. ChromaDB 벡터 DB 생성 또는 로드
    print("\n" + "="*70)
    print("[3단계] ChromaDB 벡터 DB 생성/로드")
    print("="*70)
    persist_directory = os.path.join(os.path.dirname(__file__), "chroma_db")
    
    # 기존 DB 존재 확인
    db_exists = os.path.exists(persist_directory) and os.path.exists(
        os.path.join(persist_directory, "chroma.sqlite3")
    )
    
    if db_exists and not force_recreate:
        print(f"  ✓ 기존 벡터 DB 발견!")
        print(f"  ✓ 임베딩 생성을 스킵하고 기존 DB를 사용합니다.")
        print(f"  💡 새로 생성하려면: python rag_model.py --recreate")
    
    # 청킹된 텍스트에 대응하는 메타데이터 생성 (청킹으로 인해 확장)
    chunked_metadata = []
    text_idx = 0
    for text in all_texts:
        chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        # 같은 원본 텍스트의 청크는 같은 메타데이터 사용
        for _ in chunks:
            if text_idx < len(all_metadata):
                chunked_metadata.append(all_metadata[text_idx])
        text_idx += 1
    
    print(f"  ✓ 청킹 완료: {len(chunked_metadata)}개 메타데이터 생성")
    
    vector_db = create_vector_db(
        texts=chunked_texts,
        persist_directory=persist_directory,
        collection_name="company_data",
        force_recreate=force_recreate,
        metadata_list=chunked_metadata
    )
    
    if db_exists and not force_recreate:
        print(f"  ✓ 벡터 DB 로드 완료: {persist_directory}")
    else:
        print(f"  ✓ 벡터 DB 생성 완료: {persist_directory}")
        print(f"  ✓ 저장된 문서 수: {len(chunked_texts)}개")
    
    # 4. LLM 및 프롬프트 설정
    print("\n" + "="*70)
    print("[4단계] LLM 및 프롬프트 설정")
    print("="*70)
    llm = create_llm(model_name="gpt-4o", temperature=0)
    prompt = create_reviewer_prompt()
    print("  LLM 모델: gpt-4o")
    print("  프롬프트: 기술금융 심사역 페르소나")
    
    # 5. RAG 체인 구축
    print("\n" + "="*70)
    print("[5단계] RAG 체인 구축")
    print("="*70)
    qa_chain = create_rag_chain(
        vector_db, 
        llm=llm, 
        prompt=prompt,
        search_kwargs={"k": 5}  # 상위 5개 문서 검색
    )
    print("  RAG 체인 구축 완료")
    
    # 6. 실행 예제 (XGBoost + RAG 통합)
    print("\n" + "="*70)
    print("[6단계] XGBoost + RAG 통합 분석")
    print("="*70)
    
    company_name = "비츠로셀"
    year = 2023
    use_xgboost = True  # XGBoost 통합 사용 여부
    
    print(f"\n분석 대상 회사: {company_name} ({year}년)")
    
    try:
        from .company_search import search_by_company, format_company_context
        from .xgb_integration import analyze_company_with_xgb_and_rag
    except ImportError:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from genai.company_search import search_by_company, format_company_context
        from genai.xgb_integration import analyze_company_with_xgb_and_rag
    
    # XGBoost 통합 사용 시
    if use_xgboost:
        try:
            print("\n[XGBoost 모델 예측 수행]")
            xgb_result, shap_contributions, question = analyze_company_with_xgb_and_rag(
                company_name=company_name,
                year=year,
                top_n_features=10
            )
            
            if xgb_result is None:
                print("⚠️  XGBoost 예측 실패, 기본 RAG 모드로 전환합니다.")
                use_xgboost = False
        except Exception as e:
            print(f"⚠️  XGBoost 통합 중 오류 발생: {e}")
            print("⚠️  기본 RAG 모드로 전환합니다.")
            use_xgboost = False
    
    # 기본 RAG 모드 (XGBoost 없이)
    if not use_xgboost:
        question = f"{company_name}의 기술적 강점과 재무 안정성을 종합적으로 분석하여 성장 가능성을 평가해주세요."
    
    # 특허 + 재무 검색
    patent_docs, financial_docs = search_by_company(vector_db, company_name, k=50)
    print(f"\n검색 결과: 특허 {len(patent_docs)}건, 재무 {len(financial_docs)}건\n")
    
    # 컨텍스트 생성
    context = format_company_context(patent_docs, financial_docs)
    
    print(f"질문:\n{question}\n")
    
    # 프롬프트 직접 생성
    prompt_template = prompt
    full_prompt = prompt_template.format(context=context, question=question)
    
    print("="*70)
    print("생성된 보고서:")
    print("="*70)
    report = llm.invoke(full_prompt).content
    print(report)
    print("="*70)
    
    # XGBoost 결과 요약 출력 (사용 시)
    if use_xgboost and xgb_result:
        print("\n" + "="*70)
        print("[XGBoost 모델 예측 요약]")
        print("="*70)
        print(f"  기업명: {xgb_result['company_name']}")
        print(f"  성장 확률: {xgb_result['growth_probability']:.2%}")
        print(f"  예측: {'✅ 상위 30% 성장 예상' if xgb_result['predicted_growth'] == 1 else '⚠️  하위 70%'}")
        if shap_contributions:
            print(f"\n  주요 기여 피처 (SHAP Top 5):")
            for i, contrib in enumerate(shap_contributions[:5], 1):
                impact_icon = "📈" if contrib['impact'] == "긍정적" else "📉"
                print(f"    {i}. {impact_icon} {contrib['feature']}: {contrib['impact']} (SHAP: {contrib['shap_value']:+.4f})")
        print("="*70)


if __name__ == "__main__":
    import sys
    
    # 커맨드 라인 인자 파싱
    # 예: python rag_model.py 1000  -> 1000개만 처리
    # 예: python rag_model.py --recreate  -> 기존 DB 삭제하고 새로 생성
    # 예: python rag_model.py 1000 --recreate  -> 1000개로 재생성
    sample_size = None
    force_recreate = False
    
    for arg in sys.argv[1:]:
        if arg == "--recreate":
            force_recreate = True
            print("⚠️  벡터 DB 재생성 모드")
        else:
            try:
                sample_size = int(arg)
                print(f"샘플 크기: {sample_size}개 문서")
            except ValueError:
                print(f"알 수 없는 인자: {arg}")
    
    main(sample_size=sample_size, force_recreate=force_recreate)