"""
LLM 프롬프트 모듈
RAG 체인에서 사용할 프롬프트 템플릿을 정의합니다.
"""
from langchain_core.prompts import PromptTemplate
from typing import Optional, Dict, Any


def create_reviewer_prompt(
    template: Optional[str] = None
) -> PromptTemplate:
    """
    기술금융 심사역 페르소나를 위한 프롬프트 템플릿을 생성합니다.
    
    Args:
        template: 사용자 정의 템플릿 (None이면 기본 템플릿 사용)
    
    Returns:
        PromptTemplate 인스턴스
    """
    if template is None:
        template = """
당신은 은행 본점 기술금융부의 전문 심사역입니다.
제공된 [입력 데이터]만을 바탕으로 기업 분석 보고서를 작성하십시오.

[작성 원칙]
1. 환각 방지: 데이터에 없는 사실을 지어내지 말 것. 근거가 없으면 '정보 없음'으로 표기.
2. 근거 표기: 모든 분석 문장 뒤에 반드시 (근거: 출처)를 명시할 것.
3. 논리적 연결: 모델 예측값, 재무지표, 특허정보를 연결하여 종합적으로 분석할 것.
4. 객관성 유지: 감정적 표현을 배제하고 수치와 사실에 기반하여 기술할 것.
5. 특허 우선 분석: 기술적 강점 분석 시 특허 정보를 최우선으로 검토하고, 특히 피인용횟수가 높은 특허(⭐ 10회 이상, ✓ 5회 이상)를 중점 분석할 것.

[검색된 데이터]
{context}

[분석 대상 기업 정보]
{question}

[출력 양식]
## 1. 성장 가능성 요약 (2~3문장)
- 핵심 내용 요약
- 주요 긍정/부정 요인

## 2. 기술적 강점 분석 (특허 중심)
### 2.1 핵심 특허 기술력
- 피인용횟수 높은 특허 분석 (⭐ 10회 이상, ✓ 5회 이상 특허 우선 검토)
- 특허 등록 현황 및 권리 안정성 (등록/공개/거절 상태)
- 주요 특허의 출원번호 및 등록일자 명시

### 2.2 기술 분야 및 경쟁력 (IPC 코드 기반)
- 주요 IPC 코드 분석 (어떤 기술 분야에 집중하는지)
- 기술의 차별성 및 독창성 평가
- 기술 트렌드와의 부합성

### 2.3 특허 포트폴리오 강도
- 특허 건수 및 출원 추이
- 최근 특허 출원 활동성 (출원년도 기준)
- 기술 영향력 지표 (피인용횟수 분석)

### 2.4 R&D 투자와의 연계
- 연구개발 투자 대비 특허 성과
- 기술 사업화 가능성 평가

## 3. 재무적 안정성
- 수익성 지표 분석
- 성장성 지표 분석
- 재무 건전성

## 4. 여신 심사 시 유의사항 (리스크)
- 주요 리스크 요인
- 모니터링 필요 사항

---
보고서 작성:
"""
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
    
    return prompt


def generate_tech_finance_prompt(company_data: Dict[str, Any]) -> tuple[str, str]:
    """
    기업 데이터를 바탕으로 기술금융 심사 보고서 프롬프트를 생성합니다.
    
    Args:
        company_data: 기업 분석 데이터
            - overview: 기업 개요
            - prediction: 성장 예측값 및 확률
            - shap_features: SHAP Top 변수 (feature, impact, value)
            - financial_metrics: 주요 재무지표
            - patent_info: 특허 정보 (IPC 코드, 특허 수 등)
            - news_list: 최근 뉴스 (선택사항)
    
    Returns:
        (system_instruction, user_context) 튜플
    """
    # 1. 시스템 페르소나 및 가이드라인 정의
    system_instruction = """
당신은 은행 본점 기술금융부의 전문 심사역입니다.
제공된 [입력 데이터]만을 바탕으로 기업 분석 보고서를 작성하십시오.

[작성 원칙]
1. 환각 방지: 데이터에 없는 사실을 지어내지 말 것. 근거가 없으면 '정보 없음'으로 표기.
2. 근거 표기: 모든 분석 문장 뒤에 반드시 (근거: 피처명, 특허번호, 출원번호 등)를 명시할 것.
3. 논리적 연결: SHAP 수치(기여도)와 실제 기술 내용(IPC/재무지표/특허)을 연결하여 기술할 것.
4. 객관성 유지: 감정적 표현을 배제하고 수치와 사실에 기반하여 기술할 것.
"""
    
    # 2. SHAP 특성 포맷팅
    shap_text = ""
    if 'shap_features' in company_data:
        shap_text = "\n".join([
            f"   - {feat['feature']}: {feat['impact']} 영향 (기여도: {feat['value']:.3f})"
            for feat in company_data['shap_features']
        ])
    
    # 3. 재무지표 포맷팅
    financial_text = ""
    if 'financial_metrics' in company_data:
        financial_text = "\n".join([
            f"   - {k}: {v}" 
            for k, v in company_data['financial_metrics'].items()
        ])
    
    # 4. 특허 정보 포맷팅
    patent_text = ""
    if 'patent_info' in company_data:
        if 'ipc_codes' in company_data['patent_info']:
            patent_text = "\n".join([
                f"   - {ipc['code']}: {ipc['desc']}"
                for ipc in company_data['patent_info']['ipc_codes']
            ])
        if 'patent_count' in company_data['patent_info']:
            patent_text = f"   총 특허 수: {company_data['patent_info']['patent_count']}건\n" + patent_text
    
    # 5. 뉴스 정보 포맷팅 (있는 경우)
    news_text = "   정보 없음"
    if 'news_list' in company_data and company_data['news_list']:
        news_text = "\n".join([
            f"   [{news['id']}] {news['title']}"
            for news in company_data['news_list']
        ])
    
    # 6. 입력 데이터 포맷팅
    user_context = f"""
[입력 데이터]

1. 기업 개요
   {company_data.get('overview', '정보 없음')}

2. 성장 예측 결과
   {company_data.get('prediction', '정보 없음')}

3. 모델 기여도 분석 (SHAP Top 5)
{shap_text if shap_text else '   정보 없음'}

4. 주요 재무지표
{financial_text if financial_text else '   정보 없음'}

5. 특허 기술력
{patent_text if patent_text else '   정보 없음'}

6. 최근 뉴스/이슈
{news_text}

---

[출력 양식]

## 1. 성장 가능성 요약 (2~3문장)
   - 핵심 내용 및 예측 결과 요약
   - 주요 긍정/부정 요인

## 2. 기술적 강점 분석 (특허 중심)
### 2.1 핵심 특허 기술력
   - 피인용횟수 높은 특허 분석 (⭐ 10회 이상, ✓ 5회 이상 특허 우선 검토)
   - 특허 등록 현황 및 권리 안정성 (등록/공개/거절 상태)
   - 주요 특허의 출원번호 및 등록일자 명시

### 2.2 기술 분야 및 경쟁력 (IPC 코드 기반)
   - 주요 IPC 코드 분석 (어떤 기술 분야에 집중하는지)
   - 기술의 차별성 및 독창성 평가
   - 기술 트렌드와의 부합성

### 2.3 특허 포트폴리오 강도
   - 특허 건수 및 출원 추이
   - 최근 특허 출원 활동성 (출원년도 기준)
   - 기술 영향력 지표 (피인용횟수 분석)

### 2.4 R&D 투자와의 연계
   - 연구개발 투자 대비 특허 성과
   - 기술 사업화 가능성 평가

## 3. 재무적 안정성
   - 수익성 지표 분석
   - 성장성 지표 분석  
   - 재무 건전성 평가

## 4. 여신 심사 시 유의사항 (리스크)
   - 주요 리스크 요인 식별
   - 모니터링 필요 사항
   - 대출 심사 시 고려사항
"""
    
    return system_instruction, user_context


def create_custom_prompt(
    template: str,
    input_variables: list = None
) -> PromptTemplate:
    """
    사용자 정의 프롬프트 템플릿을 생성합니다.
    
    Args:
        template: 프롬프트 템플릿 문자열
        input_variables: 템플릿에서 사용할 변수 리스트 (None이면 자동 감지)
    
    Returns:
        PromptTemplate 인스턴스
    """
    if input_variables is None:
        prompt = PromptTemplate.from_template(template)
    else:
        prompt = PromptTemplate(
            template=template,
            input_variables=input_variables
        )
    
    return prompt
