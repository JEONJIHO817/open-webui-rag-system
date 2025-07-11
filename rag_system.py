import os
import argparse
import sys
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from vector_store import get_embeddings, load_vector_store
from llm_loader import load_llama_model

def create_refine_prompts_with_pages(language="ko"):
    if language == "ko":
        question_prompt = PromptTemplate(
            input_variables=["context_str", "question"],
            template="""
다음은 검색된 문서 조각들입니다:

{context_str}

위 문서들을 참고하여 질문에 답변해주세요. 

**중요한 규칙:**
- 답변 시 참고한 문서가 있다면 해당 정보를 인용하세요
- 문서에 명시된 정보만 사용하고, 추측하지 마세요  
- 페이지 번호나 출처는 위 문서에서 확인된 것만 언급하세요
- 확실하지 않은 정보는 "문서에서 확인되지 않음"이라고 명시하세요

질문: {question}
답변:"""
        )

        refine_prompt = PromptTemplate(
            input_variables=["question", "existing_answer", "context_str"],
            template="""
기존 답변:
{existing_answer}

추가 문서:
{context_str}

기존 답변을 위 추가 문서를 바탕으로 보완하거나 수정해주세요.

**규칙:**
- 새로운 정보가 기존 답변과 다르다면 수정하세요
- 추가 문서에 명시된 정보만 사용하세요
- 하나의 완결된 답변으로 작성하세요
- 확실하지 않은 출처나 페이지는 언급하지 마세요

질문: {question}
답변:"""
        )
    else:
        question_prompt = PromptTemplate(
            input_variables=["context_str", "question"],
            template="""
Here are the retrieved document fragments:

{context_str}

Please answer the question based on the above documents.

**Important rules:**
- Only use information explicitly stated in the documents
- If citing sources, only mention what is clearly indicated in the documents above
- Do not guess or infer page numbers not shown in the context
- If unsure, state "not confirmed in the provided documents"

Question: {question}
Answer:"""
        )

        refine_prompt = PromptTemplate(
            input_variables=["question", "existing_answer", "context_str"],
            template="""
Existing answer:
{existing_answer}

Additional documents:
{context_str}

Refine the existing answer using the additional documents.

**Rules:**
- Only use information explicitly stated in the additional documents
- Create one coherent final answer
- Do not mention uncertain sources or page numbers

Question: {question}
Answer:"""
        )

    return question_prompt, refine_prompt

def build_rag_chain(llm, vectorstore, language="ko", k=7):
    """RAG 체인 구축"""
    question_prompt, refine_prompt = create_refine_prompts_with_pages(language)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="refine",
        retriever=vectorstore.as_retriever(search_kwargs={"k": k}),
        chain_type_kwargs={
            "question_prompt": question_prompt,
            "refine_prompt": refine_prompt
        },
        return_source_documents=True
    )

    return qa_chain

def ask_question_with_pages(qa_chain, question):
    """질문 처리"""
    result = qa_chain.invoke({"query": question})

    # 결과에서 A: 이후 문장만 추출
    answer = result['result']
    final_answer = answer.split("A:")[-1].strip() if "A:" in answer else answer.strip()

    print(f"\n🧾 질문: {question}")
    print(f"\n🟢 최종 답변: {final_answer}")

    # 메타데이터 디버깅 정보 출력 (비활성화)
    # debug_metadata_info(result["source_documents"])

    # 참고 문서를 페이지별로 정리
    print("\n📚 참고 문서 요약:")
    source_info = {}
    
    for doc in result["source_documents"]:
        source = doc.metadata.get('source', 'N/A')
        page = doc.metadata.get('page', 'N/A')
        doc_type = doc.metadata.get('type', 'N/A')
        section = doc.metadata.get('section', None)
        total_pages = doc.metadata.get('total_pages', None)
        
        filename = doc.metadata.get('filename', 'N/A')
        if filename == 'N/A':
            filename = os.path.basename(source) if source != 'N/A' else 'N/A'
        
        if filename not in source_info:
            source_info[filename] = {
                'pages': set(), 
                'sections': set(),
                'types': set(),
                'total_pages': total_pages
            }
        
        if page != 'N/A':
            if isinstance(page, str) and page.startswith('섹션'):
                source_info[filename]['sections'].add(page)
            else:
                source_info[filename]['pages'].add(page)
        
        if section is not None:
            source_info[filename]['sections'].add(f"섹션 {section}")
        
        source_info[filename]['types'].add(doc_type)

    # 결과 출력
    total_chunks = len(result["source_documents"])
    print(f"총 사용된 청크 수: {total_chunks}")
    
    for filename, info in source_info.items():
        print(f"\n- {filename}")
        
        # 전체 페이지 수 정보
        if info['total_pages']:
            print(f"  전체 페이지 수: {info['total_pages']}")
        
        # 페이지 정보 출력
        if info['pages']:
            pages_list = list(info['pages'])
            print(f"  페이지: {', '.join(map(str, pages_list))}")
        
        # 섹션 정보 출력  
        if info['sections']:
            sections_list = sorted(list(info['sections']))
            print(f"  섹션: {', '.join(sections_list)}")
        
        # 페이지와 섹션이 모두 없는 경우
        if not info['pages'] and not info['sections']:
            print(f"  페이지: 정보 없음")
            
        # 문서 유형 출력
        types_str = ', '.join(sorted(info['types']))
        print(f"  유형: {types_str}")

    return result

# 기존 ask_question 함수는 ask_question_with_pages로 교체
def ask_question(qa_chain, question):
    """호환성을 위한 래퍼 함수"""
    return ask_question_with_pages(qa_chain, question)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG refine system (페이지 번호 지원)")
    parser.add_argument("--vector_store", type=str, default="vector_db", help="벡터 스토어 경로")
    parser.add_argument("--model", type=str, default="LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct", help="LLM 모델 ID")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="사용할 디바이스")
    parser.add_argument("--k", type=int, default=7, help="검색할 문서 수")
    parser.add_argument("--language", type=str, default="ko", choices=["ko", "en"], help="사용할 언어")
    parser.add_argument("--query", type=str, help="질문 (없으면 대화형 모드 실행)")

    args = parser.parse_args()

    embeddings = get_embeddings(device=args.device)
    vectorstore = load_vector_store(embeddings, load_path=args.vector_store)
    llm = load_llama_model()

    qa_chain = build_rag_chain(llm, vectorstore, language=args.language, k=args.k)

    print("🟢 RAG 페이지 번호 지원 시스템 준비 완료!")

    if args.query:
        ask_question_with_pages(qa_chain, args.query)
    else:
        print("💬 대화형 모드 시작 (종료하려면 'exit', 'quit', '종료' 입력)")
        while True:
            try:
                query = input("\n질문: ").strip()
                if query.lower() in ["exit", "quit", "종료"]:
                    break
                if query:  # 빈 입력 방지
                    ask_question_with_pages(qa_chain, query)
            except KeyboardInterrupt:
                print("\n\n프로그램을 종료합니다.")
                break
            except Exception as e:
                print(f"❗ 오류 발생: {e}\n다시 시도해주세요.")
