import os
import glob
from langchain.schema.document import Document
from e5_embeddings import E5Embeddings
from langchain_community.vectorstores import FAISS
from document_processor import load_pdf_with_pymupdf, split_documents

# 경로 설정
FOLDER = "25.05.28 RAG용 2차 업무편람 취합본"
VECTOR_STORE_PATH = "vector_db"

# 1. 임베딩 모델 로드
def get_embeddings(model_name="intfloat/multilingual-e5-large-instruct", device="cuda"):
    return E5Embeddings(
        model_name=model_name,
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True}
    )

# 2. 기존 벡터 스토어 로드
def load_vector_store(embeddings, load_path=VECTOR_STORE_PATH):
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"벡터 스토어를 찾을 수 없습니다: {load_path}")
    return FAISS.load_local(load_path, embeddings, allow_dangerous_deserialization=True)

# 3. 정리된 PDF만 임베딩
def embed_cleaned_pdfs(folder, vectorstore, embeddings):
    pattern = os.path.join(folder, "정리된*.pdf")
    pdf_files = glob.glob(pattern)
    print(f"🧾 대상 PDF 수: {len(pdf_files)}")

    new_documents = []
    for pdf_path in pdf_files:
        print(f"📄 처리 중: {pdf_path}")
        text = load_pdf_with_pymupdf(pdf_path)
        if text.strip():
            new_documents.append(Document(page_content=text, metadata={"source": pdf_path}))

    print(f"📚 문서 수: {len(new_documents)}")

    chunks = split_documents(new_documents, chunk_size=300, chunk_overlap=50)
    print(f"�� 청크 수: {len(chunks)}")

    print(f"추가 전 벡터 수: {vectorstore.index.ntotal}")
    vectorstore.add_documents(chunks)
    print(f"추가 후 벡터 수: {vectorstore.index.ntotal}")

    vectorstore.save_local(VECTOR_STORE_PATH)
    print(f"✅ 저장 완료: {VECTOR_STORE_PATH}")

# 실행
if __name__ == "__main__":
    embeddings = get_embeddings()
    vectorstore = load_vector_store(embeddings)
    embed_cleaned_pdfs(FOLDER, vectorstore, embeddings)
