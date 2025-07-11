import os
from langchain.schema.document import Document
from e5_embeddings import E5Embeddings
from langchain_community.vectorstores import FAISS

from document_processor_image import load_documents, split_documents  # 반드시 이 함수가 필요

# 경로 설정
NEW_FOLDER = "25.05.28 RAG용 2차 업무편람 취합본"
#NEW_FOLDER = "임시"
VECTOR_STORE_PATH = "faiss_index_800image"

# 1. 임베딩 모델 로딩
def get_embeddings(model_name="intfloat/multilingual-e5-large-instruct", device="cuda"):
    return E5Embeddings(
        model_name=model_name,
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True}
    )

# 2. 기존 벡터 스토어 로드
def load_vector_store(embeddings, load_path="faiss_index_800image"):
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"벡터 스토어를 찾을 수 없습니다: {load_path}")
    return FAISS.load_local(load_path, embeddings, allow_dangerous_deserialization=True)

# 3. 문서 임베딩 및 추가
def add_new_documents_to_vector_store(new_folder, vectorstore, embeddings):
    print(f"📂 새로운 문서 로드 중: {new_folder}")
    new_docs = load_documents(new_folder)
    new_chunks = split_documents(new_docs, chunk_size=800, chunk_overlap=100)

    print(f"📄 새로운 청크 수: {len(new_chunks)}")
    print(f"추가 전 벡터 수: {vectorstore.index.ntotal}")
    vectorstore.add_documents(new_chunks)
    print(f"추가 후 벡터 수: {vectorstore.index.ntotal}")

    print("✅ 새로운 문서가 벡터 스토어에 추가되었습니다.")

# 4. 전체 실행
if __name__ == "__main__":
    embeddings = get_embeddings()
    vectorstore = load_vector_store(embeddings, VECTOR_STORE_PATH)
    add_new_documents_to_vector_store(NEW_FOLDER, vectorstore, embeddings)
    vectorstore.save_local(VECTOR_STORE_PATH)
    print(f"💾 벡터 스토어 저장 완료: {VECTOR_STORE_PATH}")
