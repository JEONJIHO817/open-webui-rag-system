#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
벡터 스토어 모듈: 문서 임베딩 생성 및 벡터 스토어 구축
배치 처리 적용 + 청크 길이 확인 추가
"""

import os
import argparse
import logging
from tqdm import tqdm
from langchain_community.vectorstores import FAISS
from langchain.schema.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from e5_embeddings import E5Embeddings

# 로깅 설정
logging.getLogger().setLevel(logging.ERROR)

def get_embeddings(model_name="intfloat/multilingual-e5-large-instruct", device="cuda"):
    print(f"[INFO] 임베딩 모델 디바이스: {device}")
    return E5Embeddings(
        model_name=model_name,
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True}
    )

def build_vector_store_batch(documents, embeddings, save_path="faiss_index_pymupdf_81", batch_size=4):
    if not documents:
        raise ValueError("문서가 없습니다. 문서가 올바르게 로드되었는지 확인하세요.")

    texts = [doc.page_content for doc in documents]
    metadatas = [doc.metadata for doc in documents]

    # 청크 길이 출력
    lengths = [len(t) for t in texts]
    print(f"💡 청크 수: {len(texts)}")
    print(f"💡 가장 긴 청크 길이: {max(lengths)} chars")
    print(f"💡 평균 청크 길이: {sum(lengths) // len(lengths)} chars")

    # 배치로 나누기
    batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
    metadata_batches = [metadatas[i:i + batch_size] for i in range(0, len(metadatas), batch_size)]

    print(f"Processing {len(batches)} batches with size {batch_size}")
    print(f"Initializing vector store with batch 1/{len(batches)}")

    # ✅ from_documents 사용
    first_docs = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(batches[0], metadata_batches[0])
    ]
    vectorstore = FAISS.from_documents(first_docs, embeddings)

    for i in tqdm(range(1, len(batches)), desc="Processing batches"):
        try:
            docs_batch = [
                Document(page_content=text, metadata=meta)
                for text, meta in zip(batches[i], metadata_batches[i])
            ]
            vectorstore.add_documents(docs_batch)

            if i % 10 == 0:
                temp_save_path = f"{save_path}_temp"
                os.makedirs(os.path.dirname(temp_save_path) if os.path.dirname(temp_save_path) else '.', exist_ok=True)
                vectorstore.save_local(temp_save_path)
                print(f"Temporary vector store saved to {temp_save_path} after batch {i}")

        except Exception as e:
            print(f"Error processing batch {i}: {e}")
            error_save_path = f"{save_path}_error_at_batch_{i}"
            os.makedirs(os.path.dirname(error_save_path) if os.path.dirname(error_save_path) else '.', exist_ok=True)
            vectorstore.save_local(error_save_path)
            print(f"Partial vector store saved to {error_save_path}")
            raise

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    vectorstore.save_local(save_path)
    print(f"Vector store saved to {save_path}")

    return vectorstore

def load_vector_store(embeddings, load_path="faiss_index_pymupdf_81"):
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"벡터 스토어를 찾을 수 없습니다: {load_path}")
    return FAISS.load_local(load_path, embeddings, allow_dangerous_deserialization=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="벡터 스토어 구축")
    parser.add_argument("--folder", type=str, default="final_dataset", help="문서가 있는 폴더 경로")
    parser.add_argument("--save_path", type=str, default="vector_db", help="벡터 스토어 저장 경로")
    parser.add_argument("--batch_size", type=int, default=4, help="배치 크기")
    parser.add_argument("--model_name", type=str, default="intfloat/multilingual-e5-large-instruct", help="임베딩 모델 이름")
   # parser.add_argument("--device", type=str, default="cuda", help="사용할 디바이스 ('cuda' 또는 'cpu')")
    parser.add_argument("--device", type=str, default="cuda", help="사용할 디바이스 ('cuda' 또는 'cpu' 또는 'cuda:1')")

    args = parser.parse_args()

    # 문서 처리 모듈 import
    from document_processor_image_test import load_documents, split_documents

    documents = load_documents(args.folder)
    chunks = split_documents(documents, chunk_size=800, chunk_overlap=100)

    print(f"[DEBUG] 문서 로딩 및 청크 분할 완료, 임베딩 단계 진입 전")
    print(f"[INFO] 선택된 디바이스: {args.device}")

    try:
        embeddings = get_embeddings(
            model_name=args.model_name,
            device=args.device
        )
        print(f"[DEBUG] 임베딩 모델 생성 완료")
    except Exception as e:
        print(f"[ERROR] 임베딩 모델 생성 중 에러 발생: {e}")
        import traceback; traceback.print_exc()
        exit(1)

    build_vector_store_batch(chunks, embeddings, args.save_path, args.batch_size)

