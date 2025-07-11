from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from rag_system import build_rag_chain, ask_question
from vector_store import get_embeddings, load_vector_store
from llm_loader import load_llama_model
import uuid
import os
import shutil
from urllib.parse import urljoin, quote

from fastapi.responses import StreamingResponse
import json
import time

app = FastAPI()

# 정적 파일 서빙을 위한 설정
os.makedirs("static/documents", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# 전역 객체 준비
embeddings = get_embeddings(device="cpu")
vectorstore = load_vector_store(embeddings, load_path="vector_db")
llm = load_llama_model()
qa_chain = build_rag_chain(llm, vectorstore, language="ko", k=7)

# 서버 URL 설정 (실제 환경에 맞게 수정 필요)
BASE_URL = "http://220.124.155.35:8500"

class Question(BaseModel):
    question: str

def get_document_url(source_path):
    if not source_path or source_path == 'N/A':
        return None
    filename = os.path.basename(source_path)
    dataset_root = os.path.join(os.getcwd(), "dataset")
    # dataset 전체 하위 폴더에서 파일명 일치하는 파일 찾기
    found_path = None
    for root, dirs, files in os.walk(dataset_root):
        if filename in files:
            found_path = os.path.join(root, filename)
            break
    if not found_path or not os.path.exists(found_path):
        return None
    static_path = f"static/documents/{filename}"
    shutil.copy2(found_path, static_path)
    encoded_filename = quote(filename)
    return urljoin(BASE_URL, f"/static/documents/{encoded_filename}")

def create_download_link(url, filename):
    return f'출처: [{filename}]({url})'

@app.post("/ask")
def ask(question: Question):
    result = ask_question(qa_chain, question.question)
    
    # 소스 문서 정보 처리
    sources = []
    for doc in result["source_documents"]:
        source_path = doc.metadata.get('source', 'N/A')
        document_url = get_document_url(source_path) if source_path != 'N/A' else None
        
        source_info = {
            "source": source_path,
            "content": doc.page_content,
            "page": doc.metadata.get('page', 'N/A'),
            "document_url": document_url,
            "filename": os.path.basename(source_path) if source_path != 'N/A' else None
        }
        sources.append(source_info)
    
    return {
        "answer": result['result'].split("A:")[-1].strip() if "A:" in result['result'] else result['result'].strip(),
        "sources": sources
    }

@app.get("/v1/models")
def list_models():
    return JSONResponse({
        "object": "list",
        "data": [
            {
                "id": "rag",
                "object": "model",
                "owned_by": "local",
            }
        ]
    })

@app.post("/v1/chat/completions")
async def openai_compatible_chat(request: Request):
    payload = await request.json()
    messages = payload.get("messages", [])
    user_input = messages[-1]["content"] if messages else ""
    stream = payload.get("stream", False)

    result = ask_question(qa_chain, user_input)
    answer = result['result']
    
    # 소스 문서 정보 처리
    sources = []
    for doc in result["source_documents"]:
        source_path = doc.metadata.get('source', 'N/A')
        document_url = get_document_url(source_path) if source_path != 'N/A' else None
        filename = os.path.basename(source_path) if source_path != 'N/A' else None
        
        source_info = {
            "source": source_path,
            "content": doc.page_content,
            "page": doc.metadata.get('page', 'N/A'),
            "document_url": document_url,
            "filename": filename
        }
        sources.append(source_info)

    # 소스 정보를 한 줄씩만 출력
    sources_md = "\n참고 문서:\n"
    seen = set()
    for source in sources:
        key = (source['filename'], source['document_url'])
        if source['document_url'] and source['filename'] and key not in seen:
            sources_md += f"출처: [{source['filename']}]({source['document_url']})\n"
            seen.add(key)

    final_answer = answer.split("A:")[-1].strip() if "A:" in answer else answer.strip()
    final_answer += sources_md

    if not stream:
        return JSONResponse({
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": final_answer
                },
                "finish_reason": "stop"
            }],
            "model": "rag",
        })

    # 스트리밍 응답을 위한 generator
    def event_stream():
        # 답변 본문만 먼저 스트리밍
        answer_main = answer.split("A:")[-1].strip() if "A:" in answer else answer.strip()
        for char in answer_main:
            chunk = {
                "id": f"chatcmpl-{uuid.uuid4()}",
                "object": "chat.completion.chunk",
                "choices": [{
                    "index": 0,
                    "delta": {
                        "content": char
                    },
                    "finish_reason": None
                }]
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            time.sleep(0.005)
        # 참고 문서(다운로드 링크)는 마지막에 한 번에 붙여서 전송
        sources_md = "\n참고 문서:\n"
        seen = set()
        for source in sources:
            key = (source['filename'], source['document_url'])
            if source['document_url'] and source['filename'] and key not in seen:
                sources_md += f"출처: [{source['filename']}]({source['document_url']})\n"
                seen.add(key)
        if sources_md.strip() != "참고 문서:":
            chunk = {
                "id": f"chatcmpl-{uuid.uuid4()}",
                "object": "chat.completion.chunk",
                "choices": [{
                    "index": 0,
                    "delta": {
                        "content": sources_md
                    },
                    "finish_reason": None
                }]
            }
            yield f"data: {json.dumps(chunk)}\n\n"
        done = {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion.chunk",
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }]
        }
        yield f"data: {json.dumps(done)}\n\n"
        return

    return StreamingResponse(event_stream(), media_type="text/event-stream")
