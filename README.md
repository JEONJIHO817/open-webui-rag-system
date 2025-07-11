# Open WebUI RAG System

Open WebUI와 연동 가능한 한국어 문서 기반 RAG(Retrieval-Augmented Generation) 시스템입니다. PDF와 HWPX 파일을 지원하며, 페이지별 정확한 정보 추출과 출처 추적이 가능합니다.

## 주요 기능

### 1. 문서 처리
- **PDF 문서**: PyMuPDF 기반 텍스트, 표, 이미지 OCR 추출
- **HWPX 문서**: XML 파싱을 통한 섹션별 텍스트, 표, 이미지 추출
- **페이지별 처리**: 각 문서를 페이지/섹션 단위로 정확하게 분리
- **다중 콘텐츠 타입**: 본문, 표, OCR 텍스트를 각각 식별하여 처리

### 2. 벡터 검색
- **E5-Large 임베딩**: 다국어 지원 고성능 임베딩 모델
- **FAISS 벡터스토어**: 빠른 유사도 검색
- **배치 처리**: 대용량 문서 처리 최적화
- **청크 분할**: 문맥 유지를 위한 겹침 처리

### 3. RAG 시스템
- **Refine 체인**: 다중 문서 참조를 통한 정확한 답변 생성
- **출처 추적**: 페이지 번호와 문서명을 포함한 정확한 인용
- **Hallucination 방지**: 문서에 명시된 정보만 사용하는 엄격한 프롬프트

### 4. API 서버
- **FastAPI 기반**: 비동기 처리 지원
- **OpenAI 호환**: `/v1/chat/completions` 엔드포인트 제공
- **스트리밍 지원**: 실시간 답변 생성
- **Open WebUI 연동**: 플러그인 없이 바로 연결 가능

## 시스템 요구사항

### 하드웨어
- **GPU**: CUDA 지원 (임베딩 및 LLM 추론용)
- **RAM**: 최소 16GB (대용량 문서 처리 시 더 필요)
- **저장공간**: 모델 및 벡터스토어용 10GB+

### 소프트웨어
- Python 3.8+
- CUDA 11.7+ (GPU 사용 시)
- Tesseract OCR

## 설치 방법

### 1. 저장소 클론
```bash
git clone <repository-url>
cd open-webui-rag-system
```

### 2. 의존성 설치
```bash
pip install -r requirements.txt
```

### 3. Tesseract OCR 설치
**Ubuntu/Debian:**
```bash
sudo apt-get install tesseract-ocr tesseract-ocr-kor
```

**Windows:**
- [Tesseract 공식 페이지](https://github.com/UB-Mannheim/tesseract/wiki)에서 설치

### 4. LLM 서버 설정
`llm_loader.py`에서 사용할 LLM 서버 설정:
```python
# EXAONE 모델 사용 예시
base_url="http://vllm:8000/v1"
model="LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
openai_api_key="token-abc123"
```

## 실행 방법

### 1. 문서 준비
처리할 문서들을 `dataset_test` 폴더에 저장:
```
dataset_test/
├── document1.pdf
├── document2.hwpx
└── document3.pdf
```

### 2. 문서 처리 및 벡터스토어 생성
```bash
python document_processor_image_test.py
```
또는 벡터스토어 빌드 스크립트 사용:
```bash
python vector_store_test.py --folder dataset_test --save_path faiss_index_pymupdf
```

### 3. RAG 서버 실행
```bash
python rag_server.py
```
서버는 기본적으로 8000번 포트에서 실행됩니다.

### 4. Open WebUI 연동
Open WebUI의 모델 설정에서 다음과 같이 설정:
- **API Base URL**: `http://localhost:8000/v1`
- **API Key**: `token-abc123`
- **Model Name**: `rag`

### 5. 개별 테스트
명령줄에서 직접 질문:
```bash
python rag_system.py --query "문서에서 찾고 싶은 내용"
```

대화형 모드:
```bash
python rag_system.py
```

## 프로젝트 구조

```
open-webui-rag-system/
├── document_processor_image_test.py    # 문서 처리 메인 모듈
├── vector_store_test.py                # 벡터스토어 생성 모듈
├── rag_system.py                       # RAG 체인 구성 및 질의응답
├── rag_server.py                       # FastAPI 서버
├── llm_loader.py                      # LLM 모델 로더
├── e5_embeddings.py                   # E5 임베딩 모듈
├── requirements.txt                   # 의존성 목록
├── dataset_test/                      # 문서 저장 폴더
└── faiss_index_pymupdf/              # 생성된 벡터스토어
```

## 핵심 모듈 설명

### document_processor_image_test.py
- PDF와 HWPX 파일의 텍스트, 표, 이미지를 페이지별로 추출
- PyMuPDF, pdfplumber, pytesseract를 활용한 다층 처리
- 섹션별 메타데이터와 페이지 정보 유지

### vector_store_test.py
- E5-Large 임베딩 모델을 사용한 벡터화
- FAISS를 이용한 효율적인 벡터스토어 구축
- 배치 처리를 통한 메모리 최적화

### rag_system.py
- Refine 체인을 활용한 다단계 답변 생성
- 페이지 번호 hallucination 방지 프롬프트
- 출처 추적과 메타데이터 관리

### rag_server.py
- OpenAI 호환 API 엔드포인트 제공
- 스트리밍 응답 지원
- Open WebUI와의 원활한 연동

## 설정 옵션

### 문서 처리 옵션
- **청크 크기**: `chunk_size=500` (기본값)
- **청크 겹침**: `chunk_overlap=100` (기본값)
- **OCR 언어**: `lang='kor+eng'` (한국어+영어)

### 검색 옵션
- **검색 문서 수**: `k=7` (기본값)
- **임베딩 모델**: `intfloat/multilingual-e5-large-instruct`
- **디바이스**: `cuda` 또는 `cpu`

### LLM 설정
지원하는 모델들:
- LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct
- meta-llama/Meta-Llama-3-8B-Instruct
- 기타 OpenAI 호환 모델

## 트러블슈팅

### 1. CUDA 메모리 부족
```bash
# CPU 모드로 실행
python vector_store_test.py --device cpu
```

### 2. 한글 폰트 문제
```bash
# 한글 폰트 설치 (Ubuntu)
sudo apt-get install fonts-nanum
```

### 3. Tesseract 경로 문제
```python
# pytesseract 경로 수동 설정
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
```

### 4. 모델 다운로드 실패
```bash
# Hugging Face 캐시 경로 확인
export HF_HOME=/path/to/huggingface/cache
```

## API 사용 예시

### 직접 질의
```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "문서에서 예산 관련 내용을 찾아주세요"}'
```

### OpenAI 호환 API
```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "rag",
    "messages": [{"role": "user", "content": "예산 현황이 어떻게 되나요?"}],
    "stream": false
  }'
```

## 성능 최적화

### 1. 배치 크기 조정
```bash
python vector_store_test.py --batch_size 32  # GPU 메모리에 따라 조정
```

### 2. 청크 크기 최적화
```python
# 긴 문서의 경우 청크 크기 증가
chunks = split_documents(docs, chunk_size=800, chunk_overlap=150)
```

### 3. 검색 결과 수 조정
```bash
python rag_system.py --k 10  # 더 많은 문서 참조
```

## 라이선스

MIT License

## 기여 방법

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request
