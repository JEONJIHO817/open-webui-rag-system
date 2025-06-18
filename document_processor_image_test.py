import os
import re
import glob
import time
from collections import defaultdict

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# PyMuPDF 라이브러리
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
    print("✅ PyMuPDF 라이브러리 사용 가능")
except ImportError:
    PYMUPDF_AVAILABLE = False
    print("⚠️ PyMuPDF 라이브러리가 설치되지 않음. pip install PyMuPDF로 설치하세요.")

# PDF 처리용
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
import pdfplumber
from pymupdf4llm import LlamaMarkdownReader

# --------------------------------
# 로그 출력
# --------------------------------

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

# --------------------------------
# 텍스트 정제 함수
# --------------------------------

def clean_text(text):
    return re.sub(r"[^\uAC00-\uD7A3\u1100-\u11FF\u3130-\u318F\w\s.,!?\"'()$:\-]", "", text)

def apply_corrections(text):
    corrections = {
        'º©': '정보', 'Ì': '의', '½': '운영', 'Ã': '', '©': '',
        'â€™': "'", 'â€œ': '"', 'â€': '"'
    }
    for k, v in corrections.items():
        text = text.replace(k, v)
    return text

# --------------------------------
# HWPX 처리 (섹션별 처리만 사용)
# --------------------------------

def load_hwpx(file_path):
    """HWPX 파일 로딩 (XML 파싱 방식만 사용)"""
    import zipfile
    import xml.etree.ElementTree as ET
    import chardet
    
    log(f"📥 HWPX 섹션별 처리 시작: {file_path}")
    start = time.time()
    documents = []
    
    try:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            section_files = [f for f in file_list 
                           if f.startswith('Contents/section') and f.endswith('.xml')]
            section_files.sort()  # section0.xml, section1.xml 순서로 정렬
            
            log(f"📄 발견된 섹션 파일: {len(section_files)}개")
            
            for section_idx, section_file in enumerate(section_files):
                section_num = section_idx + 1  # 1-based 섹션 번호
                
                with zip_ref.open(section_file) as xml_file:
                    raw = xml_file.read()
                    encoding = chardet.detect(raw)['encoding'] or 'utf-8'
                    try:
                        text = raw.decode(encoding)
                    except UnicodeDecodeError:
                        text = raw.decode("cp949", errors="replace")

                    tree = ET.ElementTree(ET.fromstring(text))
                    root = tree.getroot()
                    
                    # 네임스페이스 없이 텍스트 찾기
                    t_elements = [elem for elem in root.iter() if elem.tag.endswith('}t') or elem.tag == 't']
                    body_text = ""
                    for elem in t_elements:
                        if elem.text:
                            body_text += clean_text(elem.text) + " "

                    if body_text.strip():
                        documents.append(Document(
                            page_content=apply_corrections(body_text),
                            metadata={
                                "source": file_path,
                                "filename": os.path.basename(file_path),
                                "type": "hwpx_body",
                                "page": f"섹션 {section_num}",  # 섹션 번호를 페이지로 사용
                                "section": section_num,
                                "total_pages": len(section_files)
                            }
                        ))
                        log(f"✅ 섹션 {section_num}: 텍스트 추출 완료 ({len(body_text)} chars)")

                    # 표 찾기
                    table_elements = [elem for elem in root.iter() if elem.tag.endswith('}table') or elem.tag == 'table']
                    if table_elements:
                        table_text = ""
                        for table_idx, table in enumerate(table_elements):
                            table_text += f"[Table {table_idx + 1}]\n"
                            rows = [elem for elem in table.iter() if elem.tag.endswith('}tr') or elem.tag == 'tr']
                            for row in rows:
                                row_text = []
                                cells = [elem for elem in row.iter() if elem.tag.endswith('}tc') or elem.tag == 'tc']
                                for cell in cells:
                                    cell_texts = []
                                    for t_elem in cell.iter():
                                        if (t_elem.tag.endswith('}t') or t_elem.tag == 't') and t_elem.text:
                                            cell_texts.append(clean_text(t_elem.text))
                                    row_text.append(" ".join(cell_texts))
                                if row_text:
                                    table_text += "\t".join(row_text) + "\n"
                        
                        if table_text.strip():
                            documents.append(Document(
                                page_content=apply_corrections(table_text),
                                metadata={
                                    "source": file_path,
                                    "filename": os.path.basename(file_path),
                                    "type": "hwpx_table",
                                    "page": f"섹션 {section_num}",  # 섹션 번호를 페이지로 사용
                                    "section": section_num,
                                    "total_pages": len(section_files)
                                }
                            ))
                            log(f"📊 섹션 {section_num}: 표 {len(table_elements)}개 추출 완료")

                    # 이미지 찾기
                    if [elem for elem in root.iter() if elem.tag.endswith('}picture') or elem.tag == 'picture']:
                        documents.append(Document(
                            page_content="[이미지 포함]",
                            metadata={
                                "source": file_path,
                                "filename": os.path.basename(file_path),
                                "type": "hwpx_image",
                                "page": f"섹션 {section_num}",  # 섹션 번호를 페이지로 사용
                                "section": section_num,
                                "total_pages": len(section_files)
                            }
                        ))
                        log(f"🖼️ 섹션 {section_num}: 이미지 발견")
                        
    except Exception as e:
        log(f"❌ HWPX 처리 오류: {e}")

    duration = time.time() - start
    
    # 페이지 정보 요약 출력
    if documents:
        sections = [doc.metadata.get('section', 0) for doc in documents if doc.metadata.get('section')]
        if sections:
            log(f"📋 추출된 섹션 범위: {min(sections)} ~ {max(sections)}")
    
    log(f"✅ HWPX 처리 완료: {file_path} ⏱️ {duration:.2f}초, 총 {len(documents)}개 문서")
    return documents

# --------------------------------
# PDF 처리 함수들 (기존과 동일)
# --------------------------------

def run_ocr_on_image(image: Image.Image, lang='kor+eng'):
    return pytesseract.image_to_string(image, lang=lang)

def extract_images_with_ocr(pdf_path, lang='kor+eng'):
    try:
        images = convert_from_path(pdf_path)
        page_ocr_data = {}
        for idx, img in enumerate(images):
            page_num = idx + 1
            text = run_ocr_on_image(img, lang=lang)
            if text.strip():
                page_ocr_data[page_num] = text.strip()
        return page_ocr_data
    except Exception as e:
        print(f"❌ 이미지 OCR 실패: {e}")
        return {}

def extract_tables_with_pdfplumber(pdf_path):
    page_table_data = {}
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                page_num = i + 1
                tables = page.extract_tables()
                table_text = ""
                for t_index, table in enumerate(tables):
                    if table:
                        table_text += f"[Table {t_index+1}]\n"
                        for row in table:
                            row_text = "\t".join(cell if cell else "" for cell in row)
                            table_text += row_text + "\n"
                if table_text.strip():
                    page_table_data[page_num] = table_text.strip()
        return page_table_data
    except Exception as e:
        print(f"❌ 표 추출 실패: {e}")
        return {}

def extract_body_text_with_pages(pdf_path):
    page_body_data = {}
    try:
        pdf_processor = LlamaMarkdownReader()
        docs = pdf_processor.load_data(file_path=pdf_path)
        
        combined_text = ""
        for d in docs:
            if isinstance(d, dict) and "text" in d:
                combined_text += d["text"]
            elif hasattr(d, "text"):
                combined_text += d.text
        
        if combined_text.strip():
            chars_per_page = 2000
            start = 0
            page_num = 1
            
            while start < len(combined_text):
                end = start + chars_per_page
                if end > len(combined_text):
                    end = len(combined_text)
                
                page_text = combined_text[start:end]
                if page_text.strip():
                    page_body_data[page_num] = page_text.strip()
                    page_num += 1
                
                if end == len(combined_text):
                    break
                start = end - 100
                
    except Exception as e:
        print(f"❌ 본문 추출 실패: {e}")
    
    return page_body_data

def load_pdf_with_metadata(pdf_path):
    """PDF 파일에서 페이지별 정보를 추출"""
    log(f"📑 PDF 페이지별 처리 시작: {pdf_path}")
    start = time.time()

    # 먼저 PyPDFLoader로 실제 페이지 수 확인
    try:
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(pdf_path)
        pdf_pages = loader.load()
        actual_total_pages = len(pdf_pages)
        log(f"📄 PyPDFLoader로 확인한 실제 페이지 수: {actual_total_pages}")
    except Exception as e:
        log(f"❌ PyPDFLoader 페이지 수 확인 실패: {e}")
        actual_total_pages = 1

    try:
        page_tables = extract_tables_with_pdfplumber(pdf_path)
    except Exception as e:
        page_tables = {}
        print(f"❌ 표 추출 실패: {e}")

    try:
        page_ocr = extract_images_with_ocr(pdf_path)
    except Exception as e:
        page_ocr = {}
        print(f"❌ 이미지 OCR 실패: {e}")

    try:
        page_body = extract_body_text_with_pages(pdf_path)
    except Exception as e:
        page_body = {}
        print(f"❌ 본문 추출 실패: {e}")

    duration = time.time() - start
    log(f"✅ PDF 페이지별 처리 완료: {pdf_path} ⏱️ {duration:.2f}초")

    # 실제 페이지 수를 기준으로 설정
    all_pages = set(page_tables.keys()) | set(page_ocr.keys()) | set(page_body.keys())
    if all_pages:
        max_extracted_page = max(all_pages)
        # 실제 페이지 수와 추출된 페이지 수 중 큰 값 사용
        total_pages = max(actual_total_pages, max_extracted_page)
    else:
        total_pages = actual_total_pages

    log(f"📊 최종 설정된 총 페이지 수: {total_pages}")

    docs = []
    
    for page_num in sorted(all_pages):
        if page_num in page_tables and page_tables[page_num].strip():
            docs.append(Document(
                page_content=clean_text(apply_corrections(page_tables[page_num])),
                metadata={
                    "source": pdf_path,
                    "filename": os.path.basename(pdf_path),
                    "type": "table",
                    "page": page_num,
                    "total_pages": total_pages
                }
            ))
            log(f"📊 페이지 {page_num}: 표 추출 완료")
        
        if page_num in page_body and page_body[page_num].strip():
            docs.append(Document(
                page_content=clean_text(apply_corrections(page_body[page_num])),
                metadata={
                    "source": pdf_path,
                    "filename": os.path.basename(pdf_path),
                    "type": "body",
                    "page": page_num,
                    "total_pages": total_pages
                }
            ))
            log(f"📄 페이지 {page_num}: 본문 추출 완료")
        
        if page_num in page_ocr and page_ocr[page_num].strip():
            docs.append(Document(
                page_content=clean_text(apply_corrections(page_ocr[page_num])),
                metadata={
                    "source": pdf_path,
                    "filename": os.path.basename(pdf_path),
                    "type": "ocr",
                    "page": page_num,
                    "total_pages": total_pages
                }
            ))
            log(f"🖼️ 페이지 {page_num}: OCR 추출 완료")
    
    if not docs:
        docs.append(Document(
            page_content="[내용 추출 실패]",
            metadata={
                "source": pdf_path,
                "filename": os.path.basename(pdf_path),
                "type": "error",
                "page": 1,
                "total_pages": total_pages
            }
        ))
    
    # 페이지 정보 요약 출력
    if docs:
        page_numbers = [doc.metadata.get('page', 0) for doc in docs if doc.metadata.get('page')]
        if page_numbers:
            log(f"📋 추출된 페이지 범위: {min(page_numbers)} ~ {max(page_numbers)}")
    
    log(f"📊 추출된 페이지별 PDF 문서: {len(docs)}개 (총 {total_pages}페이지)")
    return docs

# --------------------------------
# 문서 로딩 및 분할
# --------------------------------

def load_documents(folder_path):
    documents = []

    for file in glob.glob(os.path.join(folder_path, "*.hwpx")):
        log(f"📄 HWPX 파일 확인: {file}")
        docs = load_hwpx(file)
        documents.extend(docs)

    for file in glob.glob(os.path.join(folder_path, "*.pdf")):
        log(f"📄 PDF 파일 확인: {file}")
        documents.extend(load_pdf_with_metadata(file))

    log(f"📚 문서 로딩 전체 완료! 총 문서 수: {len(documents)}")
    return documents

def split_documents(documents, chunk_size=500, chunk_overlap=100):
    log("🔪 청크 분할 시작")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    chunks = []
    for doc in documents:
        split = splitter.split_text(doc.page_content)
        for i, chunk in enumerate(split):
            enriched_chunk = f"passage: {chunk}"
            chunks.append(Document(
                page_content=enriched_chunk,
                metadata={**doc.metadata, "chunk_index": i}
            ))
    log(f"✅ 청크 분할 완료: 총 {len(chunks)}개 생성")
    return chunks

# --------------------------------
# 메인 실행
# --------------------------------

if __name__ == "__main__":
    folder = "dataset_test"
    log("🚀 PyMuPDF 기반 문서 처리 시작")
    docs = load_documents(folder)
    log("📦 문서 로딩 완료")

    # 페이지 정보 확인
    log("📄 페이지 정보 요약:")
    page_info = {}
    for doc in docs:
        source = doc.metadata.get('source', 'unknown')
        page = doc.metadata.get('page', 'unknown')
        doc_type = doc.metadata.get('type', 'unknown')
        
        if source not in page_info:
            page_info[source] = {'pages': set(), 'types': set()}
        page_info[source]['pages'].add(page)
        page_info[source]['types'].add(doc_type)
    
    for source, info in page_info.items():
        max_page = max(info['pages']) if info['pages'] and isinstance(max(info['pages']), int) else 'unknown'
        log(f"  📄 {os.path.basename(source)}: {max_page}페이지, 타입: {info['types']}")

    chunks = split_documents(docs)
    log("💡 E5-Large-Instruct 임베딩 준비 중")
    embedding_model = HuggingFaceEmbeddings(
        model_name="intfloat/e5-large-v2",
        model_kwargs={"device": "cuda"}
    )

    vectorstore = FAISS.from_documents(chunks, embedding_model)
    vectorstore.save_local("faiss_index_pymupdf")

    log(f"📊 전체 문서 수: {len(docs)}")
    log(f"🔗 청크 총 수: {len(chunks)}")
    log("✅ FAISS 저장 완료: faiss_index_pymupdf")
    
    # 페이지 정보가 포함된 샘플 출력
    log("\n📋 실제 페이지 정보 포함 샘플:")
    for i, chunk in enumerate(chunks[:5]):
        meta = chunk.metadata
        log(f"  청크 {i+1}: {meta.get('type')} | 페이지 {meta.get('page')} | {os.path.basename(meta.get('source', 'unknown'))}")