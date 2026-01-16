import os
import sys
import re

# DLL 경로 추가
dll_path = os.path.join(sys.prefix, 'Lib', 'site-packages', 'torch', 'lib')
if os.path.exists(dll_path):
    os.add_dll_directory(dll_path)

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pymupdf4llm
import tiktoken
from sentence_transformers import SentenceTransformer
import time
import numpy as np
from numpy.linalg import norm
import requests
import json

app = FastAPI()


class QueryRequest(BaseModel):
    query: str


# 전역 변수
embedding_model = None
document_store = {
    'documents': [],
    'embeddings': None,
    'ids': []
}

PDF_FILE_PATH = "../recycle.pdf"


@app.on_event("startup")
async def startup_event():
    global embedding_model

    print("Loading embedding model...")
    embedding_model = SentenceTransformer('jhgan/ko-sroberta-multitask')

    await load_pdf_to_db()

    print("Startup complete!")


def preprocess_text(text):
    """텍스트 전처리"""

    # 1. 연속된 공백/줄바꿈 제거
    text = re.sub(r'\s+', ' ', text)

    # 2. 특수문자 정리 (한글, 영문, 숫자, 기본 문장부호만 유지)
    text = re.sub(r'[^\w\s가-힣.,!?·\-]', '', text)

    # 3. 연속된 구두점 제거
    text = re.sub(r'([.,!?])\1+', r'\1', text)

    # 4. 앞뒤 공백 제거
    text = text.strip()

    # 5. Markdown 헤더 제거 (###, ##, # 등)
    text = re.sub(r'#{1,6}\s', '', text)

    # 6. 대시/언더스코어 반복 제거
    text = re.sub(r'[-_]{3,}', '', text)

    # 7. 짧은 라인 필터링 (10자 미만 제거)
    lines = [line.strip() for line in text.split('\n') if len(line.strip()) > 10]
    text = ' '.join(lines)

    return text


def split_text(full_text, chunk_size=1000):
    """텍스트를 토큰 기반으로 청크 분할"""
    encoder = tiktoken.encoding_for_model("gpt-4")
    total_encoding = encoder.encode(full_text)
    total_token_count = len(total_encoding)
    text_list = []

    for i in range(0, total_token_count, chunk_size):
        chunk = total_encoding[i: i + chunk_size]
        decoded = encoder.decode(chunk)
        text_list.append(decoded)

    return text_list


async def load_pdf_to_db():
    """PDF 로드 및 전처리"""
    global embedding_model, document_store

    try:
        if not os.path.exists(PDF_FILE_PATH):
            print(f"ERROR: PDF file not found at {PDF_FILE_PATH}")
            return

        print(f"Loading PDF from: {PDF_FILE_PATH}")
        start_time = time.time()

        # PDF 텍스트 추출
        md_text = pymupdf4llm.to_markdown(PDF_FILE_PATH)
        print(f"⏱️  PDF extraction took: {time.time() - start_time:.2f}s")

        if not md_text or len(md_text.strip()) == 0:
            print("ERROR: No text extracted from PDF")
            return

        print(f"Extracted {len(md_text)} characters (raw)")

        # 전처리
        preprocess_start = time.time()
        cleaned_text = preprocess_text(md_text)
        print(f"⏱️  Preprocessing took: {time.time() - preprocess_start:.2f}s")
        print(f"Cleaned to {len(cleaned_text)} characters")

        # 청크로 분할
        chunk_list = split_text(cleaned_text, chunk_size=1000)
        print(f"Split into {len(chunk_list)} chunks")

        # 임베딩 생성
        print(f"Creating embeddings...")
        embeddings = embedding_model.encode(
            chunk_list,
            batch_size=8,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        # 메모리에 저장
        document_store['documents'] = chunk_list
        document_store['embeddings'] = embeddings
        document_store['ids'] = [f'{i}' for i in range(len(chunk_list))]

        print(f"✅ Total time: {time.time() - start_time:.2f}s")
        print(f"✓ Successfully loaded {len(chunk_list)} chunks")

    except Exception as e:
        print(f"ERROR loading PDF: {str(e)}")
        import traceback
        traceback.print_exc()


@app.post("/answer")
async def llm_response(request: QueryRequest):
    global embedding_model, document_store

    if not document_store['documents']:
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": "No documents loaded"}
        )

    try:
        # 쿼리 전처리
        cleaned_query = preprocess_text(request.query)

        # 쿼리 임베딩
        query_embedding = embedding_model.encode([cleaned_query], convert_to_numpy=True)[0]

        # 코사인 유사도 계산
        doc_embeddings = document_store['embeddings']
        similarities = []

        for doc_emb in doc_embeddings:
            sim = np.dot(query_embedding, doc_emb) / (norm(query_embedding) * norm(doc_emb))
            similarities.append(sim)

        # 상위 3개 선택
        top_indices = np.argsort(similarities)[-3:][::-1]
        refer = [document_store['documents'][i] for i in top_indices]

        # Ollama LLM 호출
        url = "http://localhost:11434/api/generate"

        payload = {
            "model": "gemma2:2b",
            "prompt": f'''분리수거어케함

*Context*:
{refer}

*질문*: {request.query}

한국어로 답변해주세요:''',
            "stream": False
        }

        headers = {"Content-Type": "application/json"}

        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)

        return {
            "response": response.json()["response"],
            "context": refer
        }

    except Exception as e:
        print(f"Error in llm_response: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": f"Error: {str(e)}"}
        )


@app.get("/collection-info")
def get_collection_info():
    global document_store
    count = len(document_store['documents'])
    return {
        "ok": True,
        "collection_name": "in_memory_store",
        "document_count": count
    }


@app.get("/")
def root():
    return {
        "message": "RAG Server is running",
        "endpoints": ["/collection-info", "/answer"],
        "document_count": len(document_store['documents'])
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)