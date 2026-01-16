import os
import sys

# 현재 실행 중인 venv의 lib 폴더를 DLL 탐색 경로에 강제로 추가
dll_path = os.path.join(sys.prefix, 'Lib', 'site-packages', 'torch', 'lib')
if os.path.exists(dll_path):
    os.add_dll_directory(dll_path)

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import chromadb
import pymupdf4llm
import tiktoken
from sentence_transformers import SentenceTransformer
from chromadb import Documents, EmbeddingFunction, Embeddings
from dotenv import load_dotenv
import google.generativeai as genai

# 환경 변수 로드
load_dotenv()

app = FastAPI()


class QueryRequest(BaseModel):
    query: str


# 전역 변수로 모델과 클라이언트 초기화
embedding_model = None
chroma_client = None
gemini_model = None

# PDF 파일 경로 설정 (여기서 수정!)
PDF_FILE_PATH = "../recycle.pdf"  # 실제 PDF 파일 경로로 변경


@app.on_event("startup")
async def startup_event():
    global embedding_model, chroma_client, gemini_model

    print("Loading embedding model...")
    embedding_model = SentenceTransformer('jhgan/ko-sroberta-multitask')

    print("Initializing ChromaDB client...")
    chroma_client = chromadb.PersistentClient()

    # Gemini 초기화
    print("Initializing Gemini...")
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("WARNING: GEMINI_API_KEY not found in environment variables")
    else:
        genai.configure(api_key=api_key)
        gemini_model = genai.GenerativeModel('gemini-3-flash')
        print("Gemini initialized successfully!")

    # 시작 시 PDF 자동 로드
    await load_pdf_to_db()

    print("Startup complete!")


def split_text(full_text, chunk_size=1000):
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
    """서버 시작 시 PDF를 자동으로 로드"""
    global embedding_model, chroma_client

    try:
        # PDF 파일 존재 확인
        if not os.path.exists(PDF_FILE_PATH):
            print(f"ERROR: PDF file not found at {PDF_FILE_PATH}")
            return

        print(f"Loading PDF from: {PDF_FILE_PATH}")

        # PDF 텍스트 추출
        md_text = pymupdf4llm.to_markdown(PDF_FILE_PATH)

        if not md_text or len(md_text.strip()) == 0:
            print("ERROR: No text extracted from PDF")
            return

        print(f"Extracted {len(md_text)} characters from PDF")

        # 청크로 분할
        chunk_list = split_text(md_text, chunk_size=1000)
        print(f"Split into {len(chunk_list)} chunks")

        # 임베딩 생성
        print(f"Creating embeddings...")
        embeddings = embedding_model.encode(chunk_list)

        # ChromaDB에 저장
        class MyEmbeddingFunction(EmbeddingFunction):
            def __call__(self, input: Documents) -> Embeddings:
                return embedding_model.encode(input).tolist()

        collection_name = 'samsung_collection6'

        # 기존 collection 삭제
        try:
            chroma_client.delete_collection(name=collection_name)
            print(f"Deleted existing collection: {collection_name}")
        except:
            pass

        # 새 collection 생성
        samsung_collection = chroma_client.create_collection(
            name=collection_name,
            embedding_function=MyEmbeddingFunction()
        )

        # 문서 추가
        id_list = [f'{index}' for index in range(len(chunk_list))]
        samsung_collection.add(documents=chunk_list, ids=id_list)

        print(f"✓ Successfully loaded {len(chunk_list)} chunks to ChromaDB")

    except Exception as e:
        print(f"ERROR loading PDF: {str(e)}")


# PDF 수동 재로드 엔드포인트 (필요시)
@app.post("/reload-pdf")
async def reload_pdf():
    """PDF를 다시 로드"""
    try:
        await load_pdf_to_db()
        return {"ok": True, "message": "PDF reloaded successfully"}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": str(e)}
        )


# Gemini를 사용한 답변 생성
@app.post("/answer")
async def llm_response(request: QueryRequest):
    global embedding_model, chroma_client, gemini_model

    if not gemini_model:
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": "Gemini API key not configured"}
        )

    collection_name = 'samsung_collection6'

    class MyEmbeddingFunction(EmbeddingFunction):
        def __call__(self, input: Documents) -> Embeddings:
            return embedding_model.encode(input).tolist()

    try:
        samsung_collection = chroma_client.get_collection(
            name=collection_name,
            embedding_function=MyEmbeddingFunction()
        )

        retrieved_doc = samsung_collection.query(query_texts=request.query, n_results=3)
        refer = retrieved_doc['documents'][0]

        # Gemini 프롬프트
        prompt = f'''당신은 한국의 비즈니스 분석 전문가입니다.
        주어진 *Context*에서 사용자의 질문에 대한 답변을 찾아주세요. 
        만약 관련 정보가 없다면, 회사에 문의하도록 안내해주세요.
        답변은 사용자가 이해하기 쉽게 정리해주세요.
        
        *Context*:
{refer}

*질문*: {request.query}

한국어로 답변해주세요:'''

        # Gemini API 호출
        response = gemini_model.generate_content(prompt)

        return {
            "response": response.text,
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
    global chroma_client

    try:
        collection = chroma_client.get_collection(name='samsung_collection6')
        count = collection.count()
        return {
            "ok": True,
            "collection_name": "samsung_collection6",
            "document_count": count
        }
    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "message": "Collection not found. Please upload a document first."
        }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)