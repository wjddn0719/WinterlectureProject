import os
import requests
from starlette import status
from fastapi import FastAPI, UploadFile, HTTPException, Body
from pydantic import BaseModel
from starlette.requests import Request
from starlette.responses import JSONResponse
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient
from PIL import Image
import io

load_dotenv()

app = FastAPI()

RAG_SERVER_URL = os.getenv("RAG_SERVER_URL", "http://localhost:7777")
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")

# 로그인(쿠키 사용)
class User(BaseModel):
    username: str
    password: str

users = [
    User(username='admin', password='q1w2e3')
]

# 유저 이름 중복 확인
@app.get('/check_username/{username}')
def check_username(username):
    exist = any(username==u.username for u in users)
    return JSONResponse(
        {"ok": not exist},
    )

# 유저 추가
@app.post('/add_user')
def add_user(user: User):
    if not check_username(user.username)['ok']:
        return JSONResponse({"message": "username already exist"}, status_code=status.HTTP_409_CONFLICT)
    users.append(user)
    return JSONResponse({"message": f"User {user.username} added successfully"})

# 로그인
@app.post('/login')
def login(user: User):
    if any(u.username==user.username and u.password==user.password for u in users):
        res = JSONResponse({"message": f"Welcome {user.username}"})
        res.set_cookie("username", user.username)
        return res
    else:
        return JSONResponse({"message": "error"}, status_code=status.HTTP_401_UNAUTHORIZED)

# 유저 확인
def get_current_user(request: Request) -> str:
    username = request.cookies.get('username')
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not logged in")
    if username not in [u.username for u in users]:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized")
    return username

def llm_response(query):
    response = requests.post(
        f'{RAG_SERVER_URL}/answer',
        json={"query": query},
        timeout=180
    )
    response.raise_for_status()
    return response.json()

@app.post('/query')
def query(request: Request, query = Body()):
    username = get_current_user(request)

    lr = llm_response(query)
    return {
        "ok": True,
        "user": username,
        "query": query,
        "llm_response": lr,
    }

# 이미지 업로드 및 분류
client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=ROBOFLOW_API_KEY
)

@app.post('/classify_trash')
async def classify_trash(request: Request, file: UploadFile):
    get_current_user(request)

    # 1. 파일 형식 검증
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 허용됩니다.")

    try:
        # 2. 업로드된 파일 읽기 및 이미지 변환
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # 3. 워크플로우 실행 (image 변수에 실제 이미지 객체 전달)
        result = client.run_workflow(
            workspace_name="winterlecture",
            workflow_id="detect-and-classify",
            images={
                "image": image  # 파일 경로 대신 이미지 객체 주입
            },
            use_cache=True
        )[0]

        # 4. 결과 파싱 (워크플로우 응답 구조에 따라 조정 필요)
        predictions = result["classification_predictions"][0]["predictions"]

        # 5. 가장 신뢰도가 높은 결과 추출
        top = predictions["top"]

        return {
            "item_type": top,
            "confidence": round(predictions["confidence"], 3)
        }

    except Exception as e:
        # 에러 발생 시 상세 내용 반환
        raise HTTPException(status_code=500, detail=f"서버 내부 오류: {str(e)}")