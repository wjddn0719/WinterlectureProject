import os
import requests
from starlette import status
from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from pydantic import BaseModel
from starlette.requests import Request
from starlette.responses import JSONResponse
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient



load_dotenv()

app = FastAPI()

RAG_SERVER_URL = os.getenv("RAG_SERVER_URL", "http://localhost:7777")
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")



full_text = ""

@app.get('/')
def root():
    return {"message": "Hello"}

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
async def classify_trash(file: UploadFile):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 허용됩니다.")

    result = client.run_workflow(
        workspace_name="winterlecture",
        workflow_id="detect-and-classify",
        images={
            "image": "YOUR_IMAGE.jpg"
        },
        use_cache=True # Speeds up repeated requests
    )

    # 예측 결과 정리
    predictions = result.get("predictions", [])
    if not predictions:
        return {
            "result": "분류 실패",
            "confidence": 0.0
        }

    top = max(predictions, key=lambda x: x["confidence"])

    return {
        "item_type": top["class"],
        "confidence": round(top["confidence"], 3)
    }

