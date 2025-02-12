from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from utils.status import update_spring_status, exit_status
from utils.ocr import try_all_readers
from utils.hate_expression import detect_hate_expression
from utils.hate_gesture import detect_gestures
from utils.video import analyze_video_frames
from utils.file_download import download_file_from_url
from utils.mime_detector import categorize_file, UnsupportedFileTypeError
from utils.type import AnalysisCategoryResultRequestDto, ContentAnalysisRequestDto
from typing import List
from pydantic import BaseModel
import numpy as np
import asyncio
import os
import cv2

app = FastAPI()

@app.get("/test")
async def test_connection():
    return {"status": "success", "message": "FastAPI 서버 정상 작동"}

class AnalysisStartRequestDTO(BaseModel):
    employeeId: str
    postId: int
    boardId: int
    thumbnail: str

@app.post("/analyze/start")
async def start(request: AnalysisStartRequestDTO, background_tasks: BackgroundTasks):
    try:
        print(f"analysis start with request {request}")
        background_tasks.add_task(analyze, request)
        await update_spring_status(request.boardId, request.postId, "Start Analysis", 0)
        return True
    except Exception:
        await update_spring_status(request.boardId, request.postId, "FAILED", 0)
        return False


async def analyzeText(file_path: str, boardId: int, postId: int, employeeId: int) :
    try:
        await update_spring_status(boardId, postId, "Start Text Analysis", 10) # TODO 상태 및 progress 추가
        await asyncio.sleep(1)        
        # 파일 존재 여부 확인
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"파일을 찾을 수 없음: {file_path}")
        
        # 파일 읽어 오기
        with open(file_path, "r", encoding="utf-8") as f:
            text_content = f.read()
        
        await update_spring_status(boardId, postId, "Processing Text Analysis", 30) # TODO 상태 및 progress 추가
        await asyncio.sleep(1)
        # 텍스트 분석    
        detection_result = detect_hate_expression(text_content)
        
        # 탐지 결과 전송
        print(detection_result)
        return detection_result
    except Exception :
        raise

async def analyzeImage(file_path: str, boardId: int, postId: int, employeeId: int) :
    try:
        await update_spring_status(boardId, postId, "Start Image Analysis", 10) # TODO 상태 및 progress 추가
        await asyncio.sleep(1)
        
        # 파일 존재 여부 확인
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"파일을 찾을 수 없음: {file_path}")
        
        # 이미지 로드
        image = cv2.imread(file_path)
        if image is None:
            raise ValueError("이미지를 로드할 수 없음")

        await update_spring_status(boardId, postId, "Processing OCR & Text Analysis", 30)
        await asyncio.sleep(1)
        ##### 혐오 텍스트 감지 #####
        # OCR
        ocr_result = try_all_readers(image)
        # 텍스트 분석
        text_detection_result = []
        text_content = ocr_result['text']
        if not text_content.strip():
            print("no valid ocr result")
        else :
            print(f"ocr result {text_content}")
            text_detection_result = detect_hate_expression(text_content)
        
        
        await update_spring_status(boardId, postId, "Processing Image Analysis", 60) # TODO 상태 및 progress 추가
        await asyncio.sleep(1)
        ##### 혐오 제스처 감지 #####  
        # 제스쳐 분석
        gesture_detection_result = detect_gestures(image)
        
        
        # 탐지 결과 병합
        await update_spring_status(boardId, postId, "Merging Detection Results", 90) # TODO 상태 및 progress 추가
        await asyncio.sleep(1)
        detection_result = text_detection_result + gesture_detection_result
        
        # 탐지 결과 전송
        print(detection_result)
        return detection_result
    except Exception :
        raise
    
async def analyzeVideo(file_path: str, boardId: int, postId: int, employeeId: int) :
    try:
        await update_spring_status(boardId, postId, "Start Video Analysis", 10)
        await asyncio.sleep(1)
        # 파일 존재 여부 확인
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"파일을 찾을 수 없음: {file_path}")
        
        # 비디오 파일 열기
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            raise ValueError("비디오 파일을 열 수 없음")
        
        # 비디오 분석
        detection_result = analyze_video_frames(boardId, postId, cap)
        
        await update_spring_status(boardId, postId, "Merging Detection Results", 90)
        await asyncio.sleep(1)
        
        # 탐지 결과 전송
        print(detection_result)
        return detection_result
        
    except Exception :
        raise
        
        
        
options = {
    "text" : analyzeText,
    "image" : analyzeImage,
    "video" : analyzeVideo,
}
        
async def analyze(request: AnalysisStartRequestDTO):        
    try:
        await update_spring_status(request.boardId, request.postId, "Start Analysis", 0)
        
        # Download file
        file_path = await download_file_from_url(request.thumbnail)
        
        # Content Type 판단
        file_type = categorize_file(file_path)
        
        # text, image, video 별 분석 실행
        result = await options[file_type](file_path, request.boardId, request.postId)
        
        result_summary = ContentAnalysisRequestDto(contentType=file_type)
        
        # 분석 결과 처리 및 Spring boot 서버로 전송 & 알림 전송 후 종료
        await exit_status(request.boardId, request.postId, request.employeeId, result, result_summary)         
    except UnsupportedFileTypeError as e :
        await update_spring_status(request.boardId, request.postId, "FAILED", 0)
        print(f"지원되지 않는 파일 유형: {e}")
    except FileNotFoundError as e :
        await update_spring_status(request.boardId, request.postId, "FAILED", 0)
        print("파일을 찾을 수 없습니다.")
    except Exception as e:
        await update_spring_status(request.boardId, request.postId, "FAILED", 0)
        print(f"Unknown Error : {e}")    

class AnalysisRequest(BaseModel):
    text: str

class AnalysisResponse(BaseModel):
    result: List[AnalysisCategoryResultRequestDto]
    
@app.post("/detect/text", response_model=AnalysisResponse)
def detect_text(request: AnalysisRequest):
    try:
        result = detect_hate_expression(request.text)
        return AnalysisResponse(result=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect/image", response_model=AnalysisResponse)
def detect_image(file: UploadFile = File(...)):
    try:
        image = np.frombuffer(file.file.read(), np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        result = detect_gestures(image)
        return AnalysisResponse(result=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect/video", response_model=AnalysisResponse)
async def detect_video(file: UploadFile = File(...)):
    try:
        video_bytes = await file.read()
        temp_video_path = "temp_video.mp4"
        with open(temp_video_path, "wb") as temp_video:
            temp_video.write(video_bytes)
        
        cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Invalid video file")
        
        result = await analyze_video_frames(0,0,cap,False)
        os.remove(temp_video_path)
        return AnalysisResponse(result=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))