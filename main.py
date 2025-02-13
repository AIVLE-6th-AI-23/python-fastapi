from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from utils.status import update_spring_status, exit_status
from utils.hate_gesture import detect_gestures
from utils.hate_expression import detect_hate_expression
from utils.hate_videoframes import detect_hate_videoframes
from utils.file_download import download_file_from_url
from utils.mime_detector import categorize_file, UnsupportedFileTypeError
from utils.type import AnalysisCategoryResultRequestDto, ContentAnalysisRequestDto
from typing import List
from services.text_analysis import analyzeText
from services.image_analysis import analyzeImage
from services.video_analysis import analyzeVideo
from pydantic import BaseModel
import numpy as np
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
        
        
        if not result:
            analysisSummary = """
                    ✅ 분석 결과 해당 콘텐츠에서 혐오 표현이 감지되지 않았습니다.  
                    해당 콘텐츠는 AI 기반 혐오 표현 분석 시스템을 통해 검토되었으며,  
                    명백한 혐오 표현이나 공격적인 언어가 포함되지 않은 것으로 분석되었습니다.    

                    콘텐츠 정책 및 내부 검수 기준에 따라 추가적인 확인이 필요할 수 있습니다.
                    """
        else :
            category_counts = {}  # 카테고리별 개수 저장
            for detection in result:
                category = detection.categoryName
                category_counts[category] = category_counts.get(category, 0) + 1

            detected_summary = ", ".join(f"{count}건의 {category}" for category, count in category_counts.items())

            analysisSummary = f"""
                    ⚠️ 분석 결과 해당 콘텐츠에서 총 {len(result)}건의 혐오 표현이 감지되었습니다.
                    감지된 혐오 표현 유형
                        {detected_summary}
                    
                    본 분석 결과는 AI 기반 혐오 표현 탐지 시스템을 통해 자동으로 산출된 것으로 
                    콘텐츠 정책 및 내부 검수 기준에 따라 추가적인 확인이 필요할 수 있습니다. 
                    """
        result_summary = ContentAnalysisRequestDto(contentType=file_type, analysisDetail=analysisSummary)
        
        
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
    finally:
        os.remove(file_path)    

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
        
        result = await detect_hate_videoframes(0,0,cap,False)
        os.remove(temp_video_path)
        return AnalysisResponse(result=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/test/api/status")
async def testApi(boardId: int, postId: int,status:str ,progress:int):
    try:
        await update_spring_status(boardId,postId,status,progress)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/test/api/status/exit")
async def testApiexit(boardId: int, postId: int,employeeId: str):
    try:
        await exit_status(boardId,postId, employeeId, [], ContentAnalysisRequestDto(contentType="unknown", analysisDetail="empty"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))