from utils.status import update_spring_status
from utils.hate_videoframes import detect_hate_videoframes
import asyncio
import cv2
import os

async def analyzeVideo(file_path: str, boardId: int, postId: int) :
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
        detection_result = await detect_hate_videoframes(boardId, postId, cap)
        
        await update_spring_status(boardId, postId, "Merging Detection Results", 90)
        await asyncio.sleep(1)
        
        # 탐지 결과 전송
        print(detection_result)
        return detection_result
        
    except Exception :
        raise
