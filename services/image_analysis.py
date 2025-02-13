from utils.status import update_spring_status
from utils.ocr import try_all_readers
from utils.hate_expression import detect_hate_expression
from utils.hate_gesture import detect_gestures
import asyncio
import cv2
import os

async def analyzeImage(file_path: str, boardId: int, postId: int) :
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
        ocr_result = await try_all_readers(image)
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