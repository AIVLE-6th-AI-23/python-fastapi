from utils.status import update_spring_status
from utils.hate_expression import detect_hate_expression
import os
import asyncio


async def analyzeText(file_path: str, boardId: int, postId: int) :
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