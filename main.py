from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import cv2
import numpy as np
import easyocr
from langdetect import detect
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
import torch
import os
from datetime import datetime
import httpx
from openai import OpenAI
from dotenv import load_dotenv
import json

load_dotenv()
app = FastAPI()

# 작업 상태 추적 딕셔너리
task_status = {}

# spring boot 서버에 상태 업데이트
async def update_spring_status(post_id: int, status: str, progress: float):
    async with httpx.AsyncClient() as client:
        try:
            url = f"/api/posts/{post_id}/status"
            await client.patch(url, json={
                "status": status,
                "progress": progress
            })
        except Exception as e:
            print(f"상태 업데이트 실패: {e}")

# 혐오표현 언어 탐지 모델

api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.perplexity.ai"
)

# OCR 리더 초기화 (한글, 영어 지원)
reader = easyocr.Reader(['ko', 'en', 'ja', 'ch_sim', 'ch_tra', 'ru', 'vi', 'fr', 'hi', 'th', 'ar'])



# 한국어 혐오표현 탐지 모델
kr_model_path = "./kr_text_detector"

kr_tokenizer = AutoTokenizer.from_pretrained(kr_model_path)
kr_model = AutoModelForSequenceClassification.from_pretrained(
    kr_model_path,
    use_safetensors=True
)

kr_classification = TextClassificationPipeline(
    model=kr_model,
    tokenizer=kr_tokenizer,
    device=-1,  # GPU 사용 시 0, CPU 사용 시 -1
    return_all_scores=True
)

# 손동작 탐지 및 분류 모델
gesture_model = YOLO('YOLOv10x_gestures.pt')


def detect_hate_speech(text):
    
    try:
        language = detect(text)
        
        if language == "ko":
            text = kr_classification(text)
    
            messages = [
                {
                "role": "system",
                "content":
                    """당신은 텍스트의 혐오표현을 분석하는 AI입니다.
                    category_scores는 각 혐오 표현 종류별 확률이고
                    단어 자체의 혐오 표현과 커뮤니티 혐오 표현도 포함하여 텍스트를 분석하여 다음 형식의 JSON으로 응답해주세요.
                    응답은 raw JSON이어야 하며, 마크다운이나 코드 블록을 사용하지 마세요.:
                    {
                    "language": "언어 이름(Korean, English, Japanese, Chinese 등)",
                        "input_text": "입력 텍스트",
                        "text_length": 텍스트 길이,
                        "analysis_result":
                        {
                            "flagged": true/false,
                            "categories":
                            {
                                "hate": true/false,
                                "hate/threatening": true/false,
                                "hate/racial": true/false,
                                "hate/religious": true/false,
                                "hate/gender": true/false,
                                "hate/sexual_orientation": true/false,
                                "hate/disability": true/false,
                                "hate/age": true/false,
                                "hate/nationality": true/false,
                                "self-harm": true/false,
                                "sexual": true/false,
                                "sexual/minors": true/false,
                                "violence": true/false,
                                "violence/graphic": true/false
                            },
                            "category_scores":
                            {
                                "hate": 0~1,
                                "hate/threatening": 0~1,
                                "hate/racial": 0~1,
                                "hate/religious": 0~1,
                                "hate/gender": 0~1,
                                "hate/sexual_orientation": 0~1,
                                "hate/disability": 0~1,
                                "hate/age": 0~1,
                                "hate/nationality": 0~1,
                                "self-harm": 0~1,
                                "sexual": 0~1,
                                "sexual/minors": 0~1,
                                "violence": 0~1,
                                violence/graphic": 0~1,
                            }
                        },
                        "summary":
                        {
                            "is_toxic": true/false,
                            "highest_category": "가장 높은 점수의 카테고리",
                            "highest_score": 최고 점수
                        },
                        "ai_analysis": "텍스트에 대한 한글 상세 분석"
                        "analyzedAt": "대한민국 기준 현재 날짜 시 분 초"
                    }"""
                },
                {
                    "role": "user",
                    "content": text
                }
            ]
            
            response = client.chat.completions.create(
                model="sonar",
                messages=messages,
                temperature=0.2
            )
            
            result = response.choices[0].message.content
            
            result = result[7:-3]        
            
            # 정제된 문자열을 JSON으로 파싱
            json_data = json.loads(result)
            
            # JSONResponse로 반환
            return json_data
            
    except Exception as e:
        return {"error": str(e), "text": text}

@app.get("/test")
async def test_connection():
    return {"status": "success", "message": "FastAPI 서버가 정상적으로 응답했습니다."}

@app.post("/analyze/text/{post_id}")
async def analyze_text(post_id: int, text: str):
    try:
        # 초기 상태 설정
        task_status[post_id] = {"progress": 0, "status": "PROCESSING"}
        await update_spring_status(post_id, "PROCESSING", 0)
        
        # 텍스트 분석 시작 (50% 진행)
        await update_spring_status(post_id, "PROCESSING", 50)
        text_analysis = detect_hate_speech(text)
        
        # 최종 결과 생성 (90% 진행)
        await update_spring_status(post_id, "PROCESSING", 90)
        # content_type = "TEXT"
        # analysis_detail = {
        #     "text_analysis": text_analysis
        # }
        
        # 완료 처리
        await update_spring_status(post_id, "COMPLETED", 100)
        task_status[post_id] = {"progress": 100, "status": "COMPLETED"}
        
        return text_analysis
        # return {
        #     "contentType": content_type,
        #     "analysisDetail": analysis_detail,
        #     "analyzedAt": datetime.now().isoformat()
        # }
        
    except Exception as e:
        await update_spring_status(post_id, "FAILED", 0)
        task_status[post_id] = {"progress": 0, "status": "FAILED"}
        raise e

@app.post("/analyze/image/{post_id}")
async def analyze_image(post_id: int, file: UploadFile = File(...)):
    try:
        # 초기 상태 설정
        task_status[post_id] = {"progress": 0, "status": "PROCESSING"}
        await update_spring_status(post_id, "PROCESSING", 0)
        
        # 이미지 읽기 (10% 진행)
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        await update_spring_status(post_id, "PROCESSING", 10)
        
        # 텍스트 분석 시작 (30% 진행)
        await update_spring_status(post_id, "PROCESSING", 30)
        text_results = reader.readtext(image)
        extracted_text = ' '.join([text[1] for text in text_results])
        
        # 텍스트 분석 결과 처리 (50% 진행)
        await update_spring_status(post_id, "PROCESSING", 50)
        if extracted_text.strip():
            text_analysis = detect_hate_speech(extracted_text)
        else:
            text_analysis = {"error": "No text detected", "detected_texts": []}
        
        # 제스처 분석 (70% 진행)
        await update_spring_status(post_id, "PROCESSING", 70)
        gesture_results = gesture_model.predict(image)
        gesture_detections = []
        
        for result in gesture_results:
            for box in result.boxes:
                gesture_detection = {
                    "bbox": box.xyxy[0].tolist(),
                    "confidence": float(box.conf),
                    "class": int(box.cls),
                    "gesture": result.names[int(box.cls)]
                }
                gesture_detections.append(gesture_detection)
        
        # 최종 결과 생성 (90% 진행)
        await update_spring_status(post_id, "PROCESSING", 90)
        content_type = "IMAGE"
        analysis_detail = {
            "text_analysis": text_analysis,
            "gesture_analysis": {
                "detections": gesture_detections,
                "total_gestures": len(gesture_detections)
            }
        }
        
        # 완료 처리
        await update_spring_status(post_id, "COMPLETED", 100)
        task_status[post_id] = {"progress": 100, "status": "COMPLETED"}
        
        return {
            "contentType": content_type,
            "analysisDetail": analysis_detail,
            "analyzedAt": datetime.now().isoformat()
        }
        
    except Exception as e:
        await update_spring_status(post_id, "FAILED", 0)
        task_status[post_id] = {"progress": 0, "status": "FAILED"}
        raise e

@app.post("/analyze/video/{post_id}")
async def analyze_video(post_id: int, file: UploadFile = File(...)):
    try:
        # 초기 상태 설정
        task_status[post_id] = {"progress": 0, "status": "PROCESSING"}
        await update_spring_status(post_id, "PROCESSING", 0)
        
        # 비디오 파일 임시 저장 (10% 진행)
        temp_file = f"temp_{file.filename}"
        with open(temp_file, "wb") as buffer:
            buffer.write(await file.read())
        await update_spring_status(post_id, "PROCESSING", 10)
        
        cap = cv2.VideoCapture(temp_file)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_results = []
        frame_count = 0
        
        # 프레임별 처리 (10-90% 진행)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # 진행률 계산 (10-90%)
            progress = 10 + (frame_count / total_frames * 80)
            await update_spring_status(post_id, "PROCESSING", progress)
            task_status[post_id]["progress"] = progress
            
            # 프레임 분석 로직...
            text_results = reader.readtext(frame)
            extracted_text = ' '.join([text[1] for text in text_results])
            
            if extracted_text.strip():
                text_analysis = detect_hate_speech(extracted_text)
                text_analysis["detected_texts"] = [
                    {"text": text[1], "confidence": float(text[2])} 
                    for text in text_results
                ]
            else:
                text_analysis = {"error": "No text detected", "detected_texts": []}
            
            gesture_results = gesture_model.predict(frame)
            gesture_detections = []
            
            for result in gesture_results:
                for box in result.boxes:
                    gesture_detection = {
                        "bbox": box.xyxy[0].tolist(),
                        "confidence": float(box.conf),
                        "class": int(box.cls),
                        "gesture": result.names[int(box.cls)]
                    }
                    gesture_detections.append(gesture_detection)
            
            frame_results.append({
                "frame_number": frame_count,
                "text_analysis": text_analysis,
                "gesture_analysis": {
                    "detections": gesture_detections,
                    "total_gestures": len(gesture_detections)
                }
            })
            
            frame_count += 1
        
        cap.release()
        os.remove(temp_file)
        
        # 완료 처리
        await update_spring_status(post_id, "COMPLETED", 100)
        task_status[post_id] = {"progress": 100, "status": "COMPLETED"}
        
        return {
            "contentType": "VIDEO",
            "analysisDetail": {
                "total_frames": frame_count,
                "frame_analysis": frame_results
            },
            "analyzedAt": datetime.now().isoformat()
        }
        
    except Exception as e:
        await update_spring_status(post_id, "FAILED", 0)
        task_status[post_id] = {"progress": 0, "status": "FAILED"}
        raise e

@app.get("/analyze/status/{post_id}")
async def get_analysis_status(post_id: int):
    """작업 진행 상태 확인 엔드포인트"""
    if post_id not in task_status:
        return {"status": "NOT_FOUND", "progress": 0}
    return task_status[post_id]

