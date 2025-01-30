from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import cv2
import numpy as np
import easyocr
from langdetect import detect
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

app = FastAPI()

# 모델 초기화
gesture_model = YOLO('yolov10x_gestures.pt')

# OCR 리더 초기화 (한글, 영어 지원)
reader = easyocr.Reader(['ko', 'en'])

# 한국어 혐오표현 탐지 모델
kr_tokenizer = AutoTokenizer.from_pretrained("beomi/KcELECTRA-base")
kr_model = AutoModelForSequenceClassification.from_pretrained(
    "beomi/KcELECTRA-base",
    num_labels=2
)

# 영어 혐오표현 탐지 모델
en_classifier = pipeline("text-classification", 
                       model="facebook/roberta-hate-speech-dynabench-r4-target")

def detect_hate_speech(text):
    try:
        # 언어 감지
        lang = detect(text)
        
        if lang == 'ko':
            # 한국어 텍스트 처리
            inputs = kr_tokenizer(text, 
                                return_tensors="pt", 
                                truncation=True, 
                                max_length=512,
                                padding=True)
            with torch.no_grad():
                outputs = kr_model(**inputs)
            predictions = torch.softmax(outputs.logits, dim=-1)
            is_hate = bool(predictions.argmax().item())
            confidence = float(predictions.max().item())
            return {
                "language": "Korean",
                "is_hate": is_hate,
                "confidence": confidence,
                "text": text
            }
            
        elif lang == 'en':
            # 영어 텍스트 처리
            result = en_classifier(text)[0]
            return {
                "language": "English",
                "is_hate": result['label'] == 'hate',
                "confidence": result['score'],
                "text": text
            }
            
        else:
            return {"error": f"Unsupported language: {lang}", "text": text}
            
    except Exception as e:
        return {"error": str(e), "text": text}

@app.post("/analyze/image")
async def analyze_image(file: UploadFile = File(...)):
    # 이미지 읽기
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # 1. 텍스트 분석
    text_results = reader.readtext(image)
    extracted_text = ' '.join([text[1] for text in text_results])
    
    # 텍스트 분석 결과
    if extracted_text.strip():
        text_analysis = detect_hate_speech(extracted_text)
        text_analysis["detected_texts"] = [
            {"text": text[1], "confidence": float(text[2])} 
            for text in text_results
        ]
    else:
        text_analysis = {"error": "No text detected", "detected_texts": []}
    
    # 2. 제스처 분석
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
    
    # 통합 결과 반환
    return {
        "text_analysis": text_analysis,
        "gesture_analysis": {
            "detections": gesture_detections,
            "total_gestures": len(gesture_detections)
        }
    }

@app.post("/analyze/video")
async def analyze_video(file: UploadFile = File(...)):
    # 비디오 파일 임시 저장
    temp_file = f"temp_{file.filename}"
    with open(temp_file, "wb") as buffer:
        buffer.write(await file.read())
    
    cap = cv2.VideoCapture(temp_file)
    frame_results = []
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # 1. 텍스트 분석
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
        
        # 2. 제스처 분석
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
        
        # 프레임별 분석 결과 저장
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
    
    # 임시 파일 삭제
    import os
    os.remove(temp_file)
    
    return {
        "total_frames": frame_count,
        "frame_analysis": frame_results
    }

