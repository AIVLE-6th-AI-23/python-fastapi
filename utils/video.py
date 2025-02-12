import cv2
from .ocr import try_all_readers
from .hate_gesture import detect_gestures
from .hate_expression import detect_hate_expression 
from .status import update_spring_status
from .type import AnalysisCategoryResultRequestDto
from .constants import FRAME_THRESHOLD
from typing import List

async def analyze_video_frames(boardId, postId, cap, send=True) -> List[AnalysisCategoryResultRequestDto]:
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_results: List[AnalysisCategoryResultRequestDto] = []
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            progress = 10 + (frame_count / total_frames * 80)
            print(progress)
            if(send):
                await update_spring_status(boardId, postId, "Processing Video", progress)

            timestamp = frame_count / fps if fps > 0 else 0 

            text_result = await try_all_readers(frame)
            text_detections = detect_hate_expression(text_result['text']) if text_result['text'].strip() else []

            gesture_detections = detect_gestures(frame)

            for i, detection in enumerate(text_detections):
                metadata = eval(detection.detectionMetadata) if detection.detectionMetadata else {}
                metadata["frame"] = frame_count
                metadata["timestamp"] = round(timestamp, 2)
                text_detections[i] = detection.model_copy(update={"detectionMetadata": str(metadata)})

            for i, detection in enumerate(gesture_detections):
                metadata = eval(detection.detectionMetadata) if detection.detectionMetadata else {}
                metadata["frame"] = frame_count
                metadata["timestamp"] = round(timestamp, 2)
                gesture_detections[i] = detection.model_copy(update={"detectionMetadata": str(metadata)})


            frame_results += text_detections + gesture_detections

            frame_count += FRAME_THRESHOLD

        cap.release()

        if(send):
            await update_spring_status(boardId, postId, "Video Analysis Completed", 90)
        return frame_results

    except Exception as e:
        await update_spring_status(boardId, postId, "FAILED", 0)
        return {"error": str(e)}
