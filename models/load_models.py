from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
from ultralytics import YOLO

def load_kr_model(model_path="kr_text_detector"):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, use_safetensors=True)
    classification = TextClassificationPipeline(model=model, tokenizer=tokenizer, device=-1, top_k=3)
    return classification

def load_gesture_model():
    return YOLO('YOLOv10x_gestures.pt')
