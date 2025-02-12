from langdetect import detect
import re
from .type import AnalysisCategoryResultRequestDto
from .constants import OPENAI_TEXT_ANALYSIS_PROMPT
from .openai import load_openai_client
from models import load_kr_model
from typing import List
import json
import re

def extract_json_array(text):
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        return match.group(0)
    return None
        
def detect_hate_expression(text : str) -> List[AnalysisCategoryResultRequestDto]:
    try:
        sentences = re.split(r'(?<=[.!?。！？])\s+', text)
        detection_results:List[AnalysisCategoryResultRequestDto] = []    
        for txt in sentences:
            language = detect(txt)
            client = load_openai_client()
            if language == "ko":
                print("start")
                kr_classification = load_kr_model()
                kr_result = kr_classification(txt)[0]
                print(kr_result)
                highest_score_result = max(kr_result, key=lambda x: x['score'])
                isClean = highest_score_result['label'] == 'clean'
                additional_info = f"""
                            다음은 입력 텍스트를 한국어 혐오 표현 탐지 모델에 처리한 결과 입니다:
                            {kr_result}
                            이 결과를 참고하여 분석해주세요.
                            """ if not isClean else ""
                
                response = client.chat.completions.create(
                    model="sonar",
                    messages=[
                    {
                        "role":"system",
                        "content": OPENAI_TEXT_ANALYSIS_PROMPT    
                    },
                    {
                        "role": "user",
                        "content": f"""분석할 입력 텍스트:
                        {text}
                        {additional_info}
                        """
                    }
                    ]
                )
                result = response.choices[0].message.content
                json_array = extract_json_array(result)
                if json_array is None:
                    return []
                print(json_array)
                parsed_result = json.loads(json_array)
                for item in parsed_result:
                    item["detectionMetadata"] = json.dumps(item["detectionMetadata"])
            
            else:
                additional_info = f"""
                    다음은 입력 텍스트를 언어 감지 모델에 처리한 결과 입니다.
                    - 이 텍스트는 {language}로 작성 되었습니다.
                    {language}에 존재하는 혐오 표현을 감지할 수 있도록 해당 언어의 맥락을 고려하세요.
                    """
                response = client.chat.completions.create(
                    model="sonar",
                    messages=[
                    {
                        "role":"system",
                        "content": OPENAI_TEXT_ANALYSIS_PROMPT    
                    },
                    {
                        "role": "user",
                        "content": f"""분석할 입력 텍스트:
                        {text}
                        """
                    }
                    ]
                )
                result = response.choices[0].message.content
                json_array = extract_json_array(result)
                if json_array is None:
                    return []
                
                parsed_result = json.loads(json_array)
                for item in parsed_result:
                    item["detectionMetadata"] = json.dumps(item["detectionMetadata"])
        
            detection_results += [AnalysisCategoryResultRequestDto(**item) for item in parsed_result]        
        return detection_results

    except Exception as e:
        raise