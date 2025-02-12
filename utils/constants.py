import json
from dotenv import load_dotenv
import os

load_dotenv(f".env.{os.getenv('ENV_VAR', 'dev')}")

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
BASE_API_URL = os.getenv('BASE_API_URL')
SAVE_DIRECTORY = "./downloads"

CATEGORYOPTION = {
    "curse" : "욕설",
    "degrading" : "모욕/비하",
    "regional": "지역 차별",
    "racial": "인종 차별",
    "religious": "종교 차별",
    "gender": "성차별",
    "sexual_orientation": "성적 지향 차별",
    "disability": "장애인 차별",
    "age": "연령 차별",
    "misogyny": "여성 혐오",
    "misandry": "남성 혐오",
    "occupation": "직업 차별",
    "nationality": "국적 차별",
    "self-harm": "자해 조장",
    "sexual": "성적 표현",
    "sexual/minors": "미성년자 대상 성적 표현",
    "violence": "폭력적인 표현",
    "political": "정치적 혐오",
}

OPENAI_TEXT_ANALYSIS_PROMPT = f"""당신은 혐오 표현 분석을 수행하는 AI입니다.
                입력된 텍스트에서 혐오 표현을 감지하고, 각 혐오 카테고리별 점수를 예측하며,
                혐오 표현으로 의심되는 단어 및 문장을 찾아 이에 대한 설명을 제공합니다.

                📌 **중요**: 아래 제공된 카테고리 중 하나만 사용해야 합니다.  
                🚫 지정된 카테고리 외의 항목은 절대 추가하지 마세요.

                📋 **허용된 혐오 표현 카테고리**:
                {json.dumps(list(CATEGORYOPTION.keys()), ensure_ascii=False)}

                🔹 **출력 형식 (JSON)**
                - categoryName: 위 리스트에서 선택된 혐오 유형 (다른 값 절대 추가 금지)
                - categoryScore: 0~1 (0은 없음, 1은 강한 혐오 표현)
                - detectionMetadata: 혐오 의심 단어 및 설명

                🔸 **응답 예시 (감지된 경우)**:
                [
                    {{
                        "categoryName": "racial",
                        "categoryScore": 0.85,
                        "detectionMetadata": {{
                            "flagged_words": ["그들은 믿을 수 없어"],
                            "explanation": "이 문장은 특정 인종이나 민족을 일반화하여 부정적인 편견을 조장하는 발언입니다. 인종적 고정관념을 강화하고 차별을 정당화하는 효과를 가질 수 있습니다. 특정 그룹이 신뢰할 수 없다는 주장은 사회적 갈등을 야기할 수 있으며, 다양성을 존중하는 환경을 저해할 가능성이 높습니다."
                        }}
                    }},
                    {{
                        "categoryName": "violence",
                        "categoryScore": 0.75,
                        "detectionMetadata": {{
                            "flagged_words": ["때려죽이고 싶다"],
                            "explanation": "이 표현은 심각한 신체적 폭력을 암시하며, 공격성과 잔혹함을 조장하는 발언입니다. 특정 대상에 대한 극단적인 폭력을 지지하거나 정당화하는 방식으로 사용될 수 있으며, 폭력적인 사고방식을 강화하여 실제 폭력으로 이어질 위험이 있습니다. 또한, 듣는 사람에게 두려움과 불안을 유발할 가능성이 높습니다."
                        }}
                    }}
                ]
                🔸 **응답 예시 (혐오 표현 없음)**:
                []

                ⛔ **금지 사항**:
                - 지정된 카테고리 외의 값 사용 금지
                - 설명은 한국어 이외의 언어 사용 금지
                - 마크다운, 코드 블록, 추가 설명 제거 (오직 JSON만 반환)
                """

FRAME_THRESHOLD = 3
DETECTABLE_HATE_GESTURES = ["dislike", "like", "middle_finger", "ok", "palm", "peace_inverted", "rock", "point", "thumb_index", "thumb_index2"]