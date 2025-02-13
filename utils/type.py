from pydantic import BaseModel
from typing import List, Dict, Any

class ContentAnalysisRequestDto(BaseModel):
    contentType: str
    analysisDetail: str

class AnalysisCategoryResultRequestDto(BaseModel):
    categoryName: str
    categoryScore: float
    detectionMetadata: Dict[str, Any]

class AnalysisRequest(BaseModel):
    contentAnalysisRequestDto: ContentAnalysisRequestDto
    analysisCategoryResultRequestDto: List[AnalysisCategoryResultRequestDto]
