from pydantic import BaseModel
from typing import List

class ContentAnalysisRequestDto(BaseModel):
    contentType: str
    analysisDetail: str

class AnalysisCategoryResultRequestDto(BaseModel):
    categoryName: str
    categoryScore: float
    detectionMetadata: str

class AnalysisRequest(BaseModel):
    contentAnalysisRequestDto: ContentAnalysisRequestDto
    analysisCategoryResultRequestDto: List[AnalysisCategoryResultRequestDto]
