import httpx
from .constants import BASE_API_URL
from .type import AnalysisCategoryResultRequestDto, ContentAnalysisRequestDto
from typing import List
import asyncio

async def update_spring_status(board_id: int, post_id: int, status: str, progress: int):
    async with httpx.AsyncClient() as client:
        try:
            url = f"{BASE_API_URL}/api/{board_id}/posts/{post_id}/status/{status}-{progress}"
            await client.patch(url)
        except Exception as e:
            print(f"상태 업데이트 실패: {e}")

async def exit_status(board_id: int, post_id: int, employee_id: str, result: List[AnalysisCategoryResultRequestDto], result_summary: ContentAnalysisRequestDto):
    async with httpx.AsyncClient() as client:
        try:
            analysis_url = f"{BASE_API_URL}/api/content-analysis/create"
            analysis_payload = {
                "contentAnalysisRequestDto": result_summary,
                "analysisCategoryResultRequestDto": result 
            }
            response_analysis = await client.post(analysis_url, json=analysis_payload)
            await asyncio.sleep(1)
            
            url = f"{BASE_API_URL}/api/{board_id}/posts/{post_id}/status"
            response_exit = await client.patch(url)
            
            notification_url = f"{BASE_API_URL}/api/{post_id}/content-analysis/notifications"
            notification_payload = {
                "employeeId": employee_id,
                "resultSummary": result_summary
            }
            response_noti = await client.post(notification_url, json=notification_payload)
            if response_analysis.status_code == 200 and response_exit.status_code == 200 and response_noti.status_code == 200:
                print("종료 처리 및 알림 전송 성공")
            else :
                print(f"종료 처리 {response_exit.status_code} 및 알림 전송 실패 {response_noti.status_code}")
        except Exception as e:
            print(f"상태 업데이트 실패: {e}")