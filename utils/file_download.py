import os
import uuid
import httpx
from .constants import SAVE_DIRECTORY

def ensure_directory_exists(directory: str):
    if not os.path.exists(directory):
        os.makedirs(directory)

# 파일 다운로드 (Azure Blob Storage Public URL에서 스트리밍 다운로드)
async def download_file_from_url(url: str) -> str:
    ensure_directory_exists(SAVE_DIRECTORY)

    file_extension = url.split(".")[-1]
    filename = f"{uuid.uuid4()}.{file_extension}"
    file_path = os.path.join(SAVE_DIRECTORY, filename)

    async with httpx.AsyncClient() as client:
        async with client.stream("GET", url) as response:
            if response.status_code != 200:
                raise Exception(f"파일 다운로드 실패: {response.status_code}")

            with open(file_path, "wb") as f:
                async for chunk in response.aiter_bytes():
                    f.write(chunk)

    return file_path
