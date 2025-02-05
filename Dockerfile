# 베이스 이미지 설정
FROM python:3.9-slim

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip

# Python 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 환경 변수 설정
ENV PERPLEXITY_API_KEY="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"
ENV PORT=3000

# 애플리케이션 코드 복사
COPY . .

# 포트 설정
EXPOSE 8000

# 실행 명령어
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
# CMD ["gunicorn", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "main:app"]
