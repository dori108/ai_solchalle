FROM python:3.10-slim

# 작업 디렉토리 설정
WORKDIR /code

# 시스템 의존성 설치
RUN apt-get update && apt-get install -y git

# requirements 설치
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# 전체 코드 복사
COPY . .

# 포트 설정
ENV PORT 7860
EXPOSE 7860

# 실행 명령
CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app"]
