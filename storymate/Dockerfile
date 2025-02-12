FROM python:3.12.8
# 작업 디렉토리 설정
WORKDIR /app

# 필요한 파일 복사
COPY requirements.txt .
COPY app.py .
COPY character.py .  
COPY utils.py .
COPY template.py .
COPY chatbot_sql.py .
COPY 운수좋은날/ /app/운수좋은날
COPY 인어공주/ /app/인어공주


# db환경변수
ENV DB_HOST=localhost
ENV DB_USER=root
ENV DB_PASSWORD=1234
ENV DB_NAME=chatdb
ENV DB_CHARSET=utf8mb4

# 패키지 설치
RUN pip install --no-cache-dir -r requirements.txt

# 5⃣ Flask 서버 실행
CMD ["python", "app.py"]