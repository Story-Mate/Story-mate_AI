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
COPY quiz.py .

COPY 운수좋은날/ /app/운수좋은날
COPY 인어공주/ /app/인어공주
COPY 엄지공주/ /app/엄지공주
COPY 메밀꽃필무렵/ /app/메밀꽃필무렵
COPY 날개/ /app/날개
COPY 동백꽃/ /app/동백꽃
COPY 미운아기오리/ /app/미운아기오리
COPY 성냥팔이소녀/ /app/성냥팔이소녀
COPY 시골쥐서울구경/ /app/시골쥐서울구경
COPY 심봉사/ /app/심봉사



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
