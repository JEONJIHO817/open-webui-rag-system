FROM python:3.10

WORKDIR /app
COPY . /app

# /tmp 디렉토리 생성 및 권한 부여
RUN mkdir -p /tmp && chmod 1777 /tmp

# TMPDIR 환경변수를 지정하여 pip install 실행
RUN TMPDIR=/tmp pip install --upgrade pip && TMPDIR=/tmp pip install --no-cache-dir -r requirements.txt

EXPOSE 8500

CMD ["uvicorn", "rag_server:app", "--host", "0.0.0.0", "--port", "8500"]
