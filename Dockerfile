FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 NUMBA_CACHE_DIR=/tmp/numba_cache
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN python -m pip install --no-cache-dir --default-timeout=300 --retries=10 -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn","api:app","--host","0.0.0.0","--port","8000","--workers","1"]
