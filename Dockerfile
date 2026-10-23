FROM python:3.13-slim
WORKDIR /app
COPY pyproject.toml ./
COPY uma_api ./uma_api
COPY web ./web
RUN pip install --no-cache-dir .
ENV PORT=8000
CMD ["sh", "-c", "uvicorn uma_api.main:app --host 0.0.0.0 --port ${PORT}"]
