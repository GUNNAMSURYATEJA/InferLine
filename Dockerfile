# CPU-only inference server image.
FROM python:3.12-slim

WORKDIR /app

# torch CPU wheel first (avoids the giant CUDA wheel), then the rest.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY inferline/ ./inferline/

EXPOSE 8000

CMD ["uvicorn", "inferline.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
