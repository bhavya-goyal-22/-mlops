FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY configs/ configs/
COPY models/heart_disease.onnx models/heart_disease.onnx

ENV PYTHONPATH=src
EXPOSE 8000

CMD ["uvicorn", "mlops.api:app", "--host", "0.0.0.0", "--port", "8000"]
