FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app/src

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN python -m heart_disease_pca.train

EXPOSE 8000
CMD ["uvicorn", "heart_disease_pca.api:app", "--host", "0.0.0.0", "--port", "8000"]
