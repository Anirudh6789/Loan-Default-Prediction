FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Generate data, run all tracked experiments, and register+promote the best
# model to the MLflow Model Registry at build time, so the image ships with
# a ready-to-serve Production model baked into mlflow.db + mlruns/.
RUN python src/generate_data.py && python src/train.py

ENV PORT=8000
EXPOSE 8000

CMD uvicorn src.app:app --host 0.0.0.0 --port ${PORT}
