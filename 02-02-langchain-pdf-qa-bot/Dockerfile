FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    GRADIO_ANALYTICS_ENABLED=False \
    ANONYMIZED_TELEMETRY=False

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY qabot ./qabot
COPY sample_docs ./sample_docs
COPY app.py step1_load_split.py step2_embed_store.py step3_retrieve.py step4_qa_chain.py ./

# Do not run as root inside the container.
RUN useradd --create-home appuser && mkdir -p /app/data && chown -R appuser /app
USER appuser

EXPOSE 7860
CMD ["python", "app.py"]
