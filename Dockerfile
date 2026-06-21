# Render deployment — Python 3.12 com ffmpeg
FROM python:3.12-slim

# Instalar ffmpeg (necessário para YouTube → WAV)
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Criar diretórios necessários
RUN mkdir -p uploads outputs

# O MediaPipe faz download do modelo na primeira execução.
# Pré-download para evitar timeout no primeiro request.
ENV MEDIAPIPE_MODEL_CACHE=/app/app/infrastructure/vision/_model_cache

EXPOSE 8000

CMD ["uvicorn", "app.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
