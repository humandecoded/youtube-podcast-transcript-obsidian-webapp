FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Add ffmpeg for audio demuxing (yt-dlp + ASR)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates tzdata curl ffmpeg unzip \
  && rm -rf /var/lib/apt/lists/*

# Install Bun for yt-dlp JavaScript runtime
RUN curl -fsSL https://bun.sh/install | bash

# Add Bun to PATH
ENV PATH="/root/.bun/bin:${PATH}"

WORKDIR /app

# Keep pip tooling current so wheels resolve smoothly
RUN python -m pip install --no-cache-dir -U pip setuptools wheel

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV FLASK_ENV=production PORT=5050
EXPOSE 5050
