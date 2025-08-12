FROM python:3.11-slim
WORKDIR /app
COPY . /app

# Combine apt update and install for better caching behavior
RUN apt-get update -y && \
    apt-get install -y awscli ffmpeg libsm6 libxext6 unzip && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

CMD ["python3", "app.py"]