# Zero-Shot Video Analytics Surveillance System (Edge Compute + Client UI)
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface PYTHONUNBUFFERED=1
WORKDIR /home/user/app

COPY --chown=user requirements.txt requirements-nvidia.txt ./
RUN pip install --no-cache-dir --user -r requirements-nvidia.txt

COPY --chown=user . .
EXPOSE 7860

CMD ["python", "server.py", "--compile", "--host", "0.0.0.0", "--port", "7860"]
