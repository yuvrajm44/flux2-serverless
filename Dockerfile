FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install SimpleTuner
RUN git clone https://github.com/bghira/SimpleTuner.git /tmp/SimpleTuner && \
    cd /tmp/SimpleTuner && \
    pip install --no-cache-dir -e ".[cuda]" && \
    rm -rf /tmp/SimpleTuner/.git

# HuggingFace login
ENV HF_TOKEN=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
RUN pip install --no-cache-dir huggingface-hub && \
    huggingface-cli login --token ${HF_TOKEN}

# Copy code
COPY . .

# Download base models (FLUX 2 + Mistral)
RUN python download_models.py

# Start handler
CMD ["python", "-u", "runpod_handler.py"]