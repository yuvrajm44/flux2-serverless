FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

# Install git
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Clone and install SimpleTuner with CUDA support (like your RunPod setup)
RUN git clone https://github.com/bghira/SimpleTuner.git /tmp/SimpleTuner && \
    cd /tmp/SimpleTuner && \
    pip install -e ".[cuda]" && \
    pip install gdown && \
    cd /app

# Copy application code
COPY . .

# Download models at build time
RUN python download_models.py

# Start the RunPod handler
CMD ["python", "-u", "runpod_handler.py"]