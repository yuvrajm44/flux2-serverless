FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

# Install git
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Clone and install SimpleTuner FIRST (establishes baseline dependencies)
RUN git clone https://github.com/bghira/SimpleTuner.git /tmp/SimpleTuner && \
    pip install --no-cache-dir /tmp/SimpleTuner && \
    rm -rf /tmp/SimpleTuner

# Then install additional requirements (if any are still needed)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Download models at build time
RUN python download_models.py

# Start the RunPod handler
CMD ["python", "-u", "runpod_handler.py"]