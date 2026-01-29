FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

# Install git AND git-lfs
RUN apt-get update && apt-get install -y git git-lfs && rm -rf /var/lib/apt/lists/*

# Initialize git-lfs
RUN git lfs install



# Copy both requirements.txt AND constraints.txt
COPY requirements.txt constraints.txt ./

# Install with constraints to lock torch/torchvision versions
RUN pip install --no-cache-dir -c constraints.txt -r requirements.txt



# VERIFY transformers
RUN python -c "import transformers; print('Transformers:', transformers.__version__); from transformers import AutoProcessor; print('✓ AutoProcessor imported successfully')"

# Copy SimpleTuner code (DON'T pip install - avoids skrample dependency)
RUN git clone https://github.com/bghira/SimpleTuner.git /tmp/SimpleTuner && \
    cp -r /tmp/SimpleTuner/simpletuner /app/ && \
    rm -rf /tmp/SimpleTuner

# Copy application code
COPY . .

# Download models at build time
#RUN python download_models.py

CMD ["python", "-u", "runpod_handler.py"]
