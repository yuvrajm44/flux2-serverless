FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

# Install git
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*


# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# VERIFY transformers
RUN python -c "import transformers; print('Transformers:', transformers.__version__); from transformers import AutoProcessor; print('✓ AutoProcessor imported successfully')"

# FORCE upgrade transformers (critical - base image has old version)
RUN pip install --no-cache-dir --upgrade --force-reinstall "transformers>=4.55.0"

# Copy SimpleTuner code (DON'T pip install - avoids skrample dependency)
RUN git clone https://github.com/bghira/SimpleTuner.git /tmp/SimpleTuner && \
    cp -r /tmp/SimpleTuner/simpletuner /app/ && \
    rm -rf /tmp/SimpleTuner

# Copy application code
COPY . .

# Download models at build time
RUN python download_models.py

CMD ["python", "-u", "runpod_handler.py"]
