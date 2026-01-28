FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Clone SimpleTuner but DON'T install it - just copy the needed modules
RUN git clone https://github.com/bghira/SimpleTuner.git /tmp/SimpleTuner && \
    mkdir -p /app/simpletuner && \
    cp -r /tmp/SimpleTuner/helpers /app/simpletuner/ && \
    touch /app/simpletuner/__init__.py && \
    rm -rf /tmp/SimpleTuner

# Copy application files
COPY . .

# Download models during build
RUN python download_models.py

CMD ["python", "-u", "runpod_handler.py"]