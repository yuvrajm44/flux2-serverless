FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

WORKDIR /app

# Install Python 3.11 and essential tools
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

# Initialize git-lfs
RUN git lfs install

# Make python3.11 the default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Upgrade pip
RUN pip install --upgrade pip

# Clone SimpleTuner and install with CUDA support
RUN git clone https://github.com/bghira/SimpleTuner.git /tmp/SimpleTuner && \
    cd /tmp/SimpleTuner && \
    pip install -e ".[cuda]" && \
    cp -r /tmp/SimpleTuner/simpletuner /app/ && \
    cd /app && \
    rm -rf /tmp/SimpleTuner

# Install additional requirements
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

CMD ["python", "-u", "runpod_handler.py"]