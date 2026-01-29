FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

WORKDIR /app

# Install Python 3.11 (or 3.10) and essential tools
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

# Copy requirements and constraints
COPY requirements.txt constraints.txt ./

# Install packages with constraints
RUN pip install --no-cache-dir -c constraints.txt -r requirements.txt

# Copy SimpleTuner code
RUN git clone https://github.com/bghira/SimpleTuner.git /tmp/SimpleTuner && \
    cp -r /tmp/SimpleTuner/simpletuner /app/ && \
    rm -rf /tmp/SimpleTuner

# Copy application code
COPY . .

CMD ["python", "-u", "runpod_handler.py"]
