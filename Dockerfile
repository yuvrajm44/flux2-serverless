FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# Set timezone non-interactively
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

WORKDIR /app

# Install Python 3.12 and essential tools
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    curl \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

# Initialize git-lfs
RUN git lfs install

# Make python3.12 the default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Install pip for Python 3.12 using get-pip.py
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

# Clone SimpleTuner
RUN git clone https://github.com/bghira/SimpleTuner.git /tmp/SimpleTuner

# Install SimpleTuner with CUDA support
WORKDIR /tmp/SimpleTuner
RUN pip install -e ".[cuda]"

# Copy SimpleTuner code to /app
WORKDIR /app
RUN cp -r /tmp/SimpleTuner/simpletuner /app/ && rm -rf /tmp/SimpleTuner

# Install additional requirements
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

CMD ["python", "-u", "runpod_handler.py"]
