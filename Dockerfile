# ComfyUI Refinement Worker for RunPod Serverless
# CUDA 12.4 (compatible with RunPod us-il-1 driver 550.127.05)
# Single venv approach — no comfy-cli, no dual venv conflict

ARG BASE_IMAGE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
FROM ${BASE_IMAGE} AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_PREFER_BINARY=1
ENV PYTHONUNBUFFERED=1

# Install Python 3.12 via deadsnakes (not available natively on Ubuntu 22.04)
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    git \
    wget \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    openssh-server \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3 \
    && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Single virtual environment for everything
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.4
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install ComfyUI via git clone (not comfy-cli)
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /comfyui \
    && cd /comfyui && pip install -r requirements.txt

# Install ComfyUI-Manager
RUN cd /comfyui/custom_nodes \
    && git clone https://github.com/ltdrdata/ComfyUI-Manager.git \
    && cd ComfyUI-Manager && pip install -r requirements.txt || true

# Install Impact-Pack for ADetailer/FaceDetailer
RUN cd /comfyui/custom_nodes \
    && git clone https://github.com/ltdrdata/ComfyUI-Impact-Pack.git \
    && cd ComfyUI-Impact-Pack && pip install -r requirements.txt || true

# Install handler dependencies
RUN pip install runpod requests websocket-client

# Download YOLO detection models
RUN mkdir -p /comfyui/models/ultralytics/bbox \
    && wget -q -O /comfyui/models/ultralytics/bbox/face_yolov8n.pt \
       "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8n.pt" \
    && wget -q -O /comfyui/models/ultralytics/bbox/hand_yolov8s.pt \
       "https://huggingface.co/Bingsu/adetailer/resolve/main/hand_yolov8s.pt" \
    && wget -q -O /comfyui/models/ultralytics/bbox/nipples_yolov8s.pt \
       "https://huggingface.co/ashllay/YOLO_Models/resolve/main/bbox/nipples_yolov8s.pt"

# Download 4x-UltraSharp upscale model
RUN mkdir -p /comfyui/models/upscale_models \
    && wget -q -O /comfyui/models/upscale_models/4x-UltraSharp.pth \
       "https://huggingface.co/lokCX/4x-Ultrasharp/resolve/main/4x-UltraSharp.pth"

# Add extra model paths for network volume
WORKDIR /comfyui
COPY src/extra_model_paths.yaml ./

# Add handler and startup scripts
WORKDIR /
COPY src/start.sh src/network_volume.py handler.py test_input.json ./
RUN chmod +x /start.sh
COPY scripts/comfy-manager-set-mode.sh /usr/local/bin/comfy-manager-set-mode
RUN chmod +x /usr/local/bin/comfy-manager-set-mode || true

# Prevent pip from asking for confirmation
ENV PIP_NO_INPUT=1

WORKDIR /comfyui
CMD ["/start.sh"]
