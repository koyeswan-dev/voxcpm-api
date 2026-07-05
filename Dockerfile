FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y git ffmpeg

WORKDIR /app
RUN git clone https://github.com/OpenBMB/VoxCPM.git /app/VoxCPM
WORKDIR /app/VoxCPM
RUN pip install -e .
RUN pip install runpod soundfile

COPY handler.py /app/VoxCPM/handler.py

RUN python -c "from voxcpm import VoxCPM; VoxCPM.from_pretrained('openbmb/VoxCPM2')"

CMD ["python", "-u", "/app/VoxCPM/handler.py"]
