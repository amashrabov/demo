FROM nvcr.io/nvidia/pytorch:22.10-py3

RUN apt -y update
RUN apt install -y libaio-dev
RUN python3 -m pip install --no-cache-dir --upgrade pip

RUN python3 -m pip uninstall -y torch torchvision torchaudio
RUN pip install --pre torch==2.2.0.dev20231007+cu121 --index-url https://download.pytorch.org/whl/nightly/cu121

RUN python3 -m pip install transformers
RUN python3 -m pip install accelerate
RUN python3 -m pip uninstall -y transformer-engine
RUN python3 -m pip uninstall -y torch-tensorrt apex
RUN python3 -m pip uninstall -y deepspeed || true


# setting default since buildargs of docker-py is not working
ARG UID=1000
ARG GID=1000


WORKDIR /srv

RUN mkdir -p /home/nonroot/ || true

RUN sudo chown -R nonroot:nonroot /srv || true

COPY requirements.txt requirements.txt

RUN python3 -m pip install -r requirements.txt
