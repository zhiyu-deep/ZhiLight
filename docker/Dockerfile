###
 # ZhiLight CUDA Base Image Dockerfile
 # ubuntu20.04 + cmake>=3.26 + miniconda3_py38_py310 + cuda12.4.1 + nccl2.21.5
##
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu20.04

ARG tsinghua_pypi=https://pypi.tuna.tsinghua.edu.cn/simple

# Install some system tools & python
RUN apt-get update && apt install -y --no-install-recommends\
    sudo wget curl inetutils-ping gdb git gnupg2 vim ca-certificates\
    linux-tools-common linux-tools-`uname -r` linux-cloud-tools-`uname -r` && \
    wget -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /usr/local/miniconda && \
    rm -rf /tmp/miniconda.sh && \
    /usr/local/miniconda/bin/conda create -n py38 python=3.8 -y && \
    /usr/local/miniconda/bin/conda create -n py310 python=3.10 -y && \
    rm -rf /var/lib/apt/lists/*


ENV PATH /usr/local/miniconda/envs/py310/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64
ENV NVIDIA_DISABLE_REQUIRE 1
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# Install cmake
RUN pip install --no-cache-dir cmake==3.30.1 -i ${tsinghua_pypi}

COPY . /build/zhilight

WORKDIR /build/zhilight

# https://hub.gitmirror.com/ is a proxy, maybe you need
# you need choose a right flash-attn wheel package, cuda version, torch version, python version and right abi
RUN pip install --no-deps --no-cache-dir https://hub.gitmirror.com/https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.0.post2/flash_attn-2.7.0.post2+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl && \
	pip install --no-cache-dir ninja -i ${tsinghua_pypi} && \
	pip install --no-cache-dir -r requirements.txt -i ${tsinghua_pypi} && \
	python setup.py bdist_wheel && \
	pip install dist/* && \
	rm -rf /build

WORKDIR /app

ENTRYPOINT ["python", "-m", "zhilight.server.openai.entrypoints.api_server"]