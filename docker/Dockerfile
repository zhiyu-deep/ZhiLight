FROM nvidia/cuda:12.5.1-devel-ubuntu20.04
RUN export DEBIAN_FRONTEND=noninteractive && \
apt update && \
apt install -y --no-install-recommends sudo curl git git-lfs vim tzdata netcat net-tools openjdk-8-jdk rsync build-essential ssh wget
COPY . /build/zhilight
RUN pip3 install ninja loguru torch pytrie transformers  sentencepiece -i https://pypi.tuna.tsinghua.edu.cn/simple  --no-cache-dir
WORKDIR /build/zhilight
RUN pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple && python3 setup.py bdist_wheel && pip3 install dist/*.whl && rm -rf /build
WORKDIR /app
ENTRYPOINT ["python3", "-m", "zhilight.server.openai.entrypoints.api_server"]