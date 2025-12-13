FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# System packages and media/network tools
RUN apt-get update && apt-get install -y \
    ffmpeg \
    trickle \
    net-tools \
    iputils-ping \
    iproute2 \
    curl \
    iperf3 \
    wget \
    build-essential \
    pkg-config \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

CMD ["/bin/bash"]
