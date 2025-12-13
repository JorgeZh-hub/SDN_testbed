FROM ubuntu:22.04


ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    curl \
    iputils-ping \
    iproute2 \
    net-tools \
    wget \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app

RUN wget https://github.com/bluenviron/mediamtx/releases/download/v1.12.2/mediamtx_v1.12.2_linux_amd64.tar.gz && \
    tar -xvzf mediamtx_v1.12.2_linux_amd64.tar.gz && \
    mv mediamtx /usr/local/bin/mediamtx && \
    chmod +x /usr/local/bin/mediamtx && \
    rm -f mediamtx_v1.12.2_linux_amd64.tar.gz

EXPOSE 1935 8554 8888 8889 8189 8000 8001 8890

ENTRYPOINT []
CMD ["/usr/local/bin/mediamtx", "/etc/mediamtx/mediamtx.yml"]

