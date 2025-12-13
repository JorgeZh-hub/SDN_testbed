FROM containernet/containernet

RUN apt-get update && apt-get install -y python3-pip && \
    pip3 install --no-cache-dir numpy pyyaml && \
    rm -rf /var/lib/apt/lists/*
