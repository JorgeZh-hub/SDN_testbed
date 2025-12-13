FROM python:3.8-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    tcpdump iproute2 iputils-ping net-tools vim \
    && rm -rf /var/lib/apt/lists/*


RUN pip install --no-cache-dir --upgrade pip

RUN pip install --no-cache-dir "ryu" "eventlet<0.30.3"

WORKDIR /app

ENTRYPOINT ["ryu-manager"]
