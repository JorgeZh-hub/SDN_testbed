FROM python:3.9-slim

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN pip install --no-cache-dir \
        aiocoap \
        flask \
        gunicorn \
        pika \
        requests \
        websockets \
        paho-mqtt \
    && apt-get update && apt-get install -y \
        net-tools \
        iputils-ping \
        iproute2 \
        curl \
        procps \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

EXPOSE 5000 8080 80 5672 15672

CMD ["bash"]
