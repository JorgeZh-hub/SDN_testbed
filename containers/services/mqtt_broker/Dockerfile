FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y mosquitto net-tools iputils-ping iproute2 curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create Mosquitto runtime directories
RUN mkdir -p /var/lib/mosquitto /var/run/mosquitto
RUN mkdir -p /etc/mosquitto

WORKDIR /app

EXPOSE 1883

CMD ["/bin/bash"]
