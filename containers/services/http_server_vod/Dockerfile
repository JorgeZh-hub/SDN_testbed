FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y nginx curl iputils-ping net-tools iproute2&& \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

EXPOSE 80

CMD ["/bin/bash"]
