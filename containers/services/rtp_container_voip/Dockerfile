FROM ctaloi/sipp:latest

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    iputils-ping \
    iproute2 \
    net-tools \
    curl \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
EXPOSE 6000/udp

ENTRYPOINT []
CMD ["/bin/sh"]
