FROM ctaloi/sipp:latest

ENV DEBIAN_FRONTEND=noninteractive

# Agregamos tcpdump a la lista de instalación
RUN apt-get update && apt-get install -y \
    iputils-ping \
    iproute2 \
    net-tools \
    curl \
    tcpdump \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
EXPOSE 6000/udp

ENTRYPOINT []
CMD ["/bin/sh"]