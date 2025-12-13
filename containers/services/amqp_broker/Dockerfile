FROM rabbitmq:3.12-management

ENV RABBITMQ_DEFAULT_USER=iotuser
ENV RABBITMQ_DEFAULT_PASS=iotpass

# Install basic network tools
RUN apt-get update && \
    apt-get install -y \
    net-tools \
    iputils-ping \
    iproute2 \
    curl \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENTRYPOINT []

EXPOSE 5672 15672

CMD ["bash"]
