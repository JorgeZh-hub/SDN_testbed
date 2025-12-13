FROM debian:12-slim

ENV DEBIAN_FRONTEND=noninteractive

# Install vsftpd and basic utilities
RUN apt-get update && apt-get install -y \
    vsftpd \
    ca-certificates \
    net-tools \
    iputils-ping \
    iproute2 \
    && rm -rf /var/lib/apt/lists/*

# Prepare data directory and user
RUN mkdir -p /data

# User: ftpuser / Password: ftppass / Home: /data
RUN useradd -d /data -s /bin/bash ftpuser && echo "ftpuser:ftppass" | chpasswd
RUN chown -R ftpuser:ftpuser /data

# vsftpd needs this chroot directory
RUN mkdir -p /var/run/vsftpd/empty && \
    chown root:root /var/run/vsftpd/empty && \
    chmod 755 /var/run/vsftpd/empty

# Expose FTP control and passive range
EXPOSE 21 30000-30009

WORKDIR /data

CMD ["/bin/bash"]
