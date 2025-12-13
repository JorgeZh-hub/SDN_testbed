#!/bin/bash
set -e

# Use the official entrypoint to prepare the environment and launch the broker
exec /usr/local/bin/docker-entrypoint.sh rabbitmq-server
