#!/usr/bin/env python3
"""
AMQP (RabbitMQ) traffic generator that publishes to a queue.
Controls load by approximate bandwidth (--bw Mbps) using a fixed-size payload.
"""
import argparse
import json
import os
import random
import string
import time

import pika


def random_body(size_bytes: int) -> str:
    alphabet = string.ascii_letters + string.digits
    return "".join(random.choice(alphabet) for _ in range(size_bytes))


def main():
    parser = argparse.ArgumentParser(description="AMQP publisher to generate traffic based on target bandwidth.")
    parser.add_argument("--host", default=os.environ.get("AMQP_HOST", "localhost"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("AMQP_PORT", 5672)))
    parser.add_argument("--vhost", default=os.environ.get("AMQP_VHOST", "/"))
    parser.add_argument("--user", default=os.environ.get("AMQP_USER", "guest"))
    parser.add_argument("--password", default=os.environ.get("AMQP_PASS", "guest"))
    parser.add_argument("--queue", default=os.environ.get("AMQP_QUEUE", "iot_queue"))
    parser.add_argument("--bw", type=float, default=float(os.environ.get("AMQP_BW", 1.0)), help="Target bandwidth in Mbps")
    parser.add_argument("--duration", type=int, default=int(os.environ.get("AMQP_DURATION", 0)), help="0 = run until interrupted")
    args = parser.parse_args()

    credentials = pika.PlainCredentials(args.user, args.password)
    params = pika.ConnectionParameters(
        host=args.host, port=args.port, virtual_host=args.vhost, credentials=credentials
    )

    try:
        connection = pika.BlockingConnection(params)
    except Exception as exc:
        raise SystemExit(
            f"Could not connect to AMQP broker {args.host}:{args.port} vhost={args.vhost} "
            f"user={args.user} -> {type(exc).__name__}: {exc or repr(exc)}"
        )

    channel = connection.channel()
    channel.queue_declare(queue=args.queue, durable=False, auto_delete=False)

    payload_size = 1400
    payload_bits = (payload_size + 200) * 8  # estimated JSON/AMQP overhead
    target_bps = args.bw * 1_000_000
    rate_hz = max(target_bps / payload_bits, 1.0)
    interval = 1.0 / rate_hz
    end_time = time.time() + args.duration if args.duration > 0 else None

    print(
        f"Publishing AMQP traffic to {args.host}:{args.port} queue={args.queue} "
        f"bw={args.bw} Mbps (approx) payload={payload_size} bytes rate~{rate_hz:.1f} msg/s"
    )

    try:
        while True:
            now = time.time()
            if end_time and now >= end_time:
                break

            body = {
                "ts": time.time(),
                "payload": random_body(payload_size),
            }
            channel.basic_publish(exchange="", routing_key=args.queue, body=json.dumps(body))

            if interval > 0:
                time.sleep(interval)
    finally:
        connection.close()
        print("AMQP publisher finished")


if __name__ == "__main__":
    main()
