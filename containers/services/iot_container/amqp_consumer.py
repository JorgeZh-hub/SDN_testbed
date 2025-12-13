#!/usr/bin/env python3
"""
Simple AMQP (RabbitMQ) consumer to generate receive-side load.
Connects to the broker and consumes messages from a declared queue.
"""
import argparse
import os
import time

import pika


def main():
    parser = argparse.ArgumentParser(description="AMQP (RabbitMQ) consumer.")
    parser.add_argument("--host", default=os.environ.get("AMQP_HOST", "localhost"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("AMQP_PORT", 5672)))
    parser.add_argument("--vhost", default=os.environ.get("AMQP_VHOST", "/"))
    parser.add_argument("--user", default=os.environ.get("AMQP_USER", "guest"))
    parser.add_argument("--password", default=os.environ.get("AMQP_PASS", "guest"))
    parser.add_argument("--queue", default=os.environ.get("AMQP_QUEUE", "iot_queue"))
    parser.add_argument("--prefetch", type=int, default=int(os.environ.get("AMQP_PREFETCH", 50)))
    parser.add_argument(
        "--duration",
        type=int,
        default=int(os.environ.get("AMQP_DURATION", 0)),
        help="Seconds to consume (0 = run indefinitely).",
    )
    args = parser.parse_args()

    credentials = pika.PlainCredentials(args.user, args.password)
    params = pika.ConnectionParameters(
        host=args.host, port=args.port, virtual_host=args.vhost, credentials=credentials
    )

    start_time = time.time()
    end_time = start_time + args.duration if args.duration > 0 else None

    while True:
        if end_time and time.time() >= end_time:
            print("Duration reached; stopping AMQP consumer.")
            break
        try:
            with pika.BlockingConnection(params) as conn:
                channel = conn.channel()
                channel.queue_declare(queue=args.queue, durable=False, auto_delete=False)
                channel.basic_qos(prefetch_count=args.prefetch)

                def _on_msg(ch, method, properties, body):
                    # Minimal processing to avoid slowing the queue
                    ch.basic_ack(delivery_tag=method.delivery_tag)

                print(f"Consuming AMQP on {args.host}:{args.port} queue={args.queue}")

                for method, properties, body in channel.consume(
                    queue=args.queue, inactivity_timeout=1, auto_ack=False
                ):
                    if end_time and time.time() >= end_time:
                        print("Duration reached during consume loop; stopping.")
                        channel.cancel()
                        break
                    if method is None:
                        continue
                    _on_msg(channel, method, properties, body)
        except Exception as exc:
            print(
                f"AMQP consumer error ({type(exc).__name__}): {exc or repr(exc)}. "
                f"host={args.host} port={args.port} vhost={args.vhost} queue={args.queue}. "
                "Retrying in 2s..."
            )
            if end_time and time.time() >= end_time:
                break
            time.sleep(2)


if __name__ == "__main__":
    main()
