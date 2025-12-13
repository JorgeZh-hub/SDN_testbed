#!/usr/bin/env python3
"""
WebSocket traffic generator for coap_server_iot_0/web_server.py.
Controls load by approximate bandwidth (--bw Mbps) using a large fixed payload
and an automatically calculated send rate.
"""
import argparse
import asyncio
import json
import os
import random
import string
import time

import websockets


def random_payload(size_bytes: int) -> str:
    """Generate a random string of the requested size."""
    alphabet = string.ascii_letters + string.digits
    return "".join(random.choice(alphabet) for _ in range(size_bytes))


async def publish_loop(uri, topics, target_bw_mbps, duration):
    # Large payload to approach MTU (JSON + overhead; ~1.4 KB body)
    payload_size = 1400
    payload_bits = (payload_size + 200) * 8  # estimated JSON/TCP overhead
    target_bps = target_bw_mbps * 1_000_000
    rate_hz = max(target_bps / payload_bits, 1.0)  # at least 1 msg/s
    interval = 1.0 / rate_hz
    end_time = time.time() + duration if duration > 0 else None

    async with websockets.connect(uri) as ws:
        while True:
            now = time.time()
            if end_time and now >= end_time:
                break

            topic = random.choice(topics)
            payload = random_payload(payload_size)
            msg = {
                "topic": topic,
                "ts": time.time(),
                "payload": payload,
            }
            await ws.send(json.dumps(msg))

            if interval > 0:
                await asyncio.sleep(interval)


def main():
    parser = argparse.ArgumentParser(
        description="WebSocket publisher to web_server.py controlled by approximate bandwidth."
    )
    parser.add_argument("--host", default=os.environ.get("WEB_HOST", "localhost"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("WEB_PORT", 8080)))
    parser.add_argument(
        "--topics",
        default="sensor/temp,sensor/hum,sensor/power",
        help="Comma-separated list of simulated topics.",
    )
    parser.add_argument(
        "--bw",
        type=float,
        default=float(os.environ.get("WEB_BW", 1.0)),
        help="Target bandwidth in Mbps (approx).",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=int(os.environ.get("WEB_DURATION", 0)),
        help="Duration in seconds (0 = infinite).",
    )

    args = parser.parse_args()
    topics = [t.strip() for t in args.topics.split(",") if t.strip()]
    if not topics:
        raise SystemExit("You must specify at least one topic.")

    uri = f"ws://{args.host}:{args.port}"
    print(f"Connecting to {uri} with topics={topics}, bw={args.bw} Mbps (approx)")

    asyncio.run(
        publish_loop(
            uri=uri,
            topics=topics,
            target_bw_mbps=args.bw,
            duration=args.duration,
        )
    )


if __name__ == "__main__":
    main()
