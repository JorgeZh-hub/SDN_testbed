import argparse
import asyncio
import json
import random
import time
from aiocoap import *


def map_range(x, in_min, in_max, out_min, out_max):
    return int((x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min)


async def send_data(server_uri, target_bw_mbps, wait_response=True, duration=None):
    protocol = await Context.create_client_context()

    loop = asyncio.get_event_loop()
    start_time = loop.time()

    payload_size = 1024  # approximate body size (large JSON)
    overhead_bits = (payload_size + 200) * 8  # estimated CoAP/UDP/IPv6 overhead
    target_bps = target_bw_mbps * 1_000_000
    rate_hz = max(target_bps / overhead_bits, 1.0)
    interval = 1.0 / rate_hz

    try:
        while True:
            if duration is not None and (loop.time() - start_time) >= duration:
                break

            temperature = round(random.uniform(20.0, 40.0), 2)
            light = random.randint(0, 4095)
            scale = round(random.uniform(0.0, 1.0), 2)

            r = map_range(temperature, 20, 40, 0, 255)
            g = map_range(light, 0, 4095, 0, 255)
            b = 255 - g

            data = {
                "temperatura": temperature,
                "luz": light,
                "rgb": {"r": r, "g": g, "b": b},
                "payload": "".join(random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789") for _ in range(payload_size)),
            }

            payload = json.dumps(data).encode("utf-8")

            request = Message(code=PUT, uri=server_uri, payload=payload)

            if wait_response:
                try:
                    response = await protocol.request(request).response
                    try:
                        body = response.payload.decode()
                    except Exception:
                        body = response.payload
                except Exception:
                    pass
            else:
                protocol.request(request)

            await asyncio.sleep(interval)
    finally:
        await protocol.shutdown()


def build_coap_uri(ip, path):
    if path.startswith("/"):
        path = path[1:]
    return f"coap://[::ffff:{ip}]/{path}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CoAP sender for sensor-style data.")
    parser.add_argument("ip", help="CoAP server IP (e.g., 203.0.113.3)")
    parser.add_argument("path", help="CoAP resource (e.g., sensor/data)")
    parser.add_argument(
        "--bw",
        type=float,
        default=1.0,
        help="Approximate target bandwidth in Mbps (default 1.0)",
    )
    parser.add_argument(
        "--nowait",
        action="store_true",
        help="Do not wait for server response (generate traffic without RTT limitation)",
    )
    parser.add_argument(
        "--duration",
        "-t",
        type=float,
        default=None,
        help="Total runtime in seconds (default infinite)",
    )

    args = parser.parse_args()

    server_uri = build_coap_uri(args.ip, args.path)

    asyncio.run(
        send_data(
            server_uri,
            args.bw,
            wait_response=not args.nowait,
            duration=args.duration,
        )
    )
