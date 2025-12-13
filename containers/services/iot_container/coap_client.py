import argparse
import asyncio
import json
import time
from aiocoap import *


async def fetch_data(protocol, host, resource):
    uri = f"coap://{host}/{resource}"
    request = Message(code=GET, uri=uri)

    try:
        response = await protocol.request(request).response
        data = json.loads(response.payload.decode())
        rgb = data.get("rgb", {})
        # Data available in 'data' and 'rgb' if needed
    except Exception as e:
        print(f"Error receiving CoAP data: {e}")


async def main():
    parser = argparse.ArgumentParser(description="CoAP client to pull sensor data.")
    parser.add_argument("host", help="CoAP server host (e.g., 203.0.113.3)")
    parser.add_argument(
        "--resource",
        default="sensor/datos",
        help="CoAP resource path (default: sensor/datos)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=0,
        help="Seconds to run (0 = indefinite)",
    )
    args = parser.parse_args()

    protocol = await Context.create_client_context()
    await asyncio.sleep(0.1)

    start_ts = time.time()

    while True:
        await fetch_data(protocol, args.host, args.resource)

        if args.duration > 0 and (time.time() - start_ts) >= args.duration:
            break


asyncio.run(main())
