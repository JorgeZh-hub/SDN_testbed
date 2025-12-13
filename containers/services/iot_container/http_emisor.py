#!/usr/bin/env python3
import time
import json
import random
import argparse
import requests

def main():
    parser = argparse.ArgumentParser(
        description="HTTP sensor simulator that sends JSON with a target bandwidth."
    )

    parser.add_argument(
        "--host",
        "--ip",
        dest="host",
        default="127.0.0.1",
        help="HTTP server host (default 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="HTTP server port (default 5000)",
    )
    parser.add_argument(
        "--path",
        default="/datos",
        help="Resource path (default /datos)",
    )
    parser.add_argument(
        "--type",
        "--tipo",
        dest="data_type",
        default="PACIENTE",
        help="Data type (default PACIENTE, used in id_TYPE)",
    )
    parser.add_argument(
        "--identifier",
        "--identificador",
        dest="identifier",
        default="001",
        help="Identifier (default 001, for id_TYPE=001)",
    )
    parser.add_argument(
        "--bw-kbps",
        type=float,
        required=True,
        help="Target bandwidth in kbps (required)",
    )
    parser.add_argument(
        "-t",
        "--duration",
        "--tiempo",
        dest="duration",
        type=float,
        default=None,
        help="Total runtime in seconds (default: infinite)",
    )
    parser.add_argument(
        "--extra-bytes",
        type=int,
        default=0,
        help="Extra padding bytes in the payload to adjust size (default 0)",
    )

    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}{args.path}"
    data_type = args.data_type
    identifier = args.identifier
    bw_bps = args.bw_kbps * 1000.0  # kbps to bit/s

    measurement_id = 0
    start_ts = time.time()

    try:
        while True:
            if args.duration is not None and (time.time() - start_ts) >= args.duration:
                print("Configured duration reached, stopping HTTP emitter.")
                break

            measurement_id += 1

            type_id_key = f"id_{data_type}"   # e.g., id_PACIENTE
            payload = {
                "id_medida": measurement_id,
                type_id_key: identifier,
                "timestamp": int(time.time() * 1000),
                "valor": round(random.uniform(10.0, 50.0), 2),
                "unidad": "unitX",
                "estado": random.choice(["OK", "WARN", "FAIL"]),
            }

            if args.extra_bytes > 0:
                payload["relleno"] = "X" * args.extra_bytes

            mensaje = json.dumps(payload, ensure_ascii=False)
            mensaje_bytes = mensaje.encode("utf-8")
            tam_bytes = len(mensaje_bytes)

            intervalo = (tam_bytes * 8.0) / bw_bps

            if intervalo < 0.001:
                intervalo = 0.001

            try:
                resp = requests.post(base_url, json=payload, timeout=2)
                if resp.status_code != 200:
                    continue
            except Exception as e:
                continue

            time.sleep(intervalo)

    except KeyboardInterrupt:
        print("Interrupted by user.")

if __name__ == "__main__":
    main()
