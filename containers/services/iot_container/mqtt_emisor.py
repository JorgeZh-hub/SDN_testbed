#!/usr/bin/env python3
import paho.mqtt.client as mqtt
import time
import json
import random
import argparse

# -------------------------------------------------------------
# MQTT publisher that aims for a target bandwidth.
# -------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Sensor simulator that sends JSON over MQTT with a target bandwidth."
)
parser.add_argument(
    "--broker",
    default="203.0.113.6",
    help="MQTT broker IP (default: 203.0.113.6)",
)
parser.add_argument(
    "--port",
    type=int,
    default=1883,
    help="MQTT port (default: 1883)",
)
parser.add_argument(
    "--topic",
    default="sensores/sensorX",
    help="Topic to publish to (default: sensores/sensorX)",
)
parser.add_argument(
    "--bw-kbps",
    type=float,
    required=True,
    help="Target bandwidth in kbps (required)",
)
parser.add_argument(
    "--duration",
    "-t",
    type=float,
    default=None,
    help="Total run time in seconds (default: infinite)",
)
parser.add_argument(
    "--extra-bytes",
    type=int,
    default=0,
    help="Extra padding bytes in the payload to adjust size (default: 0)",
)
args = parser.parse_args()

bw_bps = args.bw_kbps * 1000.0  # kbps -> bps

# Cliente MQTT
client = mqtt.Client()

try:
    client.connect(args.broker, args.port, 60)
except Exception as e:
    print(f"Failed to connect to MQTT broker: {e}")
    exit(1)

if args.duration is not None:
    print(
        f"Connected to {args.broker}:{args.port} - "
        f"Publishing to '{args.topic}' targeting ~{args.bw_kbps:.2f} kbps "
        f"for {args.duration} s..."
    )
else:
    print(
        f"Connected to {args.broker}:{args.port} - "
        f"Publishing to '{args.topic}' targeting ~{args.bw_kbps:.2f} kbps..."
    )

start_time = time.time()
id_medida = 0

try:
    while True:
        if args.duration is not None and (time.time() - start_time) >= args.duration:
            break

        id_medida += 1

        payload = {
            "sensor_id": "SENSOR_X_001",
            "id_medida": id_medida,
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

        client.publish(args.topic, mensaje)

        time.sleep(intervalo)

except KeyboardInterrupt:
    pass
finally:
    client.disconnect()
    print("MQTT publisher stopped.")
