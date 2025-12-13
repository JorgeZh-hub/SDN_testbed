import paho.mqtt.client as mqtt
import struct
import json
import time
from datetime import timedelta, datetime
import argparse
import sys

# Broker configuration
MQTT_BROKER = "203.0.113.1"
MQTT_PORT = 1883
MQTT_USER = ""
MQTT_PASS = ""

TOPIC_SUBSCRIBE = "events/ESP32_001/data"
TOPIC_PUBLISH = "events/ESP32_001/request"

# Globals
timestamps_ms = []
time_strs = []
x_vals = []
y_vals = []
z_vals = []
finished = False


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to broker")
        client.subscribe(TOPIC_SUBSCRIBE)
    else:
        print(f"Failed to connect. Code: {rc}")


def on_message(client, userdata, msg):
    global timestamps_ms, x_vals, y_vals, z_vals, finished, time_strs

    try:
        payload_str = msg.payload.decode("utf-8")
        data = json.loads(payload_str)
        if isinstance(data, dict) and data.get("status") == "finished":
            print("Transmission finished (status: finished)")
            finished = True
            return
    except (UnicodeDecodeError, json.JSONDecodeError):
        pass

    if len(msg.payload) != 16:
        print(f"Unexpected message size ({len(msg.payload)} bytes)")
        return

    try:
        timestamp, x, y, z = struct.unpack("<iiii", msg.payload)
        readable_ts = str(timedelta(milliseconds=timestamp))[:-3]

        print(f"{readable_ts} -> x: {x}, y: {y}, z: {z}")

        timestamps_ms.append(timestamp)
        time_strs.append(readable_ts)
        x_vals.append(x * 0.00374)
        y_vals.append(y * 0.00374)
        z_vals.append(z * 0.00374)
    except struct.error as e:
        print(f"Error unpacking: {e}")


# Arguments
parser = argparse.ArgumentParser(description="MQTT client to receive binary data.")
parser.add_argument("id", help="Device ID")
parser.add_argument("timestamp", type=int, help="Initial UNIX timestamp")
parser.add_argument("duration", type=int, help="Duration in seconds")
args = parser.parse_args()

client = mqtt.Client()
if MQTT_USER:
    client.username_pw_set(MQTT_USER, MQTT_PASS)

client.on_connect = on_connect
client.on_message = on_message
client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
client.loop_start()

payload = {"id": args.id, "timestamp": args.timestamp, "duration": args.duration}
client.publish(TOPIC_PUBLISH, json.dumps(payload))
print(f"Request sent: {payload}")
print("Waiting for data...")

try:
    start_time = time.time()
    timeout = args.duration + 20
    while not finished and (time.time() - start_time) < timeout:
        time.sleep(0.1)
except KeyboardInterrupt:
    print("Interrupted by user.")
finally:
    client.loop_stop()
    client.disconnect()
