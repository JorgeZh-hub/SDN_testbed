import paho.mqtt.client as mqtt
import struct
import argparse
import time
import logging
import os

# Swallow logs to avoid container console noise
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(os.devnull)],
)

parser = argparse.ArgumentParser(description="MQTT client with automatic retries.")
parser.add_argument("--host", "--ip", dest="host", type=str, required=True, help="MQTT broker address")
parser.add_argument("--topic", "--topico", dest="topic", type=str, default="events/ESP32_001/data", help="Topic to subscribe")
parser.add_argument("--port", type=int, default=1883, help="MQTT broker port (default: 1883)")
parser.add_argument("--duration", "--tiempo", dest="duration", type=int, default=0, help="Listen duration in seconds (0 = unlimited)")
args = parser.parse_args()

BROKER_HOST = args.host
BROKER_PORT = args.port
LISTEN_DURATION = args.duration
TOPIC = args.topic
RETRY_DELAY = 5  # seconds

start_listening_at = None

def on_message(client, userdata, msg):
    payload = msg.payload
    block = payload

    if len(block) % 16 != 0:
        logging.warning("Received block with invalid size: %s bytes", len(block))
        return

    for i in range(0, len(block), 16):
        try:
            segment = block[i : i + 16]
            timestamp, x, y, z = struct.unpack("<iiii", segment)
            logging.info("Timestamp: %s, X: %s, Y: %s, Z: %s", timestamp, x, y, z)
        except Exception as e:
            logging.error("Error decoding segment #%s: %s", i // 16, e)

def on_disconnect(client, userdata, rc):
    if rc != 0:
        logging.warning("Unexpected disconnect. Code: %s", rc)
        reconnect_forever(client)

def reconnect_forever(client):
    while True:
        try:
            client.reconnect()
            logging.info("Reconnected successfully.")
            break
        except Exception as e:
            logging.warning(f"Reconnect failed: {e}")
            time.sleep(RETRY_DELAY)

client = mqtt.Client()
client.on_message = on_message
client.on_disconnect = on_disconnect

while True:
    try:
        client.connect(BROKER_HOST, BROKER_PORT)
        logging.info(f"Connected to MQTT broker {BROKER_HOST}:{BROKER_PORT}")
        break
    except Exception as e:
        logging.warning(f"Error connecting to broker: {e}")
        time.sleep(RETRY_DELAY)

client.subscribe(TOPIC)
client.loop_start()

if LISTEN_DURATION > 0:
    start_listening_at = time.time()
    try:
        while time.time() - start_listening_at < LISTEN_DURATION:
            time.sleep(0.1)
    finally:
        client.loop_stop()
        client.disconnect()
else:
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        client.loop_stop()
        client.disconnect()
