import argparse
import time
import requests


parser = argparse.ArgumentParser(description="HTTP client to poll for data.")
parser.add_argument("host", help="Server host or IP")
parser.add_argument("-p", "--port", default="5000", help="Server port (default 5000)")
parser.add_argument("-r", "--path", default="/datos/PACIENTE_001", help="HTTP resource path (default /datos/PACIENTE_001)")
parser.add_argument("-t", "--duration", type=int, default=0, help="Seconds to poll (0 for unlimited)")

args = parser.parse_args()
SERVER_URL = f"http://{args.host}:{args.port}{args.path}"

INTERVAL_SECONDS = 5
start_time = time.time()

while True:
    try:
        response = requests.get(SERVER_URL)
        if response.status_code == 200:
            data = response.json()
            print(f"[OK] Data received: {data}")
        elif response.status_code == 404:
            print("[INFO] No new data.")
        else:
            print(f"[ERROR] {response.status_code}: {response.text}")
    except Exception as e:
        print(f"[WARN] Exception: {e}")

    time.sleep(INTERVAL_SECONDS)

    if args.duration > 0 and (time.time() - start_time) >= args.duration:
        print("[DONE] Duration reached. Exiting.")
        break
