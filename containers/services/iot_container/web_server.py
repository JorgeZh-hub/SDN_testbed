import asyncio
import websockets
import json

HOST = '0.0.0.0'
PORT = 8080

async def handle_client(websocket):
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
            except json.JSONDecodeError:
                print("Invalid JSON message received.")
    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected.")

async def main():
    async with websockets.serve(handle_client, HOST, PORT):
        print(f"WebSocket server listening at ws://{HOST}:{PORT}")
        await asyncio.Future()  # keep running

asyncio.run(main())
