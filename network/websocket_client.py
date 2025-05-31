import asyncio
import websockets
import json
import threading

class WebSocketClient:
    def __init__(self, ws_url):
        self.ws_url = ws_url
        self.ws = None
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.run_loop, daemon=True)
        self.thread.start()
        asyncio.run_coroutine_threadsafe(self.connect(), self.loop)

    def run_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    async def connect(self):
        try:
            # Disable automatic pings to avoid keepalive ping timeout errors.
            self.ws = await websockets.connect(self.ws_url, ping_interval=None)
            print("Connected to robot WebSocket.")
        except Exception as e:
            print(f"Failed to connect to WebSocket: {e}")
            self.ws = None

    def send_command(self, left, right):
        if self.ws:
            coro = self._send_command(left, right)
            asyncio.run_coroutine_threadsafe(coro, self.loop)
        else:
            print("WebSocket not connected, cannot send command.")

    async def _send_command(self, left, right):
        try:
            command = json.dumps({"left": left, "right": right})
            await self.ws.send(command)
            print(f"Sent command: {command}")
        except Exception as e:
            print(f"Error sending command: {e}")
            # Attempt to reconnect.
            await self.connect()

    def close(self):
        if self.ws:
            asyncio.run_coroutine_threadsafe(self.ws.close(), self.loop)
        self.loop.call_soon_threadsafe(self.loop.stop)
