import asyncio
import websockets
import json
from utils.logger import debug

class WebSocketServer:
    def __init__(self, simulation, host="0.0.0.0", port=8765):
        self.simulation = simulation
        self.host = host
        self.port = port
        self.clients = set()
        self.loop = None

    async def handler(self, websocket, path):
        debug("클라이언트 연결됨")
        self.clients.add(websocket)
        self.simulation.ws = self  # Simulation에서 직접 send 가능하게
        try:
            async for message in websocket:
                debug(f"프론트에서 메시지 수신: {message}")
                data = json.loads(message)
                if data.get("type") == "event":
                    # 프론트에서 event 발생 시
                    self.simulation.on_event(data["event"])
        except websockets.ConnectionClosed:
            debug("클라이언트 연결 종료")
        finally:
            self.clients.remove(websocket)

    async def send_state_update(self, state):
        if not self.clients:
            return
        msg = json.dumps(state)
        await asyncio.gather(*(client.send(msg) for client in self.clients))

    def send(self, state):
        # Simulation에서 호출 (동기), asyncio로 변환
        if self.loop:
            asyncio.run_coroutine_threadsafe(self.send_state_update(state), self.loop)

    def start(self):
        debug(f"WebSocket 서버 시작: ws://{self.host}:{self.port}")
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        start_server = websockets.serve(self.handler, self.host, self.port)
        self.loop.run_until_complete(start_server)
        self.loop.run_forever() 