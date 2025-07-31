import asyncio
import json
import os
from utils.logger import debug
from fastapi import FastAPI, WebSocket
from fastapi.responses import PlainTextResponse
from fastapi import WebSocketDisconnect

class WebSocketServer:
    def __init__(self, simulation, host="0.0.0.0", port=None):
        self.simulation = simulation
        self.host = host
        self.port = port or int(os.environ.get("PORT", 8765))
        self.clients = set()
        self.loop = None
        self.simulation_started = False
        debug(f"WebSocket server initialized on port {self.port}")

        self.app = FastAPI()

        @self.app.get("/")
        async def healthcheck():
            return PlainTextResponse("OK")

        @self.app.get("/health")
        async def health():
            return PlainTextResponse("OK")

        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            await self.handler(websocket, "/ws")

    async def handler(self, websocket, path):
        debug("클라이언트 연결됨")
        self.clients.add(websocket)
        self.simulation.ws = self

        try:
            await websocket.send_json({
                "type": "connection_established",
                "message": "Connected to Railway backend"
            })
        except:
            pass

        try:
            while True:
                message = await websocket.receive_text()
                debug(f"프론트에서 메시지 수신: {message}")
                data = json.loads(message)

                if data.get("type") == "start_simulation":
                    await self.handle_start_simulation(data, websocket)
                elif data.get("type") == "reset_simulation":
                    await self.handle_reset_simulation(websocket)
                elif data.get("type") == "event":
                    self.simulation.on_event(data["event"])
                elif data.get("type") == "speed_control":
                    speed = data.get("speed", 1)
                    success = self.simulation.set_speed(speed)
                    response = {
                        "type": "speed_control_response",
                        "success": success,
                        "speed": speed if success else self.simulation.speed
                    }
                    await websocket.send_json(response)
        except WebSocketDisconnect:
            debug("클라이언트 연결 종료")
        finally:
            self.clients.remove(websocket)

    async def handle_reset_simulation(self, websocket):
        debug("Resetting simulation")
        self.simulation_started = False
        self.simulation.reset()

        response = {
            "type": "reset_simulation_response",
            "success": True
        }
        debug(f"Sending reset response to frontend: {response}")
        await websocket.send_json(response)
        debug("Simulation reset successfully")

    async def handle_start_simulation(self, data, websocket):
        debug(f"Received start simulation request: {data}")
        if self.simulation_started:
            response = {
                "type": "start_simulation_response",
                "success": False,
                "error": "Simulation already started. Please reset first."
            }
            await websocket.send_json(response)
            return

        algorithm = data.get("algorithm", "greedy")
        debug(f"Starting simulation with algorithm: {algorithm}")

        from sim.scheduler import Scheduler
        self.simulation.scheduler = Scheduler(algorithm, self.simulation)

        if algorithm == "rl":
            model_path = "models/ppo_best.pth"
            if os.path.exists(model_path):
                try:
                    from rl.agent import PPOAgent
                    from rl.environment import AirportEnvironment

                    rl_env = AirportEnvironment(self.simulation)
                    observation_size = rl_env.observation_space_size
                    action_space = rl_env.action_space

                    rl_agent = PPOAgent(observation_size=observation_size, action_space=action_space)
                    rl_agent.load_model(model_path)
                    self.simulation.set_rl_agent(rl_agent)

                    debug(f"훈련된 RL 모델을 로드했습니다: {model_path}")
                    debug(f"Observation size: {observation_size}, Action space: {action_space}")
                    debug(f"총 액션 수: {action_space[0] * action_space[1]}개")
                except Exception as e:
                    debug(f"RL 모델 로드 중 오류 발생: {e}")
                    response = {
                        "type": "start_simulation_response",
                        "success": False,
                        "error": f"Failed to load RL model: {str(e)}"
                    }
                    await websocket.send_json(response)
                    return
            else:
                debug(f"모델 파일을 찾을 수 없습니다: {model_path}")
                response = {
                    "type": "start_simulation_response",
                    "success": False,
                    "error": f"RL model file not found: {model_path}"
                }
                await websocket.send_json(response)
                return

        schedules = self.simulation.schedules
        min_etd = min([s.flight.etd for s in schedules]) if schedules else 0
        start_time = max(360, min_etd - 20)

        def start_sim():
            self.simulation.start(start_time=start_time)

        import threading
        sim_thread = threading.Thread(target=start_sim, daemon=True)
        sim_thread.start()

        self.simulation_started = True

        response = {
            "type": "start_simulation_response",
            "success": True,
            "algorithm": algorithm,
            "start_time": start_time
        }
        await websocket.send_json(response)
        debug("Simulation started successfully")

    async def send_state_update(self, state):
        if not self.clients:
            return
        await asyncio.gather(*(client.send_json(state) for client in self.clients))

    def send(self, state):
        if self.loop:
            asyncio.run_coroutine_threadsafe(self.send_state_update(state), self.loop)

    def start(self):
        debug(f"FastAPI 서버 시작: http://{self.host}:{self.port}")
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        async def run_server():
            import uvicorn
            config = uvicorn.Config(self.app, host=self.host, port=self.port, log_level="info")
            server = uvicorn.Server(config)
            await server.serve()

        self.loop.run_until_complete(run_server())
