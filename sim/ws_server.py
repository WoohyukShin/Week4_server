import asyncio
import websockets
import json
import os
from utils.logger import debug
import functools

class WebSocketServer:
    def __init__(self, simulation, host="0.0.0.0", port=None):
        self.simulation = simulation
        self.host = host
        # Use Railway's PORT environment variable
        self.port = port or int(os.environ.get("PORT", 8765))
        self.clients = set()
        self.loop = None
        self.simulation_started = False
        debug(f"WebSocket server initialized on port {self.port}")

    async def handler(self, websocket, path):
        debug("클라이언트 연결됨")
        self.clients.add(websocket)
        self.simulation.ws = self  # Simulation에서 직접 send 가능하게
        
        # Send CORS headers for Railway deployment
        try:
            await websocket.send(json.dumps({
                "type": "connection_established",
                "message": "Connected to Railway backend"
            }))
        except:
            pass
            
        try:
            async for message in websocket:
                debug(f"프론트에서 메시지 수신: {message}")
                data = json.loads(message)
                if data.get("type") == "start_simulation":
                    # 프론트에서 시뮬레이션 시작 요청
                    await self.handle_start_simulation(data, websocket)
                elif data.get("type") == "reset_simulation":
                    # 프론트에서 시뮬레이션 리셋 요청
                    await self.handle_reset_simulation(websocket)
                elif data.get("type") == "event":
                    # 프론트에서 event 발생 시
                    self.simulation.on_event(data["event"])
                elif data.get("type") == "speed_control":
                    # 프론트에서 속도 변경 요청
                    speed = data.get("speed", 1)
                    success = self.simulation.set_speed(speed)
                    # Send confirmation back to frontend
                    response = {
                        "type": "speed_control_response",
                        "success": success,
                        "speed": speed if success else self.simulation.speed
                    }
                    await websocket.send(json.dumps(response))
        except websockets.ConnectionClosed:
            debug("클라이언트 연결 종료")
        finally:
            self.clients.remove(websocket)

    async def handle_reset_simulation(self, websocket):
        """Handle reset simulation message from frontend"""
        debug("Resetting simulation")
        
        # Reset the simulation state
        self.simulation_started = False
        
        # Reset the simulation object
        self.simulation.reset()
        
        # Send confirmation to frontend
        response = {
            "type": "reset_simulation_response",
            "success": True
        }
        debug(f"Sending reset response to frontend: {response}")
        await websocket.send(json.dumps(response))
        debug("Simulation reset successfully")

    async def handle_start_simulation(self, data, websocket):
        """Handle start simulation message from frontend"""
        debug(f"Received start simulation request: {data}")
        if self.simulation_started:
            debug("Simulation already started")
            # Send error response to frontend
            response = {
                "type": "start_simulation_response",
                "success": False,
                "error": "Simulation already started. Please reset first."
            }
            await websocket.send(json.dumps(response))
            return
        
        algorithm = data.get("algorithm", "greedy")
        debug(f"Starting simulation with algorithm: {algorithm}")
        
        # Set the algorithm in the simulation by creating a new scheduler
        from sim.scheduler import Scheduler
        self.simulation.scheduler = Scheduler(algorithm, self.simulation)

        if algorithm == "rl":
            import os
            model_path = "models/ppo_best.pth"
            if os.path.exists(model_path):
                try:
                    from rl.agent import PPOAgent
                    from rl.environment import AirportEnvironment
                    
                    # Create environment to get correct sizes
                    rl_env = AirportEnvironment(self.simulation)
                    observation_size = rl_env.observation_space_size
                    action_space = rl_env.action_space
                    
                    # RL 에이전트 초기화 및 모델 로드
                    rl_agent = PPOAgent(observation_size=observation_size, action_space=action_space)
                    rl_agent.load_model(model_path)
                    self.simulation.set_rl_agent(rl_agent)
                    debug(f"훈련된 RL 모델을 로드했습니다: {model_path}")
                    debug(f"Observation size: {observation_size}, Action space: {action_space}")
                    debug(f"총 액션 수: {action_space[0] * action_space[1]}개")
                except Exception as e:
                    debug(f"RL 모델 로드 중 오류 발생: {e}")
                    # Send error response to frontend
                    response = {
                        "type": "start_simulation_response",
                        "success": False,
                        "error": f"Failed to load RL model: {str(e)}"
                    }
                    await websocket.send(json.dumps(response))
                    return
            else:
                debug(f"모델 파일을 찾을 수 없습니다: {model_path}")
                # Send error response to frontend
                response = {
                    "type": "start_simulation_response",
                    "success": False,
                    "error": f"RL model file not found: {model_path}"
                }
                await websocket.send(json.dumps(response))
                return
        
        # Calculate start time
        schedules = self.simulation.schedules
        min_etd = min([s.flight.etd for s in schedules]) if schedules else 0
        start_time = max(360, min_etd - 20)  # 최소 0600, 또는 첫 비행 ETD - 20분
        
        # Start the simulation in a separate thread
        def start_sim():
            self.simulation.start(start_time=start_time)
        
        import threading
        sim_thread = threading.Thread(target=start_sim, daemon=True)
        sim_thread.start()
        
        self.simulation_started = True
        
        # Send confirmation to frontend
        response = {
            "type": "start_simulation_response",
            "success": True,
            "algorithm": algorithm,
            "start_time": start_time
        }
        await websocket.send(json.dumps(response))
        debug("Simulation started successfully")

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

        async def run_server():
            server = await websockets.serve(
                self.handler,
                self.host,
                self.port
            )
            debug("WebSocket server started")
            await asyncio.Future()  # run forever

        self.loop.run_until_complete(run_server())