from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import json
import os
import threading
import time

# Create FastAPI app
app = FastAPI(title="Air Traffic Control Backend")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global simulation instance
sim = None
simulation_started = False

@app.get("/")
async def root():
    return {"message": "Air Traffic Control Backend is running!", "status": "ready"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "simulation_ready": sim is not None}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global simulation_started, sim
    
    await websocket.accept()
    print("클라이언트 연결됨")
    
    # Send connection confirmation
    await websocket.send_text(json.dumps({
        "type": "connection_established",
        "message": "Connected to Railway backend"
    }))
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            print(f"프론트에서 메시지 수신: {data}")
            
            try:
                message_data = json.loads(data)
                
                if message_data.get("type") == "start_simulation":
                    await handle_start_simulation(message_data, websocket)
                elif message_data.get("type") == "reset_simulation":
                    await handle_reset_simulation(websocket)
                elif message_data.get("type") == "event" and sim:
                    sim.on_event(message_data["event"])
                elif message_data.get("type") == "speed_control" and sim:
                    speed = message_data.get("speed", 1)
                    success = sim.set_speed(speed)
                    response = {
                        "type": "speed_control_response",
                        "success": success,
                        "speed": speed if success else sim.speed
                    }
                    await websocket.send_text(json.dumps(response))
                    
            except json.JSONDecodeError:
                print("Invalid JSON received")
                
    except WebSocketDisconnect:
        print("클라이언트 연결 종료")

async def handle_start_simulation(data, websocket):
    global simulation_started, sim
    
    if simulation_started:
        print("Simulation already started")
        response = {
            "type": "start_simulation_response",
            "success": False,
            "error": "Simulation already started. Please reset first."
        }
        await websocket.send_text(json.dumps(response))
        return
    
    try:
        # Import simulation modules only when needed
        from sim.airport import Airport, Runway, Taxiway
        from sim.simulation import Simulation, SimulationMode
        from utils.scenario_loader import generate_random_scenario
        
        # Initialize simulation
        print("Starting Air Traffic Control Backend on Railway...")
        print(f"PORT environment variable: {os.environ.get('PORT', 'Not set')}")
        
        # Generate random scenario
        print("랜덤 시나리오를 생성합니다...")
        random_scenario = generate_random_scenario(num_flights=50, num_events=5)
        
        from utils.scenario_loader import load_scenario_from_dict
        schedules, landing_flights, events = load_scenario_from_dict(random_scenario)
        
        # Create airport
        runways = [Runway("14L", "32R"), Runway("14R", "32L")]
        taxiways = [Taxiway("G2", "B2"), Taxiway("B", "G")]
        airport = Airport("RKSS", "GMP", runways, taxiways)
        
        sim = Simulation(airport, schedules, landing_flights, events=events, mode=SimulationMode.INTERACTIVE)
        print("Simulation initialized successfully")
        
        algorithm = data.get("algorithm", "greedy")
        print(f"Starting simulation with algorithm: {algorithm}")
        
        # Set the algorithm in the simulation
        from sim.scheduler import Scheduler
        sim.scheduler = Scheduler(algorithm, sim)
        
        if algorithm == "rl":
            import os
            model_path = "models/ppo_best_second.pth"
            if os.path.exists(model_path):
                try:
                    from rl.agent import PPOAgent
                    from rl.environment import AirportEnvironment
                    
                    rl_env = AirportEnvironment(sim)
                    observation_size = rl_env.observation_space_size
                    action_size = rl_env.action_space_size
                    
                    rl_agent = PPOAgent(observation_size=observation_size, action_size=action_size)
                    rl_agent.load_model(model_path)
                    sim.set_rl_agent(rl_agent)
                    print(f"훈련된 RL 모델을 로드했습니다: {model_path}")
                except Exception as e:
                    print(f"RL 모델 로드 중 오류 발생: {e}")
                    response = {
                        "type": "start_simulation_response",
                        "success": False,
                        "error": f"Failed to load RL model: {str(e)}"
                    }
                    await websocket.send_text(json.dumps(response))
                    return
        
        # Start simulation in a separate thread
        def start_sim():
            min_etd = min([s.flight.etd for s in sim.schedules]) if sim.schedules else 0
            start_time = max(360, min_etd - 20)
            sim.start(start_time=start_time)
        
        sim_thread = threading.Thread(target=start_sim, daemon=True)
        sim_thread.start()
        
        simulation_started = True
        
        response = {
            "type": "start_simulation_response",
            "success": True,
            "algorithm": algorithm
        }
        await websocket.send_text(json.dumps(response))
        print("Simulation started successfully")
        
    except Exception as e:
        print(f"Error starting simulation: {e}")
        response = {
            "type": "start_simulation_response",
            "success": False,
            "error": f"Failed to start simulation: {str(e)}"
        }
        await websocket.send_text(json.dumps(response))

async def handle_reset_simulation(websocket):
    global simulation_started, sim
    
    print("Resetting simulation")
    simulation_started = False
    if sim:
        sim.reset()
    
    response = {
        "type": "reset_simulation_response",
        "success": True
    }
    await websocket.send_text(json.dumps(response))
    print("Simulation reset successfully") 