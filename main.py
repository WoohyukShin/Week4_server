from sim.airport import Airport, Runway, Taxiway
from sim.simulation import Simulation, SimulationMode
from utils.logger import debug
from sim.ws_server import WebSocketServer
from utils.scenario_loader import load_scenario, generate_random_scenario, save_random_scenario
import threading
import time
import os

# 이륙/착륙용 RUNWAY 상수
TAKEOFF_RUNWAY = "14L"
LANDING_RUNWAY = "14R"

def create_rkss_airport():
    # 14L(32R) = 안쪽 활주로, 14R(32L) = 바깥쪽 활주로
    runways = [Runway("14L", "32R"), Runway("14R", "32L")]
    taxiways = [Taxiway("G2", "B2"), Taxiway("B", "G")]  # G2 <-> B2, B <-> G
    return Airport("RKSS", "GMP", runways, taxiways)

def main():
    # Railway deployment logging
    debug("Starting Air Traffic Control Backend on Railway...")
    debug(f"PORT environment variable: {os.environ.get('PORT', 'Not set')}")
    
    # 랜덤 시나리오 사용
    use_random_scenario = True
    # 시나리오 파일 저장
    save_random_scenario = False
    
    if use_random_scenario:
        debug("랜덤 시나리오를 생성합니다...")
        random_scenario = generate_random_scenario(num_flights=50, num_events=5)
        
        if save_random_scenario:
            scenario_dir = "scenario"
            if not os.path.exists(scenario_dir):
                os.makedirs(scenario_dir)
            save_random_scenario(random_scenario, f"{scenario_dir}/random_scenario.json")
        
        from utils.scenario_loader import load_scenario_from_dict
        schedules, landing_flights, events = load_scenario_from_dict(random_scenario)
    else:
        # 시나리오 파일 사용
        scenario_path = "scenario/sample_scenario.json"
        schedules, landing_flights, events = load_scenario(scenario_path)
    
    airport = create_rkss_airport()

    # 시작/종료 시간 계산
    min_etd = min([s.flight.etd for s in schedules]) if schedules else 0
    start_time = max(360, min_etd - 20)  # 최소 0600, 또는 첫 비행 ETD - 20분
    end_time = 1440  # 24:00 (1440분)으로 고정

    use_ws = False  # WebSocket (디버깅용으로 비활성화)
    is_training = True  # 강화학습 반복용 (랜덤 시나리오 사용 시 True 권장)

    if is_training:
        print("PPO 훈련을 시작합니다...")
        from train_rl import train_rl_with_real_simulation
        best_model_path, training_history = train_rl_with_real_simulation(episodes=150)
        print(f"훈련 완료! 최고 모델: {best_model_path}")
        return

    if is_training:
        mode = SimulationMode.TRAINING
    elif use_ws:
        mode = SimulationMode.INTERACTIVE
    else:
        mode = SimulationMode.HEADLESS

    sim = Simulation(airport, schedules, landing_flights, events=events, mode=mode)

    if mode == SimulationMode.INTERACTIVE:
        from sim.ws_server import WebSocketServer
        # Use Railway's PORT environment variable
        port = int(os.environ.get("PORT", 8765))
        debug(f"Starting WebSocket server on port {port}")
        ws_server = WebSocketServer(sim, port=port)
        ws_thread = threading.Thread(target=ws_server.start, daemon=True)
        ws_thread.start()
        sim.ws = ws_server
        # Don't start simulation immediately - wait for start message from frontend
        debug("WebSocket server started. Waiting for start message from frontend...")
        debug(f"Frontend URL: https://week4-front.vercel.app/")
        # Keep the main thread alive
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            debug("Shutting down...")
    else:
        sim.start(start_time=start_time, end_time=end_time)

if __name__ == "__main__":
    main()
