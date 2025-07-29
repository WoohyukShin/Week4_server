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
    end_time = None  # 시뮬레이션에서 자동 결정

    use_ws = False  # WebSocket
    is_training = False  # Training 여부 - True면 디버깅 X
    # RL 모델 사용 설정
    use_rl = True  # RL 모델 사용 여부
    train_rl = False  # RL 훈련 여부

    # RL 훈련 실행
    if train_rl:
        print("PPO 훈련을 시작합니다...")
        from train_rl import train_rl_with_real_simulation
        best_model_path, training_history = train_rl_with_real_simulation(episodes=150)
        print(f"훈련 완료! 최고 모델: {best_model_path}")
        return

    # 일반 시뮬레이션 실행 (훈련된 모델 사용)
    if is_training:
        mode = SimulationMode.TRAINING
    elif use_ws:
        mode = SimulationMode.INTERACTIVE
    else:
        mode = SimulationMode.HEADLESS

    sim = Simulation(airport, schedules, landing_flights, events=events, mode=mode)
    
    # 훈련된 RL 모델 사용
    if use_rl:
        sim.scheduler.algorithm = "rl"
        
        # 특정 모델 파일 지정
        model_path = "models/ppo_best_second.pth"
        if os.path.exists(model_path):
            from rl.agent import PPOAgent
            # RL 에이전트 초기화 및 모델 로드
            observation_size = 160
            action_size = 288
            rl_agent = PPOAgent(observation_size=observation_size, action_size=action_size)
            rl_agent.load_model(model_path)
            sim.set_rl_agent(rl_agent)
            print(f"훈련된 RL 모델을 로드했습니다: {model_path}")
        else:
            print(f"모델 파일을 찾을 수 없습니다: {model_path}")
            return

    # 시뮬레이션 실행
    if mode == SimulationMode.INTERACTIVE:
        from sim.ws_server import WebSocketServer
        ws_server = WebSocketServer(sim)
        ws_thread = threading.Thread(target=ws_server.start, daemon=True)
        ws_thread.start()
        sim.ws = ws_server
        # Don't start simulation immediately - wait for start message from frontend
        debug("WebSocket server started. Waiting for start message from frontend...")
        # Keep the main thread alive
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            debug("Shutting down...")
    else:
        sim.start(start_time=start_time)

if __name__ == "__main__":
    main()
