from sim.airport import Airport, Runway, Taxiway
from sim.simulation import Simulation, SimulationMode
from utils.logger import debug
from sim.ws_server import WebSocketServer
from utils.scenario_loader import load_scenario
import threading
import time

# 이륙/착륙용 RUNWAY 상수
TAKEOFF_RUNWAY = "14L"
LANDING_RUNWAY = "14R"

def create_rkss_airport():
    runways = [Runway("14L"), Runway("14R")]
    taxiways = [Taxiway("G2"), Taxiway("B")]
    return Airport("RKSS", "GMP", runways, taxiways)

def main():
    scenario_path = "scenario/sample_scenario.json"
    flights, events = load_scenario(scenario_path)
    airport = create_rkss_airport()

    # 시작/종료 시간 계산
    min_etd = min(f.etd for f in flights)
    start_time = min_etd - 20
    end_time = None  # 시뮬레이션에서 자동 결정

    use_ws = True  # 시각화/수동 이벤트용
    is_training = False  # 강화학습 반복용

    if is_training:
        mode = SimulationMode.TRAINING
    elif use_ws:
        mode = SimulationMode.INTERACTIVE
    else:
        mode = SimulationMode.HEADLESS

    sim = Simulation(airport, flights, events=events, mode=mode)

    if mode == SimulationMode.INTERACTIVE:
        ws_server = WebSocketServer(sim)
        ws_thread = threading.Thread(target=ws_server.start, daemon=True)
        ws_thread.start()
        sim.ws = ws_server
        sim.start(start_time=start_time)
    else:
        sim.start(start_time=start_time)

if __name__ == "__main__":
    main()
