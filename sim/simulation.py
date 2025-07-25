from enum import Enum
import threading
import time
from sim.event import Event
from utils.logger import debug
from sim.scheduler import Scheduler
from utils.time_utils import int_to_hhmm
from sim.flight import FlightStatus
from sim.schedule import Schedule
from sim.event_handler import EventHandler

class SimulationMode(Enum):
    INTERACTIVE = "interactive"   # WebSocket 연결, 시각화/수동 이벤트
    HEADLESS = "headless"         # 알고리즘/단순 시뮬레이션
    TRAINING = "training"         # 강화학습용 빠른 반복

DEFAULT_TAKEOFF_RUNWAY = "14L"
DEFAULT_LANDING_RUNWAY = "14R"

class Simulation:
    def __init__(self, airport, flights, events=None, ws=None, mode=None,
                 takeoff_runway=DEFAULT_TAKEOFF_RUNWAY, landing_runway=DEFAULT_LANDING_RUNWAY):
        self.airport = airport
        self.flights = flights
        self.schedules = []
        self.completed_schedules = []
        self.events = events if events else []
        self.ws = ws
        self.time = 0
        self.running = False
        self.lock = threading.Lock()
        self.event_queue = list(self.events)
        self.mode = mode
        self.scheduler = None
        self.takeoff_runway = takeoff_runway
        self.landing_runway = landing_runway
        self.event_handler = EventHandler(self)
        self.initialize_schedules()

    def debug(self, msg):
        print(f"[DEBUG] Simulation : {msg}")

    def initialize_schedules(self):
        self.schedules = [Schedule(f) for f in self.flights]
        self.completed_schedules = []

    def start(self, start_time=0, end_time=None):
        self.time = start_time
        self.running = True
        self.debug(f"시뮬레이션 시작, time={self.time}, mode={self.mode}")
        end_buffer = 5
        end_time_actual = None
        while self.running:
            self.update_status()
            self.send_state_update()
            self.handle_events()
            if self.mode == SimulationMode.INTERACTIVE:
                time.sleep(1)
            self.time += 1
            # 완료된 뒤 3 timestep이 지난 스케줄은 schedules에서 제거하고 completed_schedules로 이동
            to_remove = []
            for s in self.schedules:
                if hasattr(s, 'complete_time') and self.time - s.complete_time >= 3:
                    self.completed_schedules.append(s)
                    to_remove.append(s)
            for s in to_remove:
                self.schedules.remove(s)
            # 종료 조건: schedules가 비었으면 5분 후 종료
            if not self.schedules:
                if end_time_actual is None:
                    end_time_actual = self.time + end_buffer
                elif self.time >= end_time_actual:
                    self.debug(f"시뮬레이션 종료, time={self.time}")
                    self.running = False
            if end_time is not None and self.time >= end_time:
                self.debug(f"시뮬레이션 종료(설정된 종료 시각), time={self.time}")
                self.running = False

    def stop(self):
        self.running = False
        debug("시뮬레이션 중지")

    def update_status(self):
        self.debug(f"상태 갱신, time={self.time}")
        # RWY, TWY 점유 상태 갱신 (생략)
        for runway in self.airport.runways:
            if runway.closed:
                if runway.next_available_time >= self.time:
                    runway.closed = False
                    runway.next_available_time = 0
                else:
                    continue
            runway.occupied = runway.next_available_time > self.time
        for taxiway in self.airport.taxiways:
            if taxiway.closed:
                continue
            if taxiway.name == self.takeoff_taxiway:
                taxiway.occupied = any(s.status == FlightStatus.TAXI_TO_RUNWAY for s in self.schedules)
            else:
                taxiway.occupied = False
        # 스케줄 상태 갱신 및 완료 처리
        for s in self.schedules:
            if not hasattr(s, 'complete_time'):
                self._update_schedule_status(s)
                # 완료 조건: taxiToGate 1 timestep 후 등에서 직접 complete_time 기록
                if s.status in [FlightStatus.DORMANT, FlightStatus.CANCELLED]:
                    if not hasattr(s, 'complete_time'):
                        s.complete_time = self.time
            else:
                # 완료된 후 3분 대기 후 completed_schedules로 이동
                if self.time - s.complete_time >= 3 and s not in self.completed_schedules:
                    self.completed_schedules.append(s)

    def _update_schedule_status(self, schedule):
        f = schedule.flight
        prev_status = schedule.status
        # FlightStatus: taxiToRunway, taxiToGate, waiting, takeOff, landing, delayed, dormant, cancelled
        if schedule.status == FlightStatus.DORMANT:
            if not self._taxiway_blocked(self.takeoff_taxiway):
                schedule.status = FlightStatus.TAXI_TO_RUNWAY
                schedule.location = self.takeoff_taxiway
                schedule.start_taxi_time = self.time
        elif schedule.status == FlightStatus.TAXI_TO_RUNWAY:
            if self.time - schedule.start_taxi_time >= 15:
                if self._can_takeoff(schedule):
                    schedule.status = FlightStatus.TAKE_OFF
                    schedule.location = self.takeoff_runway
                    schedule.takeoff_time = self.time
                    self._occupy_runway(self.takeoff_runway, cooldown=3)
                else:
                    pass
        elif schedule.status == FlightStatus.TAKE_OFF:
            if self.time - schedule.takeoff_time >= 1:
                schedule.status = FlightStatus.WAITING
                schedule.location = None
        elif schedule.status == FlightStatus.WAITING:
            if self._can_land(schedule):
                schedule.status = FlightStatus.LANDING
                schedule.landing_time = self.time
                schedule.location = self.landing_runway
                self._occupy_runway(self.landing_runway, cooldown=3)
        elif schedule.status == FlightStatus.LANDING:
            if self.time - schedule.landing_time >= 1:
                schedule.status = FlightStatus.TAXI_TO_GATE
                schedule.taxi_to_gate_time = self.time
                schedule.location = "taxiToGate"
        elif schedule.status == FlightStatus.TAXI_TO_GATE:
            if self.time - schedule.taxi_to_gate_time >= 1:
                schedule.status = FlightStatus.DORMANT
                schedule.location = "Gate"
        if schedule.status != prev_status:
            self.debug(f"{f.flight_id} : {prev_status.value} → {schedule.status.value} (time={self.time})")

    def _taxiway_blocked(self, name):
        twy = next((t for t in self.airport.taxiways if t.name == name), None)
        return twy.occupied if twy else True

    def _can_takeoff(self, schedule):
        rw = next((r for r in self.airport.runways if r.name == self.takeoff_runway), None)
        return rw and not rw.occupied and not rw.closed and rw.next_available_time <= self.time

    def _can_land(self, schedule):
        rw = next((r for r in self.airport.runways if r.name == self.landing_runway), None)
        return rw and not rw.occupied and not rw.closed and rw.next_available_time <= self.time

    def _occupy_runway(self, name, cooldown=3):
        rw = next((r for r in self.airport.runways if r.name == name), None)
        if rw:
            rw.occupied = True
            rw.next_available_time = self.time + 1 + cooldown

    def _update_runway_roles_on_closure(self):
        # 열려있는 활주로만 추출
        open_runways = [r for r in self.airport.runways if not r.closed]
        if len(open_runways) == 1:
            self.takeoff_runway = open_runways[0].name
            self.landing_runway = open_runways[0].name
        elif len(open_runways) == 2:
            self._restore_default_runway_roles()

    def _restore_default_runway_roles(self):
        # invert 상태에 따라 14L/14R 또는 32R/32L로 복구
        runway_names = {r.name for r in self.airport.runways}
        if {"14L", "14R"}.issubset(runway_names):
            self.takeoff_runway = "14L"
            self.landing_runway = "14R"
        elif {"32R", "32L"}.issubset(runway_names):
            self.takeoff_runway = "32R"
            self.landing_runway = "32L"

    def do_action(self):
        debug("스케줄 최적화 알고리즘 실행")
        # 알고리즘(ML, RL, Greedy 등)으로 스케줄링/최적화
        self.scheduler.optimize(method="greedy")

    def handle_events(self):
        self.debug("이벤트 핸들링")
        triggered = [e for e in self.event_queue if e.time == self.time]
        for event in triggered:
            self.event_handler.handle(event, self.time)
            self.event_queue.remove(event)

    def send_state_update(self):
        debug("state_update 프론트로 전송")
        if self.ws and self.mode == SimulationMode.INTERACTIVE:
            state = self.get_state()
            self.ws.send(state)

    def get_state(self):
        time = int_to_hhmm(self.time)
        def status_to_str(s):
            return s.value
        flights = [self.schedule_to_flight_dict(s, status_to_str) for s in self.schedules]
        return {
            "type": "state_update",
            "time": time,
            "flights": flights
        }

    def schedule_to_flight_dict(self, schedule, status_to_str):
        f = schedule.flight
        # runway: 이륙이면 takeoff_runway, 착륙이면 landing_runway
        if schedule.status == FlightStatus.TAKE_OFF:
            runway = self.takeoff_runway
        elif schedule.status == FlightStatus.LANDING:
            runway = self.landing_runway
        else:
            runway = None
        return {
            "flight_id": f.flight_id,
            "status": status_to_str(schedule.status),
            "ETA": f.eta,
            "ETD": f.etd,
            "depAirport": f.dep_airport,
            "arrivalAirport": f.arr_airport,
            "airline": f.airline,
            "runway": runway
        }

    def on_event(self, event):
        self.debug(f"프론트에서 이벤트 수신: {event}")
        # 프론트에서 온 이벤트 처리
        # event dict를 Event 객체로 변환
        class E: pass
        e = E()
        e.event_type = event['event_type']
        e.target = event['target']
        e.duration = event.get('duration', 0)
        e.time = self.time
        self.event_queue.append(e)
