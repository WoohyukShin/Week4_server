from enum import Enum
import threading
import time
from sim.event import Event
from utils.logger import debug
from sim.scheduler import Scheduler
from utils.time_utils import int_to_hhmm, int_to_hhmm_colon, int_to_hhmm_str
from sim.flight import FlightStatus
from sim.schedule import Schedule
from sim.event_handler import EventHandler
import random

class SimulationMode(Enum):
    INTERACTIVE = "interactive"   # WebSocket 연결, 시각화/수동 이벤트
    HEADLESS = "headless"         # 알고리즘/단순 시뮬레이션
    TRAINING = "training"         # 강화학습용 빠른 반복

class Simulation:
    def __init__(self, airport, schedules, landing_flights, events=None, ws=None, mode=None):
        self.airport = airport
        self.schedules = schedules  # 이륙 스케줄만 가지고 시작
        self.landing_flights = landing_flights  # 착륙 flight 리스트
        self.completed_schedules = []
        self.events = events if events else []
        self.ws = ws
        self.time = 0
        self.running = False
        self.lock = threading.Lock()
        self.event_queue = list(self.events)
        self.mode = mode
        self.event_handler = EventHandler(self)
        self.scheduler = Scheduler("greedy", self)
        
        # Speed control
        self.speed = 1  # 1x, 2x, 4x, 8x
        self.speed_intervals = {
            1: 24,   # 1x: 24 seconds per sim minute
            2: 12,   # 2x: 12 seconds per sim minute  
            4: 6,    # 4x: 6 seconds per sim minute
            8: 3     # 8x: 3 seconds per sim minute
        }
        
        # Loss 계산을 위한 속성들
        self.total_delay_loss = 0
        
        self.initialize_schedules()
        self._init_landing_announce_events()

    def initialize_schedules(self):
        # schedules는 이미 Schedule 객체 리스트로 전달됨
        self.completed_schedules = []

    def start(self, start_time=0, end_time=None):
        self.time = start_time
        self.running = True
        debug(f"시뮬레이션 시작, time={int_to_hhmm_colon(self.time)}, mode={self.mode}, speed={self.speed}x")
        
        # 시뮬레이션 시작 시 초기 액션 수행
        self.do_action()
        
        end_buffer = 5
        end_time_actual = None
        while self.running:
            self.update_status()
            self.send_state_update()
            self.handle_events()
            if self.mode == SimulationMode.INTERACTIVE:
                sleep_interval = self.speed_intervals.get(self.speed, 24)
                time.sleep(sleep_interval)
            self.time += 1
            # 완료된 뒤 3 timestep이 지난 스케줄은 schedules에서 제거하고 completed_schedules로 이동
            to_remove = []
            for s in self.schedules:
                if hasattr(s, 'complete_time') and self.time - s.complete_time >= 3:
                    self.completed_schedules.append(s)
                    to_remove.append(s)
            for s in to_remove:
                self.schedules.remove(s)
            # 종료 조건: schedules & event_queue 가 모두 비었으면 5분 후 종료
            if not self.schedules and not self.event_queue:
                if end_time_actual is None:
                    end_time_actual = self.time + end_buffer
                elif self.time >= end_time_actual:
                    debug(f"시뮬레이션 종료, time={int_to_hhmm_colon(self.time)}")
                    
                    # Loss 출력
                    debug(f"=== 최종 Loss 결과 ===")
                    debug(f"DELAY TERM: {self.total_delay_loss}")
                    debug(f"TOTAL LOSS: {self.get_total_loss()}")
                    debug(f"=====================")
                    
                    self.running = False
            if end_time is not None and self.time >= end_time:
                debug(f"시뮬레이션 종료(설정된 종료 시각), time={int_to_hhmm_colon(self.time)}")
                self.running = False

    def stop(self):
        self.running = False
        debug("시뮬레이션 중지")

    def update_status(self):
        debug(f"status updated at TIME: {int_to_hhmm_colon(self.time)} | 현재 스케즐 수 : {len(self.schedules)}")
        # 대충 활주로 관리
        for runway in self.airport.runways:
            if runway.closed:
                if runway.next_available_time >= self.time:
                    runway.closed = False
                    runway.next_available_time = 0
                else:
                    continue
            runway.occupied = runway.next_available_time > self.time
        # 스케줄 상태 갱신 및 완료 처리
        for s in self.schedules:
            if not hasattr(s, 'complete_time'):
                self._update_schedule_status(s)
                # 완료 조건: 실제 완료된 상태일 때만 complete_time 기록
                if s.status in [FlightStatus.CANCELLED]:
                    if not hasattr(s, 'complete_time'):
                        s.complete_time = self.time
                        debug(f"스케줄 완료: {s.flight.flight_id} (status: {s.status.value})")
            else:
                # 완료된 후 3분 대기 후 completed_schedules로 이동
                if self.time - s.complete_time >= 3 and s not in self.completed_schedules:
                    self.completed_schedules.append(s)
                    debug(f"스케줄 제거: {s.flight.flight_id}")

    def _update_schedule_status(self, schedule):
        f = schedule.flight
        prev_status = schedule.status
        
        match schedule.status:
            case FlightStatus.DORMANT:
                # 스케줄 배정 시간 10분 전에 택시 시작
                if schedule.is_takeoff and schedule.etd is not None:
                    taxi_start_time = schedule.etd - 10
                    if self.time >= taxi_start_time:
                        debug(f"{schedule.flight.flight_id} TAXIING TO RUNWAY {schedule.runway.get_current_direction()}")
                        schedule.status = FlightStatus.TAXI_TO_RUNWAY
                        schedule.start_taxi_time = self.time
            case FlightStatus.TAXI_TO_RUNWAY:
                # 실제 배정된 시간(ETD)에 이륙
                if schedule.is_takeoff and schedule.etd is not None and self.time >= schedule.etd:
                    if self._can_takeoff(schedule):
                        debug(f"{schedule.flight.flight_id} TAKING OFF ON RUNWAY {schedule.runway.get_current_direction()}")
                        schedule.status = FlightStatus.TAKE_OFF
                        schedule.takeoff_time = self.time
                        schedule.atd = self.time  # 실제 이륙 시간 기록
                        # 계획된 활주로 점유
                        if schedule.runway:
                            self._occupy_runway(schedule.runway, cooldown=3)
                        # 이륙 지연 손실 계산
                        self._add_delay_loss(schedule, self.time, "takeoff")
            case FlightStatus.TAKE_OFF:
                if self.time - schedule.takeoff_time >= 1:
                    schedule.status = FlightStatus.DORMANT
                    # 이륙 완료 시 complete_time 기록
                    if not hasattr(schedule, 'complete_time'):
                        schedule.complete_time = self.time
                        debug(f"스케줄 완료: {schedule.flight.flight_id} (이륙 완료)")
            case FlightStatus.WAITING:
                # 실제 배정된 시간(ETA)에 착륙
                if not schedule.is_takeoff and schedule.eta is not None and self.time >= schedule.eta:
                    if self._can_land(schedule):
                        runway_direction = schedule.runway.get_current_direction() if schedule.runway else "Unknown"
                        debug(f"{schedule.flight.flight_id} LANDING ON RUNWAY {runway_direction}")
                        schedule.status = FlightStatus.LANDING
                        schedule.landing_time = self.time
                        # 계획된 활주로 점유
                        if schedule.runway:
                            self._occupy_runway(schedule.runway, cooldown=3)
            case FlightStatus.LANDING:
                if self.time - schedule.landing_time >= 1:
                    schedule.status = FlightStatus.TAXI_TO_GATE
                    runway_direction = schedule.runway.get_current_direction() if schedule.runway else "Unknown"
                    debug(f"{schedule.flight.flight_id} TAXIING TO GATE {runway_direction}")
                    schedule.taxi_to_gate_time = self.time
                    schedule.location = "Gate"
                    schedule.ata = self.time  # 실제 착륙 시간 기록
                    # 착륙 지연 손실 계산
                    self._add_delay_loss(schedule, self.time, "landing")
            case FlightStatus.TAXI_TO_GATE:
                if self.time - schedule.taxi_to_gate_time >= 1:
                    schedule.status = FlightStatus.DORMANT
                    # 착륙 완료 시 complete_time 기록
                    if not hasattr(schedule, 'complete_time'):
                        schedule.complete_time = self.time
                        debug(f"스케줄 완료: {schedule.flight.flight_id} (착륙 완료)")
        
        if schedule.status != prev_status:
            debug(f"{f.flight_id} : {prev_status.value} → {schedule.status.value} (time={int_to_hhmm_colon(self.time)})")
            # 상태 변경이 있을 때 액션 재수행 (이륙/착륙 완료 시에만)
            if schedule.status in [FlightStatus.TAKE_OFF, FlightStatus.LANDING]:
                debug("스케줄 상태 변경 후 액션 재수행")
                self.do_action()

    def _can_takeoff(self, schedule):
        """이륙 가능한 활주로 찾기"""
        for runway in self.airport.runways:
            if runway.closed:
                continue
            # 이륙용 활주로인지 확인 (14L 또는 32R)
            current_direction = runway.get_current_direction()
            if current_direction in ["14L", "32R"]:
                if runway.can_handle_operation(self.time):
                    return True
        return False
    
    def _can_land(self, schedule):
        """착륙 가능한 활주로 찾기"""
        for runway in self.airport.runways:
            if runway.closed:
                continue
            # 착륙용 활주로인지 확인 (14R 또는 32L)
            current_direction = runway.get_current_direction()
            if current_direction in ["14R", "32L"]:
                if runway.can_handle_operation(self.time):
                    return True
        return False
    
    def _get_available_runway(self, operation_type):
        """작업 타입에 맞는 사용 가능한 활주로 반환"""
        for runway in self.airport.runways:
            if runway.closed:
                continue
            if not runway.can_handle_operation(self.time):
                continue
                
            current_direction = runway.get_current_direction()
            match operation_type:
                case "takeoff" if current_direction in ["14L", "32R"]:
                    return runway
                case "landing" if current_direction in ["14R", "32L"]:
                    return runway
        return None
    
    def _occupy_runway(self, runway, cooldown=3):
        """활주로 점유"""
        runway.occupied = True
        runway.next_available_time = self.time + 1 + cooldown
        
    def _restore_default_runway_roles(self):
        """기본 활주로 역할로 복구 - 모든 활주로의 inverted = False"""
        for runway in self.airport.runways:
            runway.inverted = False

    def do_action(self):
        """현재 선택된 알고리즘으로 액션 수행"""
        debug("알고리즘 액션 수행")
        
        # 현재 스케줄 상태와 미완료 이벤트를 알고리즘에 전달
        changes = self.scheduler.optimize(self.schedules, self.time, self.event_queue)
        
        # 변경사항을 스케줄에 적용
        if changes:
            change_list = []
            for flight_id, new_time in changes.items():
                schedule = next((s for s in self.schedules if s.flight.flight_id == flight_id), None)
                if schedule:
                    if schedule.is_takeoff:
                        schedule.etd = new_time
                    else:
                        schedule.eta = new_time

    def handle_events(self):
        triggered = [e for e in self.event_queue if e.time == self.time]
        events_handled = False
        for event in triggered:
            debug("handling event...")
            self.event_handler.handle(event, self.time)
            self.event_queue.remove(event)
            events_handled = True
        
        # 이벤트가 처리되었으면 액션 재수행
        if events_handled:
            debug("이벤트 처리 후 액션 재수행")
            self.do_action()

    def send_state_update(self):
        if self.ws and self.mode == SimulationMode.INTERACTIVE:
            state = self.get_state()
            self.ws.send(state)

    def get_state(self):
        time_str = int_to_hhmm_colon(self.time)  # Returns "HH:MM" string format
        def status_to_str(s):
            return s.value
        flights = [self.schedule_to_flight_dict(s, status_to_str) for s in self.schedules]
        return {
            "type": "state_update",
            "time": time_str,
            "flights": flights,
            "speed": self.speed,
            "timestamp": time.time()  # Add timestamp for synchronization
        }

    def schedule_to_flight_dict(self, schedule, status_to_str):
        f = schedule.flight
        return {
            "flight_id": f.flight_id,
            "status": status_to_str(schedule.status),
            "ETA": int_to_hhmm_colon(schedule.eta) if schedule.eta is not None else None,
            "ETD": int_to_hhmm_colon(schedule.etd) if schedule.etd is not None else None,
            "depAirport": f.dep_airport,
            "arrivalAirport": f.arr_airport,
            "airline": f.airline,
            "runway": schedule.runway.get_current_direction() if (hasattr(schedule, 'runway') and schedule.runway and hasattr(schedule.runway, 'get_current_direction')) else None
        }

    def on_event(self, event):
        debug(f"프론트에서 이벤트 수신: {event}")
        # 프론트에서 온 이벤트 처리
        # event dict를 Event 객체로 변환
        class E: pass
        e = E()
        e.event_type = event['event_type']
        e.target = event['target']
        e.duration = event.get('duration', 0)
        e.time = self.time
        self.event_queue.append(e)

    def _init_landing_announce_events(self):
        for flight in self.landing_flights:
            noise = int(random.gauss(0, 10))  # 표준편차 10분
            announce_time = max(0, (flight.eta or 0) + noise)
            self.event_queue.append(
                type('Event', (), {
                    'event_type': 'LANDING_ANNOUNCE',
                    'target': flight.flight_id,
                    'duration': 20,
                    'time': announce_time
                })()
            )

    def _add_delay_loss(self, schedule, actual_time, operation_type):
        """지연 손실 계산 및 누적 (정규화된 priority 기반 weighted sum)"""
        match operation_type:
            case "takeoff":
                original_time = schedule.flight.etd
                if original_time is None:
                    return  # ETD가 없으면 지연 계산 불가
                delay = actual_time - original_time
                if delay > 0:
                    # 정규화된 priority 사용 (0-1 범위)
                    normalized_priority = schedule.get_normalized_priority()
                    loss = normalized_priority * delay * 100  # 스케일 조정을 위해 100 곱함
                    self.total_delay_loss += loss
                    debug(f"이륙 지연 손실: {schedule.flight.flight_id} {delay}분 지연, priority {schedule.priority} (정규화: {normalized_priority:.2f}) -> {loss:.1f} 손실 추가 (총 {self.total_delay_loss:.1f})")
            case "landing":
                original_time = schedule.flight.eta
                if original_time is None:
                    return  # ETA가 없으면 지연 계산 불가
                delay = actual_time - original_time
                if delay > 0:
                    # 정규화된 priority 사용 (0-1 범위)
                    normalized_priority = schedule.get_normalized_priority()
                    loss = normalized_priority * delay * 100  # 스케일 조정을 위해 100 곱함
                    self.total_delay_loss += loss
                    debug(f"착륙 지연 손실: {schedule.flight.flight_id} {delay}분 지연, priority {schedule.priority} (정규화: {normalized_priority:.2f}) -> {loss:.1f} 손실 추가 (총 {self.total_delay_loss:.1f})")
    
    def _add_go_around_loss(self, schedule):
        """Go-around 손실 계산 및 누적 (정규화된 priority 기반 weighted sum)"""
        # 정규화된 priority 사용 (0-1 범위)
        normalized_priority = schedule.get_normalized_priority()
        loss = normalized_priority * 15 * 100  # 15분 지연, 스케일 조정을 위해 100 곱함
        self.total_delay_loss += loss
        debug(f"Go-around 손실: {schedule.flight.flight_id} 15분 지연, priority {schedule.priority} (정규화: {normalized_priority:.2f}) -> {loss:.1f} 손실 추가 (총 {self.total_delay_loss:.1f})")
    

    
    def get_total_loss(self):
        """총 손실 반환"""
        return self.total_delay_loss

    def set_speed(self, speed):
        """Change simulation speed (1x, 2x, 4x, 8x)"""
        if speed in [1, 2, 4, 8]:
            old_speed = self.speed
            self.speed = speed
            debug(f"시뮬레이션 속도 변경: {old_speed}x → {speed}x")
            return True
        else:
            debug(f"잘못된 속도 설정: {speed}. 가능한 값: 1, 2, 4, 8")
            return False
