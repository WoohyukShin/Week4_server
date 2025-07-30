from enum import Enum
import threading
import time
import json
from sim.event import Event
from utils.logger import debug
from sim.scheduler import Scheduler
from utils.time_utils import int_to_hhmm, int_to_hhmm_colon, int_to_hhmm_str
from sim.flight import FlightStatus
from sim.schedule import Schedule
from sim.event import Event
from sim.event_handler import EventHandler
from sim.weather import Weather
import random
import math
import numpy as np

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
        self.scheduler = Scheduler("greedy", self)  # Use greedy algorithm by default

        self.rl_agent = None  # RL 에이전트
        self.training_mode = False  # 학습 모드
        self.episode_experiences = []  # 경험 저장
        self.episode_count = 0  # 에피소드 카운트
        
        # Speed control
        self.speed = 1  # 1x, 2x, 4x, 8x, 64x
        self.speed_intervals = {
            1: 24,   # 1x: 24 seconds per sim minute
            2: 12,   # 2x: 12 seconds per sim minute  
            4: 6,    # 4x: 6 seconds per sim minute
            8: 3,    # 8x: 3 seconds per sim minute
            64: 0.375 # 64x: 0.375 seconds per sim minute (very fast!)
        }
        
        # Loss 계산을 위한 속성들
        self.total_delay_loss = 0
        self.total_safety_loss = 0
        self.total_simultaneous_ops_loss = 0
        self.total_runway_occupied_loss = 0
        
        # Score 계산을 위한 속성들
        self.delay_scores = []  # 각 이착륙의 delay score 저장
        self.safety_scores = []  # 각 이착륙의 safety score 저장
        
        # Statistics tracking
        self.total_delay_time_weighted = 0.0  # Priority-weighted delay time
        self.total_flights = 0  # 완료될 때마다 1씩 증가
        self.cancelled_flights = 0  # 취소된 비행 수
        self.safety_loss_breakdown = {
            "weather_risk": 0.0,
            "runway_closed": 0.0,
            "runway_occupied": 0.0,
            "simultaneous_ops": 0.0,
            "accidents": 0.0
        }
        
        # Weather system (랜덤 날씨)
        self.weather = Weather()
        
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
            if self.event_queue:
                event_types = [e.event_type for e in self.event_queue]
                debug(f"현재 남은 이벤트들: {len(self.event_queue)}개 - {event_types}")
            self.handle_events()
            if self.mode == SimulationMode.INTERACTIVE:
                sleep_interval = self.speed_intervals.get(self.speed, 24)
                time.sleep(sleep_interval)
            self.time += 1
            # 완료된 뒤 3 timestep이 지난 스케줄은 schedules에서 제거하고 completed_schedules로 이동
            to_remove = []
            for s in self.schedules:
                if hasattr(s, 'complete_time') and self.time - s.complete_time >= 10:
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
                    
                    # Score 출력
                    debug(f"===== 최종 Score 결과 =====")
                    debug(f"DELAY SCORE: {self.get_delay_score():.1f}")
                    debug(f"SAFETY SCORE: {self.get_safety_score():.1f}")
                    debug(f"TOTAL SCORE: {self.get_total_score():.1f}")
                    debug(f"===========================")

                    # Send results to frontend
                    self.send_simulation_results()

                    if self.training_mode and self.rl_agent:
                        final_reward = -self.get_total_loss()  # 손실을 음수 보상으로 변환
                        self._end_rl_episode(final_reward)
                    
                    # 통계 정보 출력
                    stats = self.calculate_statistics()
                    debug(f"TOTAL DELAY TIME (WITH PRIORITY): {stats['total_delay_time_weighted']:.1f}")
                    debug(f"TOTAL FLIGHTS: {stats['total_flights']}")
                    debug(f"CANCELLED FLIGHTS: {self.cancelled_flights}")
                    if stats['total_flights'] > 0:
                        avg_delay = stats['total_delay_time_weighted'] / stats['total_flights']
                        debug(f"AVERAGE DELAY TIME: {avg_delay:.1f}")
                    debug(f"TOTAL SAFETY LOSS: {stats['total_safety_loss']:.1f}")
                    for cause, loss in stats['safety_breakdown'].items():
                        if loss > 0:
                            debug(f"  - {cause}: {loss:.1f}")
                    debug(f"===========================")
                    
                    self.running = False
            if end_time is not None and self.time >= end_time:
                debug(f"시뮬레이션 종료(설정된 종료 시각), time={int_to_hhmm_colon(self.time)}")
                self.running = False

    def stop(self):
        self.running = False
        debug("시뮬레이션 종료")

    def reset(self):
        """Reset simulation to initial state for restarting"""
        debug("Resetting simulation to initial state")
        
        # Stop current simulation
        self.running = False
        
        # Reset time and state
        self.time = 0
        
        # Reset all schedules to initial state
        for schedule in self.schedules:
            schedule.status = FlightStatus.DORMANT
            schedule.actual_time = None
            schedule.runway = None
            schedule.complete_time = None
        
        # Reset landing flights
        for flight in self.landing_flights:
            flight.status = FlightStatus.DORMANT
            flight.actual_time = None
            flight.runway = None
            flight.complete_time = None
        
        # Reset completed schedules
        self.completed_schedules = []
        
        # Reset event queue
        self.event_queue = list(self.events)
        
        # Reset statistics
        self.total_delay_loss = 0
        self.total_safety_loss = 0
        self.total_simultaneous_ops_loss = 0
        self.total_runway_occupied_loss = 0
        self.delay_scores = []
        self.safety_scores = []
        self.total_delay_time_weighted = 0.0
        self.total_flights = 0
        self.cancelled_flights = 0
        self.safety_loss_breakdown = {
            "weather_risk": 0.0,
            "runway_closed": 0.0,
            "runway_occupied": 0.0,
            "simultaneous_ops": 0.0,
            "accidents": 0.0
        }
        
        # Reset weather
        self.weather = Weather()
        
        # Reset RL-related state
        self.episode_experiences = []
        self.episode_count = 0
        
        # Reset speed
        self.speed = 1
        
        # Reinitialize schedules and landing announce events
        self.initialize_schedules()
        self._init_landing_announce_events()
        
        debug("Simulation reset completed")

    def update_status(self):
        debug(f"status updated at TIME: {int_to_hhmm_colon(self.time)} | 현재 스케즐 수 : {len(self.schedules)}")
        
        # 날씨 업데이트
        self.weather.update_weather(self.time)
        
        # 대충 활주로 관리
        for runway in self.airport.runways:
            if runway.closed:
                if runway.next_available_time >= self.time:
                    runway.closed = False
                    runway.next_available_time = 0
                else:
                    continue
            
            # Track runway occupation status changes
            was_occupied = runway.occupied
            runway.occupied = runway.next_available_time > self.time
            
            if was_occupied and not runway.occupied:
                debug(f"🔄 RUNWAY RELEASED: {runway.get_current_direction()} at time {self.time}")
                debug(f"   - next_available_time: {runway.next_available_time}")
            elif not was_occupied and runway.occupied:
                debug(f"🔄 RUNWAY OCCUPIED: {runway.get_current_direction()} at time {self.time}")
                debug(f"   - next_available_time: {runway.next_available_time}")

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
            case FlightStatus.DORMANT | FlightStatus.DELAYED:
                # 스케줄 배정 시간 10분 전에 택시 시작
                if schedule.is_takeoff and schedule.etd is not None:
                    taxi_start_time = schedule.etd - 10
                    if self.time >= taxi_start_time:
                        # Ensure runway is assigned before taxiing
                        if not schedule.runway:
                            assigned_runway = self._get_available_runway("takeoff")
                            if assigned_runway:
                                schedule.runway = assigned_runway
                                debug(f"Assigned runway {assigned_runway.get_current_direction()} to {schedule.flight.flight_id} for takeoff")
                            else:
                                debug(f"WARNING: No runway available for {schedule.flight.flight_id} takeoff")
                        
                        runway_direction = schedule.runway.get_current_direction() if schedule.runway else "Unknown"
                        debug(f"{schedule.flight.flight_id} TAXIING TO RUNWAY {runway_direction}")
                        schedule.status = FlightStatus.TAXI_TO_RUNWAY
                        schedule.start_taxi_time = self.time
                        
                        # Re-optimize to ensure proper separation for remaining flights
                        debug("Flight started taxiing - re-optimizing for separation")
                        self.do_action()
            case FlightStatus.TAXI_TO_RUNWAY:
                # 실제 배정된 시간(ETD)에 이륙 (알고리즘이 이미 결정했으므로 강제 실행)
                if schedule.is_takeoff and schedule.etd is not None and self.time >= schedule.etd:
                    debug(f"{schedule.flight.flight_id} TAKING OFF ON RUNWAY {schedule.runway.get_current_direction() if schedule.runway else 'Unknown'}")
                    schedule.status = FlightStatus.TAKE_OFF
                    schedule.takeoff_time = self.time
                    schedule.atd = self.time  # 실제 이륙 시간 기록
                    # 이륙 지연 손실 계산
                    self._add_delay_loss(schedule, self.time, "takeoff")
                    # 이륙 safety 손실 계산
                    self._add_safety_loss(schedule, "takeoff")
                    # 이륙 사고 확률 체크
                    self._check_accident_probability(schedule, "takeoff")
                    # 위험한 활주로 사용에 대한 추가 loss (이착륙 시작 시점에 체크)
                    self._add_runway_safety_loss(schedule, "takeoff")
            case FlightStatus.TAKE_OFF:
                if self.time - schedule.takeoff_time >= 1:
                    schedule.status = FlightStatus.DORMANT  
                    self.total_flights += 1
                    # 이륙 완료 시 complete_time 기록
                    if not hasattr(schedule, 'complete_time'):
                        schedule.complete_time = self.time
                        debug(f"스케줄 완료: {schedule.flight.flight_id} (이륙 완료)")
            case FlightStatus.WAITING:
                # 실제 배정된 시간(ETA)에 착륙 (알고리즘이 이미 결정했으므로 강제 실행)
                if not schedule.is_takeoff and schedule.eta is not None and self.time >= schedule.eta:
                    # Ensure runway is assigned before landing
                    if not schedule.runway:
                        assigned_runway = self._get_available_runway("landing")
                        if assigned_runway:
                            schedule.runway = assigned_runway
                            debug(f"Assigned runway {assigned_runway.get_current_direction()} to {schedule.flight.flight_id} for landing")
                        else:
                            debug(f"WARNING: No runway available for {schedule.flight.flight_id} landing")
                    
                    runway_direction = schedule.runway.get_current_direction() if schedule.runway else "Unknown"
                    debug(f"{schedule.flight.flight_id} LANDING ON RUNWAY {runway_direction}")
                    schedule.status = FlightStatus.LANDING
                    schedule.landing_time = self.time
            case FlightStatus.LANDING:
                if self.time - schedule.landing_time >= 1:
                    self.total_flights += 1
                    schedule.status = FlightStatus.TAXI_TO_GATE
                    runway_direction = schedule.runway.get_current_direction() if schedule.runway else "Unknown"
                    debug(f"{schedule.flight.flight_id} TAXIING TO GATE {runway_direction}")
                    schedule.taxi_to_gate_time = self.time
                    schedule.location = "Gate"
                    schedule.ata = self.time  # 실제 착륙 시간 기록
                    # 착륙 지연 손실 계산
                    self._add_delay_loss(schedule, self.time, "landing")
                    # 착륙 safety 손실 계산
                    self._add_safety_loss(schedule, "landing")
                    # 착륙 사고 확률 체크
                    self._check_accident_probability(schedule, "landing")
                    # 위험한 활주로 사용에 대한 추가 loss (착륙 시작 시점에 체크)
                    self._add_runway_safety_loss(schedule, "landing")
                    
                    if not hasattr(schedule, 'complete_time'):
                        schedule.complete_time = self.time
            case FlightStatus.TAXI_TO_GATE:
                if self.time - schedule.taxi_to_gate_time >= 10:
                    schedule.status = FlightStatus.DORMANT
        
        if schedule.status != prev_status:
            debug(f"{f.flight_id} : {prev_status.value} → {schedule.status.value} (time={int_to_hhmm_colon(self.time)})")
            # 상태 변경이 있을 때 액션 재수행 (이륙/착륙 완료 시에만)
            if self.scheduler.algorithm != "rl" and schedule.status in [FlightStatus.TAKE_OFF, FlightStatus.LANDING]:
                debug("스케줄 상태 변경 후 액션 재수행")
                self.do_action()
            # Also re-optimize when flights start taxiing to maintain separation
            elif schedule.status == FlightStatus.TAXI_TO_RUNWAY:
                debug("Flight started taxiing - re-optimizing for separation")
                self.do_action()


    
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
        debug(f"🔒 OCCUPYING RUNWAY: {runway.get_current_direction()}")
        debug(f"   - Previous occupied: {runway.occupied}")
        debug(f"   - Previous next_available_time: {runway.next_available_time}")
        debug(f"   - Current time: {self.time}")
        debug(f"   - New next_available_time: {self.time + 1 + cooldown}")
        
        runway.occupied = True
        # runway.next_available_time = self.time + 1 + cooldown
        runway.next_available_time = max(self.time + 1 + cooldown, runway.next_available_time)
        
    def _restore_default_runway_roles(self):
        """기본 활주로 역할로 복구 - 모든 활주로의 inverted = False"""
        for runway in self.airport.runways:
            runway.inverted = False

    def do_action(self):
        """현재 선택된 알고리즘으로 액션 수행"""
        debug("알고리즘 액션 수행")
        
        # 현재 날씨 예보 정보 가져오기
        weather_forecast = self.weather.get_forecast_for_action()
        debug(f"날씨 예보 (현재시간 {self.time}부터 2시간, 5분 간격): {len(weather_forecast)}개 시점")
        
        # 예보 정보 일부 출력 (처음 3개 시점)
        if weather_forecast:
            for i, forecast in enumerate(weather_forecast[:3]):
                debug(f"  {forecast['time']}분: 이륙위험 {forecast['takeoff_risk']}, 착륙위험 {forecast['landing_risk']}")
        
        # Agent에게 전달할 관측 가능한 이벤트만 필터링
        observed_events = self.get_observed_events()
        
        # 각 활주로의 현재 next_available_time 정보 수집
        runway_availability = {}
        for runway in self.airport.runways:
            runway_availability[runway.name] = runway.next_available_time
        
        # 현재 스케줄 상태와 관측 가능한 이벤트를 알고리즘에 전달 (즉시 반영)
        self.scheduler.optimize(
            self.schedules, 
            self.time, 
            event_queue=observed_events, 
            forecast=weather_forecast, 
            runway_availability=runway_availability
        )
        
        if self.training_mode and self.rl_agent:
            immediate_reward = self._calculate_immediate_reward()
            # 실제 액션과 확률을 사용하여 경험 저장
            actions = self.scheduler.get_actions()
            action_probs = self.scheduler.get_action_probs()
            value = self.scheduler.get_value()
            if actions and action_probs:
                self._store_experience_with_actions(immediate_reward, actions, action_probs, value)
            else:
                self._store_experience(immediate_reward)  # 기존 방식으로 fallback
            debug(f"즉시 피드백: {immediate_reward}")
        

    def handle_events(self):
        # 현재 시간과 일치하는 이벤트 처리
        triggered = [e for e in self.event_queue if e.time == self.time]
        events_handled = False
        
        # 이벤트 처리
        for event in triggered:
            debug("handling event...")
            self.event_handler.handle(event, self.time)
            events_handled = True
        
        # 이벤트 완료 여부 확인 및 제거
        self._cleanup_completed_events()
        # 만료된 이벤트들 정리 (무한 루프 방지)
        self._cleanup_expired_events()
        # 이벤트가 처리되었으면 액션 재수행
        if events_handled:
            self.do_action()
    
    def _cleanup_completed_events(self):
        """완료된 이벤트들을 제거"""
        events_to_remove = []
        
        for event in self.event_queue:
                if event.time == self.time:
                    events_to_remove.append(event)
        
        for event in events_to_remove:
            if event in self.event_queue:
                self.event_queue.remove(event)
    
    def _cleanup_expired_events(self):
        """만료된 이벤트들을 정리 (무한 루프 방지)"""
        events_to_remove = []
        
        for event in self.event_queue:
            if hasattr(event, 'duration') and event.duration > 0:
                if event.time + event.duration < self.time:
                    events_to_remove.append(event)
        
        for event in events_to_remove:
            if event in self.event_queue:
                self.event_queue.remove(event)

    def send_state_update(self):
        if self.ws and self.mode == SimulationMode.INTERACTIVE:
            state = self.get_state()
            self.ws.send(state)

    def get_state(self):
        time_str = int_to_hhmm_colon(self.time)  # Returns "HH:MM" string format
        def status_to_str(s):
            return s.value
        flights = [self.schedule_to_flight_dict(s, status_to_str) for s in self.schedules]
        
        # 날씨 정보를 요청된 형식으로 변환
        weather_info = self.weather.get_detailed_weather_info()
        weather_array = [{
            "condition": weather_info["condition"],
            "visibility": weather_info["visibility"],
            "wind_speed": weather_info["wind_speed"],
            "wind_direction": weather_info["wind_direction"],
            "temperature": weather_info["temperature"],
            "pressure": weather_info["pressure"]
        }]
        
        return {
            "type": "state_update",
            "time": time_str,
            "flights": flights,
            "speed": self.speed,
            "weather": weather_array,
            "timestamp": time.time()  # Add timestamp for synchronization
        }

    def schedule_to_flight_dict(self, schedule, status_to_str):
        f = schedule.flight
        
        # Ensure runway is assigned for all flight states
        runway_direction = None
        if hasattr(schedule, 'opposite_runway_direction'):
            runway_direction = schedule.opposite_runway_direction
        elif hasattr(schedule, 'runway') and schedule.runway and hasattr(schedule.runway, 'get_current_direction'):
            runway_direction = schedule.runway.get_current_direction()
        else:
            # Fallback: assign runway based on operation type and status
            runway_direction = self._get_runway_for_schedule(schedule)
        
        return {
            "flight_id": f.flight_id,
            "status": status_to_str(schedule.status),
            "ETA": int_to_hhmm_colon(schedule.eta) if schedule.eta is not None else None,
            "ETD": int_to_hhmm_colon(schedule.etd) if schedule.etd is not None else None,
            "depAirport": f.dep_airport,
            "arrivalAirport": f.arr_airport,
            "airline": f.airline,
            "runway": runway_direction
        }
    
    def _get_runway_for_schedule(self, schedule):
        """Get appropriate runway for a schedule based on operation type and status"""
        # First try to get from assigned runway
        if hasattr(schedule, 'runway') and schedule.runway:
            return schedule.runway.get_current_direction()
        
        # Fallback based on operation type
        if schedule.is_takeoff:
            # Takeoff: prefer 14L/32R
            for runway in self.airport.runways:
                current_direction = runway.get_current_direction()
                if current_direction in ["14L", "32R"] and not runway.closed:
                    return current_direction
            # Fallback to 14R/32L
            for runway in self.airport.runways:
                current_direction = runway.get_current_direction()
                if current_direction in ["14R", "32L"] and not runway.closed:
                    return current_direction
        else:
            # Landing: prefer 14R/32L
            for runway in self.airport.runways:
                current_direction = runway.get_current_direction()
                if current_direction in ["14R", "32L"] and not runway.closed:
                    return current_direction
            # Fallback to 14L/32R
            for runway in self.airport.runways:
                current_direction = runway.get_current_direction()
                if current_direction in ["14L", "32R"] and not runway.closed:
                    return current_direction
        
        # If all else fails, return the first available runway direction
        for runway in self.airport.runways:
            if not runway.closed:
                return runway.get_current_direction()
        
        # Last resort fallback
        return "14L"

    def on_event(self, event):
        debug(f"프론트에서 이벤트 수신: {event}")

        # 즉시 핸들링
        class E: pass
        e = E()
        e.event_type = event['event_type']
        e.target = event['target']
        e.duration = event.get('duration', 0)
        e.time = self.time
        
        self.event_handler.handle(e, self.time)
        
        # 즉시 액션 재수행
        debug("프론트 이벤트 즉시 처리 후 액션 재수행")
        self.do_action()

    def _init_landing_announce_events(self):
        for flight in self.landing_flights:
            # ETA가 None인 경우 기본값 사용
            if flight.eta is None:
                continue  # ETA가 없으면 이벤트 생성하지 않음
            
            noise = int(random.gauss(0, 20))  # 표준편차 20분
            announce_time = max(360, min(1320, flight.eta + noise - 20))  # 0600~2200 범위로 제한
            self.event_queue.append(
                type('Event', (), {
                    'event_type': 'LANDING_ANNOUNCE',
                    'target': flight.flight_id,
                    'duration': 20,
                    'time': announce_time
                })()
            )

    def _add_delay_loss(self, schedule, actual_time, operation_type):
        """지연 손실 계산 및 누적 (Logistic decay 기반 score)"""
        def logistic_decay_score(delay_min, priority):
            # 중간 priority (32) 기준으로 60분 지연 시 0점이 되도록 설정
            # PRI_MAX (64)는 32보다 약 1.5배 빠르게 감소하도록 조정
            
            # Priority를 0-1 범위로 정규화 (32를 기준으로)
            normalized_priority = priority / 32.0
            
            # Logistic decay 함수: 100점에서 시작해서 지연에 따라 감소
            # 중간 priority (32) 기준 60분 지연 시 0점
            decay_rate = 0.1 * normalized_priority  # Priority가 높을수록 빠르게 감소
            score = 100 / (1 + math.exp(decay_rate * (delay_min - 60)))
            
            return score
        
        match operation_type:
            case "takeoff":
                original_time = schedule.flight.etd
                if original_time is None:
                    return  # ETD가 없으면 지연 계산 불가
                delay = actual_time - original_time
                if delay > 0:
                    # Logistic decay 기반 delay score 계산
                    score = logistic_decay_score(delay, schedule.priority)
                    self.delay_scores.append(score)
                    debug(f"이륙 지연 score: {schedule.flight.flight_id} {delay}분 지연, priority {schedule.priority} -> score {score:.1f}")
                    
                    # Delay minutes 기록 (통계용)
                    schedule.delay_minutes = delay
            case "landing":
                original_time = schedule.flight.eta
                if original_time is None:
                    return  # ETA가 없으면 지연 계산 불가
                delay = actual_time - original_time
                if delay > 0:
                    # Logistic decay 기반 delay score 계산
                    score = logistic_decay_score(delay, schedule.priority)
                    self.delay_scores.append(score)
                    debug(f"착륙 지연 score: {schedule.flight.flight_id} {delay}분 지연, priority {schedule.priority} -> score {score:.1f}")
                    
                    # Delay minutes 기록 (통계용)
                    schedule.delay_minutes = delay
    
    def _add_go_around_loss(self, schedule):
        """Go-around 손실 계산 및 누적 (priority 고려하지 않음)"""
        # Priority 고려하지 않고 고정 손실
        loss = 15 * 100  # 15분 지연, 스케일 조정을 위해 100 곱함
        self.total_delay_loss += loss
        debug(f"Go-around 손실: {schedule.flight.flight_id} 15분 지연 -> {loss:.1f} 손실 추가 (총 {self.total_delay_loss:.1f})")
    
    def _add_safety_loss(self, schedule, operation_type):
        """날씨 기반 safety score 계산 (안전한 비행 시에는 큰 차이 없도록)"""
        if operation_type == "takeoff":
            risk_multiplier = self.weather.takeoff_risk_multiplier
        else:  # landing
            risk_multiplier = self.weather.landing_risk_multiplier
        
        # 기본 safety score (날씨가 좋을 때는 거의 만점)
        base_safety_score = 95
        
        # 날씨 위험도에 따른 penalty 계산 (위험할수록 작은 penalty)
        # risk_multiplier가 1.0일 때 penalty = 0, 2.0일 때 penalty = 5 정도
        weather_penalty = max(0, (risk_multiplier - 1.0) * 5)
        
        # 최종 safety score 계산
        safety_score = max(0, base_safety_score - weather_penalty)
        
        # Maximum safety loss threshold (이 값을 넘으면 score = 0)
        max_safety_threshold = 500
        
        # Safety score를 loss로 변환하여 threshold 체크
        safety_loss = base_safety_score - safety_score
        
        if safety_loss > max_safety_threshold:
            safety_score = 0
        
        self.safety_scores.append(safety_score)
        
        # Weather risk loss 기록
        self.safety_loss_breakdown["weather_risk"] += weather_penalty
        
        detailed_weather = self.weather.get_detailed_weather_info()
        debug(f"Safety score: {schedule.flight.flight_id} {operation_type}, weather: {detailed_weather['condition']}, visibility: {detailed_weather['visibility']}km, risk: {risk_multiplier:.2f} -> score {safety_score:.1f}")
        
        return safety_score
    
    def _check_accident_probability(self, schedule, operation_type):
        """사고 확률 체크 및 crash 이벤트 발생"""
        # 기본 사고 확률
        base_accident_prob = 0.01  # 1%
        
        # Emergency landing은 더 높은 확률
        if schedule.priority == 64:  # PRI_MAX
            base_accident_prob = 0.05  # 5%
        
        # 날씨 위험도에 따른 확률 조정
        if operation_type == "takeoff":
            risk_multiplier = self.weather.takeoff_risk_multiplier
        else:  # landing
            risk_multiplier = self.weather.landing_risk_multiplier
        
        # 최종 사고 확률 계산 (위험도가 높을수록 확률 증가)
        final_accident_prob = base_accident_prob * risk_multiplier
        
        # 사고 발생 여부 결정
        if random.random() < final_accident_prob:
            # Crash 이벤트 생성
            crash_event = type('Event', (), {
                'event_type': f'{operation_type.upper()}_CRASH',
                'target': schedule.flight.flight_id,
                'duration': random.randint(30, 120),  # 30-120분 활주로 폐쇄
                'time': self.time
            })()
            
            self.event_queue.append(crash_event)
            
            # 대량의 safety loss 추가 (priority 고려하지 않음)
            crash_safety_loss = 1000
            self.total_safety_loss += crash_safety_loss
            self.safety_loss_breakdown["accidents"] += crash_safety_loss
            
            weather_info = self.weather.get_detailed_weather_info()
            debug(f"🚨 CRASH EVENT: {schedule.flight.flight_id} {operation_type.upper()}_CRASH! Weather: {weather_info['condition']}, risk: {risk_multiplier:.2f}, prob: {final_accident_prob:.4f} -> {crash_safety_loss:.1f} safety loss 추가")
            
            return True
        
        return False
    
    def _add_runway_safety_loss(self, schedule, operation_type):
        """위험한 활주로 사용에 대한 추가 loss (이착륙 시작 시점에 체크)"""
        if not schedule.runway:
            return
        
        runway = schedule.runway
        safety_loss = 0.0
        
        debug(f"🔍 RUNWAY SAFETY CHECK: {schedule.flight.flight_id} {operation_type} on {runway.get_current_direction()}")
        debug(f"   - Runway closed: {runway.closed}")
        debug(f"   - Runway occupied: {runway.occupied}")
        debug(f"   - Current time: {self.time}")
        debug(f"   - Next available time: {runway.next_available_time}")
        debug(f"   - Time until available: {runway.next_available_time - self.time if runway.next_available_time > self.time else 0}")
        
        # 1. 활주로가 닫혀있는 경우
        if runway.closed:
            safety_loss += 500.0
            self.safety_loss_breakdown["runway_closed"] += 500.0
            debug(f"🚨 RUNWAY SAFETY LOSS: {schedule.flight.flight_id} using CLOSED runway {runway.get_current_direction()}")
        
        # 2. 활주로가 점유된 상태인 경우 (이착륙 시작 시점에 체크)
        if runway.occupied and self.time < runway.next_available_time:
            safety_loss += 300.0
            self.safety_loss_breakdown["runway_occupied"] += 300.0
            debug(f"🚨 RUNWAY SAFETY LOSS: {schedule.flight.flight_id} using OCCUPIED runway {runway.get_current_direction()} (current_time: {self.time}, next_available: {runway.next_available_time})")
            debug(f"   - Total runway_occupied loss so far: {self.safety_loss_breakdown['runway_occupied']}")
        
        # 3. 동시 이착륙 체크
        self._check_simultaneous_operations(schedule, operation_type)
        
        if safety_loss > 0:
            self.total_safety_loss += safety_loss
            debug(f"Runway safety loss added: {safety_loss:.1f} for {schedule.flight.flight_id}")
        
        # 이착륙 시작 시점에 활주로 점유 (위험도 체크 후)
        if operation_type == "takeoff":
            debug(f"🛫 OCCUPYING RUNWAY: {schedule.flight.flight_id} takeoff on {runway.get_current_direction()}")
            self._occupy_runway(runway, cooldown=3)
        else:  # landing
            debug(f"🛬 OCCUPYING RUNWAY: {schedule.flight.flight_id} landing on {runway.get_current_direction()}")
            self._occupy_runway(runway, cooldown=3)
    
    def _check_simultaneous_operations(self, schedule, operation_type):
        """동시 이착륙 체크 및 loss 추가"""
        current_time = self.time
        
        # 현재 시간에 다른 이착륙이 있는지 확인
        for other_schedule in self.schedules:
            if other_schedule == schedule:
                continue
            
            # 같은 시간에 이착륙하는 경우 체크
            if operation_type == "takeoff":
                if (other_schedule.status == FlightStatus.TAKE_OFF and 
                    hasattr(other_schedule, 'takeoff_time') and 
                    other_schedule.takeoff_time == current_time):
                    self._add_simultaneous_operation_loss(schedule, other_schedule, "takeoff")
            else:  # landing
                if (other_schedule.status == FlightStatus.LANDING and 
                    hasattr(other_schedule, 'landing_time') and 
                    other_schedule.landing_time == current_time):
                    self._add_simultaneous_operation_loss(schedule, other_schedule, "landing")
    
    def _add_simultaneous_operation_loss(self, schedule1, schedule2, operation_type):
        """동시 이착륙에 대한 큰 loss 추가"""
        self.total_safety_loss += 500
        self.safety_loss_breakdown["simultaneous_ops"] += 500
        debug("SIMULTANEOUS OPERATION LOSS: 500")
    
    def get_total_loss(self):
        """총 손실 반환"""
        return self.total_delay_loss + self.total_safety_loss
    
    def get_delay_score(self):
        """Delay score (100점 만점)"""
        if not self.delay_scores:
            return 100.0  # 지연이 없으면 만점
        
        # 모든 delay score의 평균
        avg_delay_score = sum(self.delay_scores) / len(self.delay_scores)
        return avg_delay_score
    
    def get_safety_score(self):
        """Safety score (100점 만점)"""
        if not self.safety_scores:
            return 100.0  # safety 이슈가 없으면 만점
        
        # 모든 safety score의 평균
        avg_safety_score = sum(self.safety_scores) / len(self.safety_scores)
        return avg_safety_score
    
    def get_total_score(self):
        """총 score (100점 만점) - delay와 safety의 평균"""
        delay_score = self.get_delay_score()
        safety_score = self.get_safety_score()
        
        # Delay와 Safety의 가중 평균 (각각 50%씩)
        total_score = (delay_score + safety_score) / 2
        return total_score
    
    def calculate_statistics(self):
        """통계 정보 계산"""
        # Priority-weighted delay time 계산
        total_weighted_delay = 0.0
        for schedule in self.completed_schedules:
            if hasattr(schedule, 'delay_minutes'):
                # Priority를 0-2 범위로 스케일링 (PRI_MAX=64를 2로 정규화)
                normalized_priority = schedule.priority / 32.0
                weighted_delay = schedule.delay_minutes * normalized_priority
                total_weighted_delay += weighted_delay
        
        self.total_delay_time_weighted = total_weighted_delay
        
        return {
            "total_delay_time_weighted": self.total_delay_time_weighted,
            "total_flights": self.total_flights,
            "total_safety_loss": self.total_safety_loss,
            "safety_breakdown": self.safety_loss_breakdown.copy()
        }
    
    def get_observed_events(self):
        """Agent에게 전달할 관측 가능한 이벤트만 반환"""
        observed_events = []
        
        for event in self.event_queue:
            # 예정된 일정만 전달 (긴급/예측 불가능한 이벤트는 제외)
            if event.event_type in ["RUNWAY_CLOSURE"]:
                observed_events.append(event)
        
        return observed_events

    def set_speed(self, speed):
        """Change simulation speed (1x, 2x, 4x, 8x, 64x)"""
        if speed in [1, 2, 4, 8, 64]:
            old_speed = self.speed
            self.speed = speed
            debug(f"시뮬레이션 속도 변경: {old_speed}x → {speed}x")
            return True
        else:
            debug(f"잘못된 속도 설정: {speed}. 가능한 값: 1, 2, 4, 8, 64")
            return False

    def set_rl_agent(self, agent, model_path=None):
        """RL 에이전트 설정"""
        self.rl_agent = agent
        
        # 모델 경로가 제공되면 로드
        if model_path and self.rl_agent:
            try:
                self.rl_agent.load_model(model_path)
                debug(f"훈련된 RL 모델을 로드했습니다: {model_path}")
            except Exception as e:
                debug(f"RL 모델 로드 실패: {e}")
        
        debug(f"RL 에이전트 설정 완료")

    def set_training_mode(self, training: bool):
        """훈련 모드 설정"""
        self.training_mode = training
        debug(f"RL 훈련 모드: {training}")
    
    def _calculate_immediate_reward(self) -> float:
        """즉시 피드백 계산 - 활주로 점유 충돌 및 동시 운영 위반 체크"""
        reward = 0.0
        
        # 활주로 점유 충돌 체크 (3분 간격)
        runway_usage = {}  # {runway_name: {time: flight_id}}
        simultaneous_ops = {}  # {time: [flight_ids]} - 동시 운영 체크
        
        for schedule in self.schedules:
            if schedule.status in [FlightStatus.TAXI_TO_RUNWAY, FlightStatus.WAITING]:
                if schedule.runway and (schedule.etd or schedule.eta):
                    time = schedule.etd if schedule.is_takeoff else schedule.eta
                    runway = schedule.runway
                    
                    # 1. 활주로 점유 충돌 체크 (3분 간격)
                    if runway not in runway_usage:
                        runway_usage[runway] = {}
                    
                    # 3분 간격으로 체크 (현재 시간부터 3분 후까지)
                    for check_time in range(time, time + 4):  # time, time+1, time+2, time+3
                        if check_time in runway_usage[runway]:
                            # 충돌 발생! -300 패널티
                            reward -= 300.0
                            break
                        else:
                            runway_usage[runway][check_time] = schedule.flight.flight_id
                    
                    # 2. 동시 운영 위반 체크 (같은 시간대에 이착륙)
                    if time not in simultaneous_ops:
                        simultaneous_ops[time] = []
                    
                    # 같은 시간대에 다른 비행이 있는지 체크
                    if len(simultaneous_ops[time]) > 0:
                        reward -= 500.0
                    else:
                        simultaneous_ops[time].append(schedule.flight.flight_id)
        
        return reward
    
    def _store_experience(self, immediate_reward: float):
        """경험 저장 (PPO용) - 기존 호환성 유지"""
        if self.rl_agent:
            # 현재 상태 관찰 (간단한 상태 표현)
            current_state = self._get_current_state()
            
            # 경험 저장 (액션 정보 없이)
            self.episode_experiences.append({
                'state': current_state,
                'reward': immediate_reward,
                'time': self.time
            })

    def _store_experience_with_actions(self, immediate_reward: float, actions: list, action_probs: list, value: float = 0.0):
        """실제 액션과 확률을 포함한 경험 저장 (PPO용)"""
        if self.rl_agent:
            # 현재 상태 관찰
            current_state = self._get_current_state()
            
            # 실제 액션과 확률을 포함한 경험 저장
            self.episode_experiences.append({
                'state': current_state,
                'actions': actions,        # 실제 선택한 액션들
                'action_probs': action_probs,  # 실제 액션 확률들
                'immediate_reward': immediate_reward,  # 즉시 보상
                'value': value,  # 현재 상태의 가치
                'time': self.time
            })

    def _get_current_state(self) -> np.ndarray:
        """현재 상태를 상세한 벡터로 표현"""
        state_features = []
        
        # 1. 시간 정보 (1차원)
        state_features.append(self.time / 1440.0)  # 정규화된 시간
        
        # 2. 활주로 상태 (각 활주로별)
        for runway in self.airport.runways:
            state_features.extend([
                runway.next_available_time / 1440.0  # 다음 가용 시간 (정규화)
            ])
        
        # 3. 날씨 예보 정보 (현재 시간부터 2시간, 5분 간격)
        weather_forecast = self.weather.get_forecast_for_action()
        for i, forecast in enumerate(weather_forecast[:24]):  # 처음 24개 시점 (2시간)
            # 날씨 위험도 정규화 (최대값 12.0으로 나누어 0~1 범위로)
            takeoff_risk = forecast.get('takeoff_risk', 1.0) / 12.0
            landing_risk = forecast.get('landing_risk', 1.0) / 12.0
            state_features.extend([
                takeoff_risk,  # 이륙 위험도 (정규화)
                landing_risk   # 착륙 위험도 (정규화)
            ])
        
        # 4. 스케줄 정보 (250차원)
        max_schedules = 50
        available_schedules = [s for s in self.schedules 
                                if s.status in [FlightStatus.DORMANT, FlightStatus.WAITING]]
        
        for i in range(max_schedules):
            if i < len(available_schedules):
                schedule = available_schedules[i]
                flight = schedule.flight
                
                # 스케줄별 상세 정보
                # FlightStatus를 숫자로 변환
                status_mapping = {
                    'dormant': 0.0,
                    'taxiToRunway': 1.0,
                    'waiting': 2.0,
                    'takeOff': 3.0,
                    'landing': 4.0,
                    'taxiToGate': 5.0,
                    'delayed': 6.0,
                    'cancelled': 7.0
                }
                status_value = status_mapping.get(schedule.status.value, 0.0)
                
                state_features.extend([
                    1.0 if schedule.is_takeoff else 0.0,  # 이륙/착륙 구분
                    (flight.priority or 0) / 64.0,  # 우선순위 (정규화) - PRI_MAX = 64
                    flight.etd / 1440.0 if flight.etd else 0.0,  # ETD (정규화)
                    flight.eta / 1440.0 if flight.eta else 0.0,  # ETA (정규화)
                    status_value / 10.0  # 상태 (정규화)
                ])
            else:
                # 패딩 (스케줄이 없는 경우) - 항상 같은 위치에 0으로 채움
                state_features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        
        # 5. 이벤트 정보 (확장된 버전) - 최대 10개 이벤트
        max_events = 10
        observed_events = self.get_observed_events()
        
        for i in range(max_events):
            if i < len(observed_events):
                event = observed_events[i]

                # 활주로 인코딩 (14L=0.0, 14R=1.0, 기타=0.5)
                runway_encoding = 0.0
                if event.target == "14L" or event.target == "32R":
                    runway_encoding = 0.0
                elif event.target == "14R" or event.target == "32L":
                    runway_encoding = 1.0
                else:
                    runway_encoding = 0.5
                
                state_features.extend([
                    runway_encoding,  # 활주로
                    event.time / 1440.0,  # 시작 시간 (정규화)
                    event.duration / 1440.0  # 지속 시간 (정규화)
                ])
            else:
                # 패딩 (이벤트가 없는 경우) - 항상 같은 위치에 0으로 채움
                state_features.extend([0.0, 0.0, 0.0])
        
        state_array = np.array(state_features)
        return state_array

    def _end_rl_episode(self, final_reward: float):
        """RL 에피소드 종료 처리 (PPO용)"""
        if not self.rl_agent or not self.episode_experiences:
            return
        
        # PPO 에이전트에 경험 전달
        for i, exp in enumerate(self.episode_experiences):
            # 실제 액션과 확률이 있는 경우 사용, 없으면 더미 사용
            if 'actions' in exp and 'action_probs' in exp:
                actions = exp['actions']
                action_probs = exp['action_probs']
            else:
                # 기존 호환성을 위한 더미 액션
                actions = [0] * 5
                action_probs = [0.2] * 5
            
            # 개별 보상 + 전체 보상
            immediate_reward = exp.get('immediate_reward', 0.0)
            gamma = 1  # 감가율 (은 적용하지 않을 예정)
            discounted_final_reward = final_reward
            total_reward = immediate_reward + discounted_final_reward
            
            # 실제 value 사용 (저장된 value가 있으면 사용, 없으면 0.0)
            value = exp.get('value', 0.0)
            
            self.rl_agent.store_transition(
                exp['state'], 
                actions, 
                action_probs,
                total_reward,  # 개별 보상 + 전체 보상
                value,  # 실제 value 사용
                i == len(self.episode_experiences) - 1  # 마지막이면 done=True
            )
        
        # PPO 에이전트 업데이트
        self.rl_agent.update()
        
        # 경험 초기화
        self.episode_experiences = []
        self.episode_count += 1
        
        debug(f"PPO 에피소드 {self.episode_count} 완료, 최종 보상: {final_reward}")

    def send_simulation_results(self):
        """Simulation results to frontend."""
        if self.ws:
            stats = self.calculate_statistics()
            results = {
                "type": "simulation_results",
                "time": int_to_hhmm_colon(self.time),
                "total_flights": self.total_flights,
                "cancelled_flights": self.cancelled_flights,
                "total_delay_time_weighted": stats["total_delay_time_weighted"],
                "total_delay_loss": self.total_delay_loss,
                "total_safety_loss": self.total_safety_loss,
                "safety_breakdown": stats["safety_breakdown"],
                "delay_scores": self.delay_scores,
                "safety_scores": self.safety_scores,
                "total_score": self.get_total_score(),
                "delay_score": self.get_delay_score(),
                "safety_score": self.get_safety_score()
            }
            self.ws.send(json.dumps(results))