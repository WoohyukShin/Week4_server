from sim.flight import FlightStatus, Flight
from sim.schedule import Schedule, PRI_MAX
from utils.logger import debug
from sim.event import Event

class EventHandler:
    def __init__(self, simulation):
        self.sim = simulation

    def handle(self, event, current_time):
        etype = event.event_type
        target = event.target
        duration = event.duration
        debug(f"{etype}({target}) at {current_time} (duration={duration})")
        
        if etype != "LANDING_ANNOUNCE":
            # WebSocket으로 프론트엔드에 이벤트 정보 전송
            self._send_event_to_frontend(etype, target, duration, current_time)
        
        match etype:
            case "EMERGENCY_LANDING":
                self._emergency_landing(target, duration, current_time)
            case "RUNWAY_CLOSURE":
                self._close_runway(target, duration, current_time)
            case "FLIGHT_CANCEL":
                self._cancel_flight(target)
            case "FLIGHT_DELAY":
                self._delay_flight(target, duration, current_time)
            case "GO_AROUND":
                self._go_around(target)
            case "TAKEOFF_CRASH":
                self._takeoff_crash(target, duration, current_time)
            case "LANDING_CRASH":
                self._landing_crash(target, duration, current_time)
            case "RUNWAY_INVERT":
                self._invert_runway()
            case "LANDING_ANNOUNCE": # 백에서 flight 보고 랜덤으로 생성한 landing event
                self._landing_announce(target, duration, current_time)


    def _emergency_landing(self, flight_id, duration, current_time):
        debug(f"EMERGENCY_LANDING: {flight_id} {duration}분 내 착륙 필요")
        flight = Flight(flight_id, etd=None, eta=current_time + 3, dep_airport=None, arr_airport=None, airline="")
        emergency_schedule = Schedule(
            flight,
            is_takeoff=False,
            priority=PRI_MAX,
            deadline=current_time + duration
        )
        emergency_schedule.status = FlightStatus.WAITING
        self.sim.schedules.append(emergency_schedule)

    def _close_runway(self, runway_name, duration, current_time):
        debug(f"RUNWAY_CLOSURE: {runway_name} {duration}분간 폐쇄")
        for r in self.sim.airport.runways:
            # 현재 활주로의 name 또는 inverted_name과 매칭되는지 확인
            if r.name == runway_name or r.inverted_name == runway_name:
                r.closed = True
                r.next_available_time = current_time + duration
                debug(f"  - {r.name}({r.inverted_name}) 활주로 폐쇄")
                
                # 해당 활주로로 가던 비행기들을 TAXI_TO_GATE로 변경
                for schedule in self.sim.schedules:
                    if (schedule.status == FlightStatus.TAXI_TO_RUNWAY):
                        # TAXI_TO_RUNWAY로 간 시간 계산
                        taxi_duration = current_time - schedule.start_taxi_time
                        # 돌아가는 시간 = 간 시간만큼만 (최대 10분)
                        return_duration = min(taxi_duration, 10)
                        
                        # 반대 방향의 runway name 저장 (프론트엔드 애니메이션용)
                        if schedule.runway is not None and schedule.runway.get_current_direction() == r.name:
                            schedule.opposite_runway_direction = r.inverted_name  # 14L -> 32R
                        elif schedule.runway is not None and schedule.runway.get_current_direction() == r.inverted_name:
                            schedule.opposite_runway_direction = r.name  # 32R -> 14L
                        else:
                            # If no runway assigned, use default based on operation type
                            if schedule.is_takeoff:
                                schedule.opposite_runway_direction = "32R" if r.name == "14L" else "14L"
                            else:
                                schedule.opposite_runway_direction = "32L" if r.name == "14R" else "14R"
                  
                        schedule.status = FlightStatus.TAXI_TO_GATE
                        schedule.taxi_to_gate_time = current_time - return_duration  # 돌아가는 시간만큼 앞으로 설정
                        
                        # etd를 None으로 초기화하여 다시 스케줄링되도록 함
                        schedule.etd = None
                        schedule.runway = None
                        
                        # TAXI_TO_GATE 완료 후 자동으로 DORMANT로 변경되고 do_action() 호출됨
                        debug(f"TAXI_TO_GATE 시작: {schedule.flight.flight_id}, 10분 후 DORMANT로 변경 예정")
                    
    def _reopen_runway(self, runway_name):
        debug(f"RUNWAY_REOPEN: {runway_name} 재개방")
        for r in self.sim.airport.runways:
            # 현재 활주로의 name 또는 inverted_name과 매칭되는지 확인
            if r.name == runway_name or r.inverted_name == runway_name:
                r.closed = False
                r.next_available_time = self.sim.time
                debug(f"  - {r.name}({r.inverted_name}) 활주로 재개방")

    def _cancel_flight(self, flight_id):
        debug(f"FLIGHT_CANCEL: {flight_id} 취소")
        for s in self.sim.schedules:
            if s.flight.flight_id == flight_id:
                s.status = FlightStatus.CANCELLED
                self.sim.cancelled_flights += 1  # 취소된 비행 수 증가

    def _delay_flight(self, flight_id, duration, current_time):
        debug(f"FLIGHT_DELAY: {flight_id} {duration}분 지연")
        for s in self.sim.schedules:
            if s.flight.flight_id == flight_id and s.status == FlightStatus.DORMANT:
                # schedule의 etd를 직접 수정
                s.etd += duration
                debug(f"FLIGHT_DELAY: {flight_id} ETD {s.etd - duration} -> {s.etd}")
                # 지연 시간 기록
                s.delay_duration = duration
                s.status = FlightStatus.DELAYED

    def _go_around(self, flight_id):
        debug(f"GO_AROUND: {flight_id} 착륙 재시도(15분 지연)")
        for s in self.sim.schedules:
            if s.flight.flight_id == flight_id and s.status == FlightStatus.LANDING:
                s.landing_time += 15
                # Go-around 손실 추가
                self.sim._add_go_around_loss(s)

    def _takeoff_crash(self, flight_id, duration, current_time):
        debug(f"TAKEOFF_CRASH: {flight_id} 이륙 중 사고, {duration}분간 이륙 활주로 폐쇄")
        # 이륙 활주로 찾기 (14L 또는 14R)
        takeoff_runway = None
        for runway in self.sim.airport.runways:
            if runway.name in ["14L", "14R"]:
                takeoff_runway = runway.name
                break
        if takeoff_runway:
            # 프론트엔드에 전송
            self._send_event_to_frontend("RUNWAY_CLOSURE", takeoff_runway, duration, current_time)
            # 직접 활주로 폐쇄
            self._close_runway(takeoff_runway, duration, current_time)

    def _landing_crash(self, flight_id, duration, current_time):
        debug(f"LANDING_CRASH: {flight_id} 착륙 중 사고, {duration}분간 착륙 활주로 폐쇄")
        # 착륙 활주로 찾기 (32L 또는 32R)
        landing_runway = None
        for runway in self.sim.airport.runways:
            if runway.name in ["32L", "32R"]:
                landing_runway = runway.name
                break
        if landing_runway:
            # 프론트엔드에 전송
            self._send_event_to_frontend("RUNWAY_CLOSURE", landing_runway, duration, current_time)
            # 직접 활주로 폐쇄
            self._close_runway(landing_runway, duration, current_time)

    def _invert_runway(self):
        debug("RUNWAY_INVERT: 모든 활주로 방향 전환")
        # 모든 활주로의 방향을 전환
        for runway in self.sim.airport.runways:
            runway.invert()
        # 모든 택시웨이의 방향을 전환
        for taxiway in self.sim.airport.taxiways:
            taxiway.invert()

    def _landing_announce(self, flight_id, duration, current_time):
        debug(f"LANDING_ANNOUNCE: {flight_id} {duration}분 뒤 랜딩 예정")
        # landing schedule을 queue에 추가
        flight = next((f for f in self.sim.landing_flights if f.flight_id == flight_id), None)
        if flight:
            landing_schedule = Schedule(flight, is_takeoff=False)
            landing_schedule.status = FlightStatus.WAITING
            for runway in self.sim.airport.runways:
                if runway.name == "14R":
                    landing_schedule.runway = runway
                    break
            self.sim.schedules.append(landing_schedule)
    
    def _send_event_to_frontend(self, event_type, target, duration, current_time):
        """WebSocket으로 프론트엔드에 이벤트 정보 전송"""
        if not self.sim.ws:
            return
        
        # 이벤트 타입에 따른 targetType 결정
        target_type = self._get_target_type(event_type)
        
        event_data = {
            "type": "event",
            "time": current_time,
            "event": {
                "event_type": event_type,
                "targetType": target_type,
                "target": target,
                "duration": duration
            }
        }
        
        try:
            self.sim.ws.send(event_data)
            debug(f"이벤트 전송: {event_type} -> 프론트엔드")
        except Exception as e:
            debug(f"이벤트 전송 실패: {e}")
    
    def _get_target_type(self, event_type):
        """이벤트 타입에 따른 targetType 반환"""
        match event_type:
            case "EMERGENCY_LANDING" | "FLIGHT_CANCEL" | "FLIGHT_DELAY" | "GO_AROUND" | "TAKEOFF_CRASH" | "LANDING_CRASH" | "LANDING_ANNOUNCE":
                return "Flight"
            case "RUNWAY_CLOSURE" | "RUNWAY_INVERT":
                return "Runway"
            case _:
                return ""


