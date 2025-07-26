from sim.flight import FlightStatus, Flight
from sim.schedule import Schedule
from utils.logger import debug

class EventHandler:
    def __init__(self, simulation):
        self.sim = simulation

    def handle(self, event, current_time):
        etype = event.event_type
        target = event.target
        duration = event.duration
        debug(f"{etype}({target}) at {current_time} (duration={duration})")
        
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
        flight = Flight(flight_id, etd=None, eta=None, dep_airport=None, arr_airport=None, airline="")
        emergency_schedule = Schedule(
            flight,
            is_takeoff=False,
            priority=10,
            deadline=current_time + duration
        )
        emergency_schedule.status = FlightStatus.WAITING
        self.sim.schedules.append(emergency_schedule)

    def _close_runway(self, runway_name, duration, current_time):
        debug(f"RUNWAY_CLOSURE: {runway_name} {duration}분간 폐쇄")
        for r in self.sim.airport.runways:
            if r.name == runway_name:
                r.closed = True
                r.next_available_time = current_time + duration
        self.sim._update_runway_roles_on_closure()

    def _reopen_runway(self, runway_name):
        debug(f"RUNWAY_REOPEN: {runway_name} 재개방")
        for r in self.sim.airport.runways:
            if r.name == runway_name:
                r.closed = False
                r.next_available_time = self.sim.time
        self.sim._update_runway_roles_on_closure()

    def _cancel_flight(self, flight_id):
        debug(f"FLIGHT_CANCEL: {flight_id} 취소")
        for s in self.sim.schedules:
            if s.flight.flight_id == flight_id:
                s.status = FlightStatus.CANCELLED

    def _delay_flight(self, flight_id, duration, current_time):
        debug(f"FLIGHT_DELAY: {flight_id} {duration}분 지연")
        for s in self.sim.schedules:
            if s.flight.flight_id == flight_id and s.status == FlightStatus.DORMANT:
                if s.flight.etd is not None:
                    s.flight.etd += duration
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
        self._close_runway(self.sim.takeoff_runway, duration, current_time)

    def _landing_crash(self, flight_id, duration, current_time):
        debug(f"LANDING_CRASH: {flight_id} 착륙 중 사고, {duration}분간 착륙 활주로 폐쇄")
        self._close_runway(self.sim.landing_runway, duration, current_time)

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
            landing_schedule = Schedule(flight, is_takeoff=False, priority=0)
            landing_schedule.status = FlightStatus.WAITING
            self.sim.schedules.append(landing_schedule)
