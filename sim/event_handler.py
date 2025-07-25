from sim.flight import FlightStatus, Flight
from sim.schedule import Schedule

class EventHandler:
    def __init__(self, simulation):
        self.sim = simulation

    def handle(self, event, current_time):
        etype = event.event_type
        target = event.target
        duration = event.duration
        self.sim.debug(f"{etype}({target}) at {current_time} (duration={duration})")
        if etype == "EMERGENCY_LANDING":
            self._emergency_landing(target, duration, current_time)
        elif etype == "RUNWAY_CLOSURE":
            self._close_runway(target, duration, current_time)
        elif etype == "FLIGHT_CANCEL":
            self._cancel_flight(target)
        elif etype == "FLIGHT_DELAY":
            self._delay_flight(target, duration, current_time)
        elif etype == "GO_AROUND":
            self._go_around(target)
        elif etype == "TAKEOFF_CRASH":
            self._takeoff_crash(target, duration, current_time)
        elif etype == "LANDING_CRASH":
            self._landing_crash(target, duration, current_time)
        elif etype == "RUNWAY_INVERT":
            self._invert_runway()

    def _emergency_landing(self, flight_id, duration, current_time):
        self.sim.debug(f"EMERGENCY_LANDING: {flight_id} {duration}분 내 착륙 필요")
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
        self.sim.debug(f"RUNWAY_CLOSURE: {runway_name} {duration}분간 폐쇄")
        for r in self.sim.airport.runways:
            if r.name == runway_name:
                r.closed = True
                r.next_available_time = current_time + duration
        self.sim._update_runway_roles_on_closure()

    def _reopen_runway(self, runway_name):
        self.sim.debug(f"RUNWAY_REOPEN: {runway_name} 재개방")
        for r in self.sim.airport.runways:
            if r.name == runway_name:
                r.closed = False
                r.next_available_time = self.sim.time
        self.sim._update_runway_roles_on_closure()

    def _cancel_flight(self, flight_id):
        self.sim.debug(f"FLIGHT_CANCEL: {flight_id} 취소")
        for s in self.sim.schedules:
            if s.flight.flight_id == flight_id:
                s.status = FlightStatus.CANCELLED

    def _delay_flight(self, flight_id, duration, current_time):
        self.sim.debug(f"FLIGHT_DELAY: {flight_id} {duration}분 지연")
        for s in self.sim.schedules:
            if s.flight.flight_id == flight_id and s.status == FlightStatus.DORMANT:
                if s.flight.etd is not None:
                    s.flight.etd += duration
                s.status = FlightStatus.DELAYED

    def _go_around(self, flight_id):
        self.sim.debug(f"GO_AROUND: {flight_id} 착륙 재시도(15분 지연)")
        for s in self.sim.schedules:
            if s.flight.flight_id == flight_id and s.status == FlightStatus.LANDING:
                s.landing_time += 15

    def _takeoff_crash(self, flight_id, duration, current_time):
        self.sim.debug(f"TAKEOFF_CRASH: {flight_id} 이륙 중 사고, {duration}분간 이륙 활주로 폐쇄")
        self._close_runway(self.sim.takeoff_runway, duration, current_time)

    def _landing_crash(self, flight_id, duration, current_time):
        self.sim.debug(f"LANDING_CRASH: {flight_id} 착륙 중 사고, {duration}분간 착륙 활주로 폐쇄")
        self._close_runway(self.sim.landing_runway, duration, current_time)

    def _invert_runway(self):
        self.sim.debug(f"RUNWAY_INVERT: 이륙/착륙 활주로 역할 스왑")
        if self.sim.takeoff_runway == "14L" and self.sim.landing_runway == "14R":
            self.sim.takeoff_runway = "32R"
            self.sim.landing_runway = "32L"
        else:
            self.sim.takeoff_runway = "14L"
            self.sim.landing_runway = "14R"
        self.sim._update_runway_roles_on_closure()
