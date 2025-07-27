from sim.flight import FlightStatus

class Schedule:
    def __init__(self, flight, atd=None, ata=None, is_takeoff=None, priority=0, deadline=None):
        self.flight = flight
        self.atd = atd
        self.ata = ata
        self.status = FlightStatus.DORMANT
        self.location = None
        self.start_taxi_time = None
        self.etd = flight.etd  # 초기값을 flight.etd로 설정
        self.eta = flight.eta  # 초기값을 flight.eta로 설정
        self.atd = None
        self.is_takeoff = is_takeoff
        self.landing_time = None
        self.taxi_to_gate_time = None
        self.priority = priority  # 기본 0, emergency 등은 더 높게
        self.deadline = deadline  # 필요시만 사용 