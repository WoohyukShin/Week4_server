from sim.flight import FlightStatus
import random

# Priority 상수들
PRI_MAX = 64
PRI_TAKEOFF_MIN = 0
PRI_TAKEOFF_MAX = 48
PRI_LANDING_MIN = 15
PRI_LANDING_MAX = 63

class Schedule:
    def __init__(self, flight, atd=None, ata=None, is_takeoff=None, priority=None, deadline=None):
        self.flight = flight
        self.atd = atd
        self.ata = ata
        self.status = FlightStatus.DORMANT
        self.start_taxi_time = None
        self.etd = flight.etd  # 초기값을 flight.etd로 설정
        self.eta = flight.eta  # 초기값을 flight.eta로 설정
        self.atd = None
        self.is_takeoff = is_takeoff
        self.landing_time = None
        self.taxi_to_gate_time = None
        self.runway = None  # 계획된 활주로
        self.deadline = deadline  # 필요시만 사용 
        
        # Priority 할당 로직
        if priority is not None:
            self.priority = priority
        else:
            self.priority = self._assign_priority()
    
    def _assign_priority(self):
        """비행 타입에 따라 priority를 랜덤으로 할당"""
        if self.is_takeoff:
            return random.randint(PRI_TAKEOFF_MIN, PRI_TAKEOFF_MAX)
        else:
            return random.randint(PRI_LANDING_MIN, PRI_LANDING_MAX)
    
    def get_normalized_priority(self):
        """Priority를 0-1 범위로 정규화"""
        return self.priority / PRI_MAX 