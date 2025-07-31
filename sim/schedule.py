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
        self.original_eta = flight.eta  # RL 전용 원본 ETA (가우시안 노이즈 적용된 초기값)
        self.atd = None
        self.is_takeoff = is_takeoff
        self.landing_time = None
        self.taxi_to_gate_time = None
        self.runway = None  # 계획된 활주로
        self.deadline = deadline  # 필요시만 사용
        
        # RL용 배정 정보 (do_action()에서 수동으로 설정)
        self.assigned_time = None  # 배정된 시간
        self.assigned_runway_id = None  # 배정된 활주로 ID 
        
        # Priority 할당 로직 (Flight에서 가져오거나 직접 설정)
        if priority is not None:
            self.priority = priority
        elif flight.priority is not None:
            self.priority = flight.priority
        else:
            # 기본값 설정
            if self.is_takeoff:
                self.priority = PRI_TAKEOFF_MIN
            else:
                self.priority = PRI_LANDING_MIN
    
    def get_normalized_priority(self):
        """Priority를 0-1 범위로 정규화"""
        return self.priority / PRI_MAX 