class Airport:
    def __init__(self, code, name, runways, taxiways):
        self.code = code
        self.name = name  # 'GMP'로 사용
        self.runways = runways  # List[Runway]
        self.taxiways = taxiways  # List[Taxiway]
        self.taxiway_time = 15    # 택시웨이 소요 시간


class Runway:
    def __init__(self, name, inverted_name):
        self.name = name  # 기본 방향 (14L, 14R)
        self.inverted_name = inverted_name  # 반대 방향 (32R, 32L)
        self.inverted = False  # 현재 방향 상태
        self.occupied = False
        self.closed = False
        self.next_available_time = 0  # 다음 사용 가능 시간
        
    def get_current_direction(self):
        """현재 활성 방향 반환"""
        return self.inverted_name if self.inverted else self.name
    
    def invert(self):
        """방향 전환"""
        self.inverted = not self.inverted
    
    def can_handle_operation(self, current_time):
        """이 활주로가 현재 작업을 처리할 수 있는지 확인"""
        if self.closed or self.occupied:
            return False
        if self.next_available_time > current_time:
            return False
        return True

class Taxiway:
    def __init__(self, name, inverted_name=None):
        self.name = name
        self.inverted_name = inverted_name if inverted_name else name  # 기본값은 name과 동일
        self.inverted = False  # 현재 방향 상태
        self.occupied = False
        self.closed = False
        
    def get_current_name(self):
        """현재 활성 이름 반환"""
        return self.inverted_name if self.inverted else self.name
    
    def invert(self):
        """방향 전환"""
        self.inverted = not self.inverted
