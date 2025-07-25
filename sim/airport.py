class Airport:
    def __init__(self, code, name, runways, taxiways):
        self.code = code
        self.name = name  # 'GMP'로 사용
        self.runways = runways  # List[Runway]
        self.taxiways = taxiways  # List[Taxiway]
        self.taxiway_time = 15    # 택시웨이 소요 시간


class Runway:
    def __init__(self, name):
        self.name = name
        self.occupied = False
        self.closed = False
        self.next_available_time = 0  # 다음 사용 가능 시간

class Taxiway:
    def __init__(self, name):
        self.name = name
        self.occupied = False
        self.closed = False
