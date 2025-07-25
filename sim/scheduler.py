from utils.logger import debug

class Scheduler:
    def __init__(self, airport, schedules):
        self.airport = airport
        self.schedules = schedules

    def optimize(self, method="greedy"):
        debug(f"스케줄 최적화 알고리즘: {method}")
        # 알고리즘 종류에 따라 최적화 수행 (greedy, ML, RL 등)
        if method == "greedy":
            self.greedy()
        elif method == "ml":
            self.ml()
        elif method == "rl":
            self.rl()

    def greedy(self):
        debug("Greedy 방식 스케줄링")
        # Greedy 방식 구현
        pass

    def rl(self):
        debug("RL 방식 스케줄링")
        # RL 방식 구현
        pass
