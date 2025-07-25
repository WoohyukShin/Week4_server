from utils.logger import debug
from sim.flight import FlightStatus
from sim.schedule import Schedule
import heapq
from utils.time_utils import int_to_hhmm_colon

class Scheduler:
    def __init__(self, algorithm="greedy"):
        self.algorithm = algorithm
    
    def optimize(self, schedules, current_time, event_queue=None):
        """스케줄 최적화 메인 메서드"""
        match self.algorithm:
            case "greedy":
                result =  self.greedy(schedules, current_time, event_queue)
            case "ml":
                result =  self.ml(schedules, current_time, event_queue)
            case "rl":
                result =  self.rl(schedules, current_time, event_queue)
            case _:
                result =  self.greedy(schedules, current_time, event_queue)
        
        # 새롭게 재배정된 결과를 출력
        self._debug_schedule_times(schedules, current_time, result)
        
        debug(f"scheduler result: {result}")
        return result
    
    def greedy(self, schedules, current_time, event_queue=None):
        debug(f"greedy algorithm started at {int_to_hhmm_colon(current_time)}")
        changes = {}
        
        # 미완료 이벤트 고려 (예: 활주로 폐쇄 예정)
        runway_closures = []
        if event_queue:
            for event in event_queue:
                if event.event_type == "RUNWAY_CLOSURE":
                    runway_closures.append({
                        'runway': event.target,
                        'start_time': event.time,
                        'end_time': event.time + event.duration
                    })
        
        # 이륙 스케줄과 착륙 스케줄 분리
        takeoff_schedules = [s for s in schedules if s.is_takeoff and s.status == FlightStatus.DORMANT]
        landing_schedules = [s for s in schedules if not s.is_takeoff and s.status == FlightStatus.WAITING]
        
        # 이륙 스케줄 그리디 처리
        takeoff_schedules.sort(key=lambda s: s.flight.etd)  # ETD 순으로 정렬
        takeoff_runway_available_time = current_time
        
        for schedule in takeoff_schedules:
            # 택시 시작 시간 계산 (이륙 15분 전)
            taxi_start_time = max(current_time, schedule.flight.etd - 15)
            takeoff_time = taxi_start_time + 15  # 15분 택시 후 이륙
            
            # 활주로 폐쇄 고려
            for closure in runway_closures:
                if closure['start_time'] <= takeoff_time <= closure['end_time']:
                    takeoff_time = closure['end_time'] + 1  # 폐쇄 후 첫 가능 시간
            
            # 활주로 사용 가능 시간 확인
            if takeoff_time < takeoff_runway_available_time:
                takeoff_time = takeoff_runway_available_time
            
            # 변경사항 기록
            if takeoff_time != schedule.flight.etd:
                changes[schedule.flight.flight_id] = takeoff_time
            
            # 다음 이륙 활주로 사용 가능 시간 (이륙 1분 + 쿨다운 3분)
            takeoff_runway_available_time = takeoff_time + 4
        
        # 착륙 스케줄 그리디 처리
        landing_schedules.sort(key=lambda s: s.flight.eta)  # ETA 순으로 정렬
        landing_runway_available_time = current_time
        
        for schedule in landing_schedules:
            eta = schedule.flight.eta if schedule.flight.eta is not None else current_time
            landing_time = max(current_time, eta, landing_runway_available_time)
            
            # 활주로 폐쇄 고려
            for closure in runway_closures:
                if closure['start_time'] <= landing_time <= closure['end_time']:
                    landing_time = closure['end_time'] + 1  # 폐쇄 후 첫 가능 시간
            
            # 변경사항 기록
            if landing_time != schedule.flight.eta:
                changes[schedule.flight.flight_id] = landing_time
            
            # 다음 착륙 활주로 사용 가능 시간 (착륙 1분 + 쿨다운 3분)
            landing_runway_available_time = landing_time + 4
        
        return changes
    
    def ml(self, schedules, current_time):
        """ML 알고리즘 (향후 구현)"""
        # TODO: ML 모델 적용
        return {}
    
    def rl(self, schedules, current_time):
        """RL 알고리즘 (향후 구현)"""
        # TODO: 강화학습 에이전트 적용
        return {}

    def _debug_schedule_times(self, schedules, current_time, changes=None):
        """모든 스케줄의 배정된 시간을 순서대로 출력 (변경사항 반영)"""
        schedule_times = []
        
        debug(f"현재 시간: {int_to_hhmm_colon(current_time)}, 총 스케줄 수: {len(schedules)}")
        
        for schedule in schedules:
            if schedule.is_takeoff:
                # 변경사항이 있으면 변경된 시간 사용, 없으면 원래 ETD 사용
                time = changes.get(schedule.flight.flight_id, schedule.flight.etd) if changes else schedule.flight.etd
                if time is not None:
                    schedule_times.append((schedule.flight.flight_id, time, "takeoff"))
                    debug(f"이륙 스케줄: {schedule.flight.flight_id} ETD={int_to_hhmm_colon(time)}")
            else:
                # 변경사항이 있으면 변경된 시간 사용, 없으면 원래 ETA 사용
                time = changes.get(schedule.flight.flight_id, schedule.flight.eta) if changes else schedule.flight.eta
                if time is not None:
                    schedule_times.append((schedule.flight.flight_id, time, "landing"))
                    debug(f"착륙 스케줄: {schedule.flight.flight_id} ETA={int_to_hhmm_colon(time)}")
        
        # 시간 순서대로 정렬
        schedule_times.sort(key=lambda x: x[1])
        
        if schedule_times:
            time_str = " | ".join([f"{flight_id} {int_to_hhmm_colon(time)}" for flight_id, time, _ in schedule_times])
            debug(f"스케줄 배정: {time_str}")
        else:
            debug("배정된 스케줄 없음")

def calculate_loss(completed_schedules):
    """시뮬레이션 완료 후 전체 손실 계산"""
    total_delay = 0
    
    for schedule in completed_schedules:
        if schedule.is_takeoff:
            # 이륙 지연 계산
            if hasattr(schedule, 'atd') and schedule.atd:
                delay = schedule.atd - schedule.flight.etd
                if delay > 0:
                    total_delay += delay
        else:
            # 착륙 지연 계산
            if hasattr(schedule, 'ata') and schedule.ata:
                delay = schedule.ata - schedule.flight.eta
                if delay > 0:
                    total_delay += delay
    
    return total_delay
