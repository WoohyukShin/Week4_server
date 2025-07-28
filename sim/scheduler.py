from utils.logger import debug
from sim.flight import FlightStatus
from sim.schedule import Schedule
import heapq
from utils.time_utils import int_to_hhmm_colon

class Scheduler:
    def __init__(self, algorithm="greedy", sim=None):
        self.algorithm = algorithm
        self.sim = sim
    
    def optimize(self, schedules, current_time, event_queue=None, forecast=None):
        """ event_queue : 관측 가능한 예정된 모든 이벤트 (지금은 RWY_CLOSUIRE, RWY_INVERT만 존재) """
        """ 스케줄 최적화 메인 메서드 """
        match self.algorithm:
            case "greedy":
                self.greedy(schedules, current_time, event_queue, forecast)
            case "ml":
                self.ml(schedules, current_time, event_queue, forecast)
            case "rl":
                self.rl(schedules, current_time, event_queue, forecast)
            case _:
                self.greedy(schedules, current_time, event_queue, forecast)
        
        # 새롭게 재배정된 결과를 출력
        self._debug_schedule_times(schedules, current_time)
        debug("=====NEW SCHEDULER RESULT=====")
        # 모든 schedule 시간 순으로 출력
        schedule_times = []
        for schedule in schedules:
            if schedule.is_takeoff:
                time = schedule.etd
                if time is not None:
                    schedule_times.append((schedule.flight.flight_id, time))
            else:
                time = schedule.eta
                if time is not None:
                    schedule_times.append((schedule.flight.flight_id, time))
        
        # 시간 순서대로 정렬
        schedule_times.sort(key=lambda x: x[1])
        
        for flight_id, time in schedule_times:
            debug(f"{flight_id} : {int_to_hhmm_colon(time)}")
        debug("===========================")
    
    def greedy(self, schedules, current_time, event_queue=None, forecast=None):
        """시간 기반 그리디 알고리즘"""
        
        # 활주로 폐쇄 정보 추출
        runway_closures = []
        if event_queue:
            for event in event_queue:
                if event.event_type == "RUNWAY_CLOSURE":
                    runway_closures.append({
                        'runway': event.target,
                        'start_time': event.time,
                        'end_time': event.time + event.duration
                    })

        # 이륙/착륙 스케줄 분리 및 정렬
        takeoff_schedules = [s for s in schedules if s.is_takeoff and s.status == FlightStatus.DORMANT]
        landing_schedules = [s for s in schedules if not s.is_takeoff and s.status == FlightStatus.WAITING]
        
        # 우선순위 순으로 정렬 (priority 높은 순, 같은 priority면 시간 빠른 순)
        takeoff_schedules.sort(key=lambda s: (-s.priority, s.etd or 0))
        landing_schedules.sort(key=lambda s: (-s.priority, s.eta or 0))

        runway_available_times = {}
        for runway in self.sim.airport.runways:
            # 실제 활주로의 현재 상태를 반영
            if runway.closed:
                runway_available_times[runway] = max(current_time, runway.next_available_time)
            else:
                runway_available_times[runway] = max(current_time, 600)
        
        time = max(current_time, 600)
        max_time = 1440  # 24시간 (24 * 60) 제한

        while time <= max_time:
            for closure in runway_closures:
                if closure['start_time'] <= time <= closure['end_time']:
                    # 해당 활주로를 찾아서 사용 가능 시간을 폐쇄 종료 시간으로 설정
                    for runway in self.sim.airport.runways:
                        if runway.name == closure['runway'] or runway.inverted_name == closure['runway']:
                            runway_available_times[runway] = max(runway_available_times[runway], closure['end_time'])
            
            for schedule in takeoff_schedules:
                if schedule.flight.etd is None or schedule.flight.etd > time:
                    continue
                
                # 이륙 가능한 활주로 찾기 (14L 활주로만)
                assigned_runway = None
                for runway in self.sim.airport.runways:
                    if runway.name == "14L":  # 안쪽 활주로
                        if runway_available_times[runway] <= time:
                            assigned_runway = runway
                            break
                
                if assigned_runway:
                    schedule.runway = assigned_runway
                    schedule.etd = time  # 즉시 ETD 수정
                    runway_available_times[assigned_runway] = time + 4  # 이륙 1분 + 쿨다운 3분
                    debug(f"{schedule.flight.flight_id} 이륙: {int_to_hhmm_colon(time)} 활주로 {assigned_runway.get_current_direction()}")
                    takeoff_schedules.remove(schedule)
                    break  # 이 시간에는 하나만 처리
            
            # 현재 시간에 착륙할 수 있는 비행기 찾기
            for schedule in landing_schedules:
                if schedule.flight.eta != time:
                    continue
                
                # 착륙 가능한 활주로 찾기 (14R 활주로만)
                assigned_runway = None
                for runway in self.sim.airport.runways:
                    if runway.name == "14R":  # 바깥쪽 활주로
                        if runway_available_times[runway] <= time:
                            assigned_runway = runway
                            break
                
                if assigned_runway:
                    schedule.runway = assigned_runway
                    runway_available_times[assigned_runway] = time + 4  # 착륙 1분 + 쿨다운 3분
                    debug(f"{schedule.flight.flight_id} 착륙: {int_to_hhmm_colon(time)} 활주로 {assigned_runway.get_current_direction()}")
                    landing_schedules.remove(schedule)
                    break  # 이 시간에는 하나만 처리
                else:
                    # 착륙 실패 - go-around 발생
                    self.sim.event_handler._go_around(schedule.flight.flight_id)
                    debug(f"{schedule.flight.flight_id} 착륙 실패: {int_to_hhmm_colon(time)} - go-around 발생")
                    landing_schedules.remove(schedule)
                    break  # 이 시간에는 하나만 처리
            
            # 모든 스케줄이 배정되었는지 확인
            all_assigned = True
            unassigned_count = 0
            for schedule in takeoff_schedules + landing_schedules:
                # 이륙 스케줄: ETD가 있고 현재 시간보다 빠르거나 같으면 배정되어야 함
                if schedule.is_takeoff and schedule.etd is not None and schedule.etd <= time and schedule.runway is None:
                    all_assigned = False
                    unassigned_count += 1
                # 착륙 스케줄: ETA가 있고 현재 시간보다 빠르거나 같으면 배정되어야 함
                elif not schedule.is_takeoff and schedule.eta is not None and schedule.eta <= time and schedule.runway is None:
                    all_assigned = False
                    unassigned_count += 1
            
            if all_assigned:
                break
            
            # 다음 시간 계산 (가장 가까운 이벤트 시간으로 건너뛰기)
            next_time = time + 1  # 기본값: 1분 후
            
            # 1. 다음 이륙 가능 시간들 중 최솟값
            next_takeoff_times = []
            for schedule in takeoff_schedules:
                if schedule.runway is None and schedule.flight.etd is not None and schedule.flight.etd > time:
                    next_takeoff_times.append(schedule.flight.etd)
            
            # 2. 다음 착륙 시간들 중 최솟값
            next_landing_times = []
            for schedule in landing_schedules:
                if schedule.runway is None and schedule.flight.eta is not None and schedule.flight.eta > time:
                    next_landing_times.append(schedule.flight.eta)
            
            # 3. 다음 활주로 사용 가능 시간들 중 최솟값
            next_runway_times = [t for t in runway_available_times.values() if t > time]
            
            # 4. 다음 폐쇄 시작/종료 시간들
            next_closure_times = []
            for closure in runway_closures:
                if closure['start_time'] > time:
                    next_closure_times.append(closure['start_time'])
                if closure['end_time'] > time:
                    next_closure_times.append(closure['end_time'])
            
            # 모든 가능한 다음 시간들 중 최솟값
            all_next_times = next_takeoff_times + next_landing_times + next_runway_times + next_closure_times
            if all_next_times:
                next_time = min(all_next_times)
            
            time = next_time

    
    def ml(self, schedules, current_time, event_queue=None, forecast=None):
        """ML 알고리즘 (향후 구현)"""
        # TODO: ML 모델 적용
        return {}
    
    def rl(self, schedules, current_time, event_queue=None, forecast=None):
        """RL 알고리즘 (향후 구현)"""
        # TODO: 강화학습 에이전트 적용
        return {}

    def _debug_schedule_times(self, schedules, current_time):
        """모든 스케줄의 배정된 시간을 순서대로 출력"""
        schedule_times = []
        
        debug(f"현재 시간: {int_to_hhmm_colon(current_time)}, 총 스케줄 수: {len(schedules)}")
        
        for schedule in schedules:
            if schedule.is_takeoff:
                time = schedule.etd
                if time is not None:
                    schedule_times.append((schedule.flight.flight_id, time, "takeoff"))
                    debug(f"이륙 스케줄: {schedule.flight.flight_id} ETD={int_to_hhmm_colon(time)}")
            else:
                time = schedule.eta
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
