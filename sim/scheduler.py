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
                result =  self.greedy(schedules, current_time, event_queue, forecast)
            case "advanced":
                result = self.advanced(schedules, current_time, event_queue, forecast)
            case "ml":
                result =  self.ml(schedules, current_time, event_queue, forecast)
            case "rl":
                result =  self.rl(schedules, current_time, event_queue, forecast)
            case _:
                result =  self.greedy(schedules, current_time, event_queue, forecast)
        
        # 새롭게 재배정된 결과를 출력
        self._debug_schedule_times(schedules, current_time, result)
        debug("=====NEW SCHEDULER RESULT=====")
        # 모든 schedule 시간 순으로 출력
        schedule_times = []
        for schedule in schedules:
            if schedule.is_takeoff:
                time = result.get(schedule.flight.flight_id, schedule.etd) if result else schedule.etd
                if time is not None:
                    schedule_times.append((schedule.flight.flight_id, time))
            else:
                time = result.get(schedule.flight.flight_id, schedule.eta) if result else schedule.eta
                if time is not None:
                    schedule_times.append((schedule.flight.flight_id, time))
        
        # 시간 순서대로 정렬
        schedule_times.sort(key=lambda x: x[1])
        
        for flight_id, time in schedule_times:
            debug(f"{flight_id} : {int_to_hhmm_colon(time)}")
        debug("===========================")
        return result
    
    def greedy(self, schedules, current_time, event_queue=None, forecast=None):
        debug(f"greedy algorithm started at {int_to_hhmm_colon(current_time)}")
        changes = {}
        
        # 미완료 이벤트 고려 (예: 활주로 폐쇄 예정)
        runway_closures = []
        runway_inverts = []
        if event_queue:
            debug(f"관측 가능한 이벤트: {len(event_queue)}개")
            for event in event_queue:
                if event.event_type == "RUNWAY_CLOSURE":
                    runway_closures.append({
                        'runway': event.target,
                        'start_time': event.time,
                        'end_time': event.time + event.duration
                    })
                    debug(f"  - {event.event_type}: {event.target} ({int_to_hhmm_colon(event.time)} ~ {int_to_hhmm_colon(event.time + event.duration)})")
                elif event.event_type == "RUNWAY_INVERT":
                    runway_inverts.append(event.time)
                    debug(f"  - {event.event_type}: {int_to_hhmm_colon(event.time)}")
        
        # 이륙 스케줄과 착륙 스케줄 분리
        takeoff_schedules = [s for s in schedules if s.is_takeoff and s.status == FlightStatus.DORMANT]
        landing_schedules = [s for s in schedules if not s.is_takeoff and s.status == FlightStatus.WAITING]
        
        debug(f"이륙 스케줄: {len(takeoff_schedules)}개, 착륙 스케줄: {len(landing_schedules)}개")

        # 이륙 스케줄 그리디 처리 (priority 우선, 그 다음 ETD 순)
        takeoff_schedules.sort(key=lambda s: (-s.priority, s.etd))
        
        for schedule in takeoff_schedules:
            # 원래 ETD를 기준으로 시작
            original_etd = schedule.etd
            takeoff_time = max(current_time, original_etd)
            
            # 활주로 폐쇄 고려
            for closure in runway_closures:
                if closure['start_time'] <= takeoff_time <= closure['end_time']:
                    takeoff_time = closure['end_time'] + 1
            
            # 활주로 배정 및 사용 가능 시간 확인
            assigned_runway = None
            for runway in self.sim.airport.runways:
                current_direction = runway.get_current_direction()
                if current_direction in ["14L", "32R"] and not runway.closed:
                    # 활주로 사용 가능 시간 확인
                    if runway.can_handle_operation(takeoff_time):
                        assigned_runway = runway
                        break
                    else:
                        # 활주로가 사용 중이면 다음 가능 시간으로 조정
                        takeoff_time = max(takeoff_time, runway.next_available_time)
                        assigned_runway = runway
                        break
            
            if assigned_runway:
                schedule.runway = assigned_runway
                changes[schedule.flight.flight_id] = takeoff_time
                debug(f"{schedule.flight.flight_id} 이륙: {int_to_hhmm_colon(takeoff_time)} 활주로 {assigned_runway.get_current_direction()}")
            else:
                debug(f"{schedule.flight.flight_id} 이륙: 사용 가능한 활주로 없음")
        
        # 착륙 스케줄 그리디 처리 (시간 조정 불가, 활주로 배정만)
        landing_schedules.sort(key=lambda s: (-s.priority, s.eta))
        
        for schedule in landing_schedules:
            original_eta = schedule.eta
            if original_eta is None:
                continue
                
            # 착륙 시간은 조정하지 않음 (원래 ETA 유지, ETA보다 빨라선 안 됨)
            landing_time = original_eta
            
            # 현재 시간보다 ETA가 빠르면 스킵 (아직 착륙할 시간이 아님)
            if landing_time < current_time:
                continue
            
            # 활주로 배정
            assigned_runway = None
            for runway in self.sim.airport.runways:
                current_direction = runway.get_current_direction()
                if current_direction in ["14R", "32L"] and not runway.closed:
                    if runway.can_handle_operation(landing_time):
                        assigned_runway = runway
                        break
            
            if assigned_runway:
                schedule.runway = assigned_runway
                debug(f"{schedule.flight.flight_id} 착륙: {int_to_hhmm_colon(landing_time)} 활주로 {assigned_runway.get_current_direction()}")
            else:
                # 착륙 시간에 활주로를 사용할 수 없으면 go-around 발생
                self.sim.event_handler._go_around(schedule.flight.flight_id)
                debug(f"{schedule.flight.flight_id} 착륙: 활주로 사용 불가로 go-around 발생")
        return changes
    
    def ml(self, schedules, current_time, event_queue=None, forecast=None):
        """ML 알고리즘 (향후 구현)"""
        # TODO: ML 모델 적용
        return {}
    
    def rl(self, schedules, current_time, event_queue=None, forecast=None):
        """RL 알고리즘 (향후 구현)"""
        # TODO: 강화학습 에이전트 적용
        return {}

    def advanced(self, schedules, current_time, event_queue=None, forecast=None):
        """Advanced scheduler using MILP optimization"""
        debug(f"Advanced scheduler algorithm started at {int_to_hhmm_colon(current_time)}")
        
        # Import AdvancedScheduler here to avoid circular imports
        from sim.advanced_scheduler import AdvancedScheduler
        
        # Create a temporary advanced scheduler instance
        advanced_scheduler = AdvancedScheduler(self.sim)
        
        # Run the advanced optimization
        result = advanced_scheduler.optimize(schedules, current_time, event_queue, forecast)
        
        # Assign runways to schedules based on the result
        # This is needed because the advanced scheduler returns times but doesn't assign runways
        for schedule in schedules:
            if schedule.flight.flight_id in result:
                # Assign appropriate runway based on operation type
                if schedule.is_takeoff:
                    # Find takeoff runway (14L or 32R)
                    for runway in self.sim.airport.runways:
                        current_direction = runway.get_current_direction()
                        if current_direction in ["14L", "32R"] and not runway.closed:
                            schedule.runway = runway
                            break
                else:
                    # Find landing runway (14R or 32L)
                    for runway in self.sim.airport.runways:
                        current_direction = runway.get_current_direction()
                        if current_direction in ["14R", "32L"] and not runway.closed:
                            schedule.runway = runway
                            break
        
        return result

    def _debug_schedule_times(self, schedules, current_time, changes=None):
        """모든 스케줄의 배정된 시간을 순서대로 출력 (변경사항 반영)"""
        schedule_times = []
        
        debug(f"현재 시간: {int_to_hhmm_colon(current_time)}, 총 스케줄 수: {len(schedules)}")
        
        for schedule in schedules:
            if schedule.is_takeoff:
                # 변경사항이 있으면 변경된 시간 사용, 없으면 스케줄 ETD 사용
                time = changes.get(schedule.flight.flight_id, schedule.etd) if changes else schedule.etd
                if time is not None:
                    schedule_times.append((schedule.flight.flight_id, time, "takeoff"))
                    debug(f"이륙 스케줄: {schedule.flight.flight_id} ETD={int_to_hhmm_colon(time)}")
            else:
                # 변경사항이 있으면 변경된 시간 사용, 없으면 스케줄 ETA 사용
                time = changes.get(schedule.flight.flight_id, schedule.eta) if changes else schedule.eta
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
