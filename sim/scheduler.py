from utils.logger import debug
from sim.flight import FlightStatus
from sim.schedule import Schedule
import heapq
from utils.time_utils import int_to_hhmm_colon

class Scheduler:
    def __init__(self, algorithm="greedy", sim=None):
        self.algorithm = algorithm
        self.sim = sim
    
    def optimize(self, schedules, current_time, event_queue=None, forecast=None, runway_availability=None):
        """ event_queue : 관측 가능한 예정된 모든 이벤트 (지금은 RWY_CLOSUIRE, RWY_INVERT만 존재) """
        """ 스케줄 최적화 메인 메서드 """
        match self.algorithm:
            case "greedy":
                result =  self.greedy(schedules, current_time, event_queue, forecast, runway_availability)
            case "advanced":
                result = self.advanced(schedules, current_time, event_queue, forecast, runway_availability)
            case "ml":
                self.ml(schedules, current_time, event_queue, forecast, runway_availability)
            case "rl":
                self.rl(schedules, current_time, event_queue, forecast, runway_availability)
            case _:
                self.greedy(schedules, current_time, event_queue, forecast, runway_availability)
        
        # Apply optimized times to schedules
        if result:
            for schedule in schedules:
                if schedule.flight.flight_id in result:
                    optimized_time = result[schedule.flight.flight_id]
                    if schedule.is_takeoff:
                        schedule.etd = optimized_time
                    else:
                        schedule.eta = optimized_time
                    debug(f"Applied optimized time: {schedule.flight.flight_id} -> {int_to_hhmm_colon(optimized_time)}")
        
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
    
    def greedy(self, schedules, current_time, event_queue=None, forecast=None, runway_availability=None):
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

        # 활주로별 사용 가능 시간 추적 (실제 상태 반영)
        # Track by runway direction to ensure proper separation
        runway_usage = {'14L': 0, '14R': 0, '32L': 0, '32R': 0}
        
        # Initialize with actual next_available_time if provided
        if runway_availability:
            for runway in self.sim.airport.runways:
                if runway.name in runway_availability:
                    runway_usage[runway.name] = max(runway_usage[runway.name], runway_availability[runway.name])
                    runway_usage[runway.inverted_name] = max(runway_usage[runway.inverted_name], runway_availability[runway.name])
        
        # Process takeoff schedules
        for schedule in takeoff_schedules:
            # Find earliest available time for takeoff
            earliest_time = max(current_time, schedule.etd or current_time)
            
            # Try to find available runway (14L/32R for takeoff)
            assigned_runway = None
            assigned_time = None
            
            for runway in self.sim.airport.runways:
                current_direction = runway.get_current_direction()
                if current_direction in ["14L", "32R"]:  # Takeoff runways
                    # Check if runway is closed
                    is_closed = False
                    for closure in runway_closures:
                        if (closure['runway'] == runway.name or closure['runway'] == runway.inverted_name) and \
                           closure['start_time'] <= earliest_time <= closure['end_time']:
                            is_closed = True
                            break
                    
                    if not is_closed:
                        # Check if runway is available at this time (respect separation)
                        available_time = max(earliest_time, runway_usage[current_direction])
                        
                        # Check if runway can handle operation
                        if runway.can_handle_operation(available_time):
                            assigned_runway = runway
                            assigned_time = available_time
                            break
            
            if assigned_runway and assigned_time:
                schedule.runway = assigned_runway
                schedule.etd = assigned_time
                # Update runway usage with 4-minute separation
                runway_usage[assigned_runway.get_current_direction()] = assigned_time + 4
                debug(f"{schedule.flight.flight_id} 이륙: {int_to_hhmm_colon(assigned_time)} 활주로 {assigned_runway.get_current_direction()}")
        
        # Process landing schedules
        for schedule in landing_schedules:
            # Find earliest available time for landing
            earliest_time = max(current_time, schedule.eta or current_time)
            
            # Try to find available runway (14R/32L for landing)
            assigned_runway = None
            assigned_time = None
            
            for runway in self.sim.airport.runways:
                current_direction = runway.get_current_direction()
                if current_direction in ["14R", "32L"]:  # Landing runways
                    # Check if runway is closed
                    is_closed = False
                    for closure in runway_closures:
                        if (closure['runway'] == runway.name or closure['runway'] == runway.inverted_name) and \
                           closure['start_time'] <= earliest_time <= closure['end_time']:
                            is_closed = True
                            break
                    
                    if not is_closed:
                        # Check if runway is available at this time (respect separation)
                        available_time = max(earliest_time, runway_usage[current_direction])
                        
                        # Check if runway can handle operation
                        if runway.can_handle_operation(available_time):
                            assigned_runway = runway
                            assigned_time = available_time
                            break
            
            if assigned_runway and assigned_time:
                schedule.runway = assigned_runway
                schedule.eta = assigned_time
                # Update runway usage with 4-minute separation
                runway_usage[assigned_runway.get_current_direction()] = assigned_time + 4
                debug(f"{schedule.flight.flight_id} 착륙: {int_to_hhmm_colon(assigned_time)} 활주로 {assigned_runway.get_current_direction()}")
            else:
                # Landing failed - go-around
                self.sim.event_handler._go_around(schedule.flight.flight_id)
                debug(f"{schedule.flight.flight_id} 착륙 실패: {int_to_hhmm_colon(earliest_time)} - go-around 발생")
        
        # Return result for consistency with other algorithms
        result = {}
        for schedule in takeoff_schedules + landing_schedules:
            if schedule.is_takeoff and schedule.etd is not None:
                result[schedule.flight.flight_id] = schedule.etd
            elif not schedule.is_takeoff and schedule.eta is not None:
                result[schedule.flight.flight_id] = schedule.eta
        
        return result
    
    def ml(self, schedules, current_time, event_queue=None, forecast=None, runway_availability=None):
        """ML 알고리즘 (향후 구현)"""
        # TODO: ML 모델 적용
        return {}
    
    def rl(self, schedules, current_time, event_queue=None, forecast=None, runway_availability=None):
        """RL 알고리즘 (향후 구현)"""
        # TODO: 강화학습 에이전트 적용
        return {}

    def advanced(self, schedules, current_time, event_queue=None, forecast=None, runway_availability=None):
        """Advanced scheduler using MILP optimization"""
        debug(f"Advanced scheduler algorithm started at {int_to_hhmm_colon(current_time)}")
        
        # Import AdvancedScheduler here to avoid circular imports
        from sim.advanced_scheduler import AdvancedScheduler
        
        # Create a temporary advanced scheduler instance
        advanced_scheduler = AdvancedScheduler(self.sim)
        
        # Run the advanced optimization
        result = advanced_scheduler.optimize(schedules, current_time, event_queue, forecast, runway_availability)
        
        # Track runway usage to ensure separation
        runway_usage = {'14L': 0, '14R': 0, '32L': 0, '32R': 0}
        
        # Initialize runway usage with actual next_available_time if provided
        if runway_availability:
            for runway in self.sim.airport.runways:
                if runway.name in runway_availability:
                    runway_usage[runway.name] = max(runway_usage[runway.name], runway_availability[runway.name])
                    runway_usage[runway.inverted_name] = max(runway_usage[runway.inverted_name], runway_availability[runway.name])
        
        # Assign runways to schedules based on the result
        # This is needed because the advanced scheduler returns times but doesn't assign runways
        for schedule in schedules:
            if schedule.flight.flight_id in result:
                assigned_time = result[schedule.flight.flight_id]
                
                # Assign appropriate runway based on operation type and availability
                if schedule.is_takeoff:
                    # Find takeoff runway (14L or 32R) that can handle the operation
                    for runway in self.sim.airport.runways:
                        current_direction = runway.get_current_direction()
                        if current_direction in ["14L", "32R"] and runway.can_handle_operation(assigned_time):
                            # Check if runway is available at this time (respect separation)
                            if assigned_time >= runway_usage[current_direction]:
                                schedule.runway = runway
                                runway_usage[current_direction] = assigned_time + 4  # 4-minute separation
                                break
                else:
                    # Find landing runway (14R or 32L) that can handle the operation
                    for runway in self.sim.airport.runways:
                        current_direction = runway.get_current_direction()
                        if current_direction in ["14R", "32L"] and runway.can_handle_operation(assigned_time):
                            # Check if runway is available at this time (respect separation)
                            if assigned_time >= runway_usage[current_direction]:
                                schedule.runway = runway
                                runway_usage[current_direction] = assigned_time + 4  # 4-minute separation
                                break
        
        return result

    def _debug_schedule_times(self, schedules, current_time, changes=None):
        """모든 스케줄의 배정된 시간을 순서대로 출력 (변경사항 반영)"""
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
