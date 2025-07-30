from utils.logger import debug
from sim.flight import FlightStatus
from sim.schedule import Schedule
import heapq
import time
from utils.time_utils import int_to_hhmm_colon

class Scheduler:
    def __init__(self, algorithm="greedy", sim=None):
        self.algorithm = algorithm
        self.sim = sim
    
    def optimize(self, schedules, current_time, event_queue, runway_availability, forecast_data):
        """스케줄러 최적화"""
        if self.algorithm == "rl":
            return self.rl_optimize(schedules, current_time, event_queue, runway_availability, forecast_data)
        elif self.algorithm == "greedy":
            return self.greedy(schedules, current_time, event_queue, runway_availability, forecast_data)
        elif self.algorithm == "advanced":
            return self.advanced(schedules, current_time, event_queue, runway_availability, forecast_data)
        else:
            debug(f"알 수 없는 알고리즘: {self.algorithm}")
            return False
    
    def rl_optimize(self, schedules, current_time, event_queue, runway_availability, forecast_data):
        """PPO 에이전트를 사용한 스케줄링 - 모든 스케줄을 한번에 배정"""
        if not hasattr(self.sim, 'rl_agent') or self.sim.rl_agent is None:
            debug("RL 에이전트가 설정되지 않았습니다.")
            return False
        
        # 현재 상태 관찰 (모든 정보 포함)
        current_state = self.sim._get_current_state()
        
        # 모든 스케줄을 대상으로 함 (DORMANT, WAITING 상태)
        all_schedules = [s for s in schedules 
                        if s.status in [FlightStatus.DORMANT, FlightStatus.WAITING]]
        
        if not all_schedules:
            debug("RL: 배정 가능한 스케줄이 없습니다.")
            return False
        
        debug(f"RL: {len(all_schedules)}개 스케줄에 대해 배정 시작")
        
        # PPO 에이전트가 모든 스케줄에 대해 액션 선택
        actions, action_probs, value = self.sim.rl_agent.select_action(current_state, len(all_schedules))
        
        # 모든 스케줄에 대해 액션 적용
        assigned_count = 0
        failed_count = 0
        
        for i, action in enumerate(actions):
            if i < len(all_schedules):
                schedule = all_schedules[i]
                success = self._apply_ppo_action_with_constraints(
                    schedule, action, current_time, runway_availability, forecast_data
                )
                if success:
                    assigned_count += 1
                    debug(f"RL: {schedule.flight.flight_id} 배정 성공")
                else:
                    failed_count += 1
                    debug(f"RL: {schedule.flight.flight_id} 배정 실패")
        
        debug(f"RL: {assigned_count}개 배정 성공, {failed_count}개 실패")
        return assigned_count > 0
    
    def _apply_ppo_action_with_constraints(self, schedule, action, current_time, 
                                         runway_availability, forecast_data):
        """PPO 액션을 제약 조건을 고려하여 스케줄에 적용"""
        # 액션 해석: [runway_choice, time_choice, landing_decision]
        runway_choice = (action // 144) % 2  # 0: 14L, 1: 14R
        time_choice = action % 144  # 0~143: 시간 선택
        landing_decision = (action // 288) % 2  # 0: landing, 1: go_around (착륙의 경우)
        
        runway_name = "14L" if runway_choice == 0 else "14R"
        
        # 비행의 원래 ETD/ETA를 기준으로 시간 선택
        if schedule.is_takeoff:
            original_etd = schedule.flight.etd
            if original_etd is None:
                return False
            
            # ETD를 분 단위로 변환
            if isinstance(original_etd, str):
                if ':' in original_etd:
                    hour, minute = map(int, original_etd.split(':'))
                else:
                    hour = int(original_etd) // 100
                    minute = int(original_etd) % 100
                etd_minutes = hour * 60 + minute
            else:
                etd_minutes = original_etd
            
            # ETD 기준으로 직접 시간 선택 (0~143분)
            selected_time = etd_minutes + time_choice
            
        else:
            # 착륙: ETA 기준으로 시간 선택 + 착륙 결정
            original_eta = schedule.flight.eta
            if original_eta is None:
                return False
            
            # ETA를 분 단위로 변환
            if isinstance(original_eta, str):
                if ':' in original_eta:
                    hour, minute = map(int, original_eta.split(':'))
                else:
                    hour = int(original_eta) // 100
                    minute = int(original_eta) % 100
                eta_minutes = hour * 60 + minute
            else:
                eta_minutes = original_eta
            
            # ETA 기준으로 직접 시간 선택 (0~143분)
            selected_time = eta_minutes + time_choice
            
            # Go-around 결정
            if landing_decision == 1:  # go_around
                from sim.event import Event
                go_around_event = Event(
                    event_type="GO_AROUND", 
                    target_type="", 
                    target=schedule.flight.flight_id, 
                    time=selected_time, 
                    duration=0
                )
                self.sim.event_queue.append(go_around_event)
                return True
        
        # 활주로 폐쇄 확인
        if not self._check_runway_closure(runway_name, selected_time):
            return False
        
        # 스케줄 할당
        try:
            if schedule.is_takeoff:
                schedule.etd = selected_time
            else:
                schedule.eta = selected_time
            
            # 활주로 객체 할당
            runway_obj = next((r for r in self.sim.airport.runways if r.name == runway_name), None)
            if runway_obj:
                schedule.runway = runway_obj
            else:
                return False
            
            return True
            
        except Exception as e:
            debug(f"스케줄 할당 실패: {e}")
            return False
    
    def _check_runway_closure(self, runway_name, selected_time):
        """활주로 닫힘 여부만 검사"""
        # 활주로가 닫혀있는지 확인
        runway = next((r for r in self.sim.airport.runways if r.name == runway_name), None)
        if runway and runway.closed:
            return False
        
        # 해당 시간에 활주로가 닫히는 이벤트가 있는지 확인
        for event in self.sim.event_queue:
            if event.event_type == 'RUNWAY_CLOSURE':
                if event.target == runway_name:
                    closure_start = event.time
                    closure_end = event.time + event.duration
                    if closure_start <= selected_time <= closure_end:
                        return False
        
        return True
    
    def greedy(self, schedules, current_time, event_queue, runway_availability, forecast_data):
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

        # 활주로별 사용 가능 시간 추적
        runway_usage = {}
        
        # runway_availability에서 시작 시간 설정
        if runway_availability:
            # runway_availability는 {runway.name: next_available_time} 형태
            for runway in self.sim.airport.runways:
                if runway.name in runway_availability:
                    runway_usage[runway.name] = runway_availability[runway.name]
                else:
                    runway_usage[runway.name] = current_time
        else:
            for runway in self.sim.airport.runways:
                runway_usage[runway.name] = current_time

        # 현재 시간 = min(current_time, available time들 중 작은거)
        min_available_time = min(runway_usage.values()) if runway_usage else current_time
        current_time = min(current_time, min_available_time)

        # 모든 스케줄이 처리될 때까지 반복
        while takeoff_schedules or landing_schedules:
            
            # 현재 시간이 runway_closed_time 범위 안에 들어가는지 확인
            for closure in runway_closures:
                if closure['start_time'] <= current_time <= closure['end_time']:
                    current_time = closure['end_time']
                    debug(f"Runway closed, jumping to {int_to_hhmm_colon(current_time)}")
                    break
            
            # 착륙 가능 여부와 이륙 가능 여부 따로 판단
            landing_possible = False
            takeoff_possible = False
            
            # 착륙 가능 여부 확인 (14R)
            if landing_schedules:
                # 14R이 closed인지 확인
                runway_14r_closed = False
                for closure in runway_closures:
                    if closure['runway'] == "14R" and closure['start_time'] <= current_time <= closure['end_time']:
                        runway_14r_closed = True
                        break
                
                # 14R 활주로 찾기
                runway_14r = next((r for r in self.sim.airport.runways if r.name == "14R"), None)
                if runway_14r and not runway_14r_closed and current_time >= runway_usage.get(runway_14r.name, current_time):
                    # 착륙 스케줄 중 ETA <= current_time인 스케줄이 있는지 확인
                    available_landings = [s for s in landing_schedules if s.eta <= current_time]
                    if available_landings:
                        landing_possible = True
            
            # 이륙 가능 여부 확인 (14L)
            if takeoff_schedules:
                # 14L이 closed인지 확인
                runway_14l_closed = False
                for closure in runway_closures:
                    if closure['runway'] == "14L" and closure['start_time'] <= current_time <= closure['end_time']:
                        runway_14l_closed = True
                        break
                
                # 14L 활주로 찾기
                runway_14l = next((r for r in self.sim.airport.runways if r.name == "14L"), None)
                if runway_14l and not runway_14l_closed and current_time >= runway_usage.get(runway_14l.name, current_time):
                    # 이륙 스케줄 중 ETD <= current_time인 스케줄이 있는지 확인
                    available_takeoffs = [s for s in takeoff_schedules if s.flight.etd <= current_time]
                    if available_takeoffs:
                        takeoff_possible = True
            
            # 둘 중 하나 배정
            if landing_possible:
                debug(f"Processing landing at {int_to_hhmm_colon(current_time)}")
                
                # 착륙 스케줄 중 ETA <= current_time인 스케줄 중 가장 높은 priority 선택
                available_landings = [s for s in landing_schedules if s.eta <= current_time]
                selected_landing = max(available_landings, key=lambda s: s.priority)
                
                # 14R 활주로 할당
                for runway in self.sim.airport.runways:
                    if runway.name == "14R":
                        selected_landing.runway = runway
                        break
                
                selected_landing.eta = current_time
                landing_schedules.remove(selected_landing)
                runway_usage[runway_14r.name] = current_time + 4  # 4분 간격
                
                debug(f"{selected_landing.flight.flight_id} 착륙: {int_to_hhmm_colon(current_time)} 활주로 14R")
                
            elif takeoff_possible:
                debug(f"Processing takeoff at {int_to_hhmm_colon(current_time)}")
                
                # 이륙 스케줄 중 ETD <= current_time인 스케줄 중 가장 높은 priority 선택
                available_takeoffs = [s for s in takeoff_schedules if s.flight.etd <= current_time]
                selected_takeoff = max(available_takeoffs, key=lambda s: s.priority)
                
                # 14L 활주로 할당
                for runway in self.sim.airport.runways:
                    if runway.name == "14L":
                        selected_takeoff.runway = runway
                        break
                
                selected_takeoff.etd = current_time
                takeoff_schedules.remove(selected_takeoff)
                runway_usage[runway_14l.name] = current_time + 4  # 4분 간격
                
                debug(f"{selected_takeoff.flight.flight_id} 이륙: {int_to_hhmm_colon(current_time)} 활주로 14L")
                
            else:
                # 둘 다 배정 불가능 - 시간 증가
                current_time += 1
                debug(f"No assignment possible at {int_to_hhmm_colon(current_time-1)}, moving to {int_to_hhmm_colon(current_time)}")
            
            # 무한 루프 방지
            max_available_time = max(runway_usage.values()) if runway_usage else current_time
            if current_time > max_available_time + 720:  # 12시간 제한
                debug("Time limit reached, forcing go-around for remaining schedules")
                break
        
        # 남은 착륙 스케줄들을 go_around로 처리
        for landing_schedule in landing_schedules:
            from sim.event import Event
            go_around_event = Event(
                event_type="GO_AROUND",
                target_type="",
                target=landing_schedule.flight.flight_id,
                time=current_time,
                duration=0
            )
            self.sim.event_queue.append(go_around_event)
            debug(f"{landing_schedule.flight.flight_id} go_around 이벤트 생성")
        
        return True
    
    def advanced(self, schedules, current_time, event_queue, runway_availability, forecast_data):
        """Advanced scheduler using MILP optimization"""
        debug(f"Advanced scheduler algorithm started at {int_to_hhmm_colon(current_time)}")
        
        # Import AdvancedScheduler here to avoid circular imports
        from sim.advanced_scheduler import AdvancedScheduler
        
        # Create a temporary advanced scheduler instance
        advanced_scheduler = AdvancedScheduler(self.sim)
        
        # Run the advanced optimization
        result = advanced_scheduler.optimize(schedules, current_time, event_queue, forecast_data, runway_availability)
        
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
