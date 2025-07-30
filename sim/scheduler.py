from utils.logger import debug
from sim.flight import FlightStatus
from sim.schedule import Schedule
import heapq
from utils.time_utils import int_to_hhmm_colon

class Scheduler:
    def __init__(self, algorithm="greedy", sim=None):
        self.algorithm = algorithm
        self.sim = sim
        self.last_actions = []  # 마지막 선택한 액션들
        self.last_action_probs = []  # 마지막 액션 확률들
    
    def get_actions(self):
        """마지막 선택한 액션들 반환"""
        return self.last_actions
    
    def get_action_probs(self):
        """마지막 액션 확률들 반환"""
        return self.last_action_probs
    
    def get_value(self):
        """마지막 value 반환"""
        return getattr(self, 'last_value', 0.0)
    
    def optimize(self, schedules, current_time, event_queue=None, forecast=None, runway_availability=None):
        """ event_queue : 관측 가능한 예정된 모든 이벤트 (지금은 RWY_CLOSUIRE, RWY_INVERT만 존재) """
        """ 스케줄 최적화 메인 메서드 """
        match self.algorithm:
            case "greedy":
                result = self.greedy(schedules, current_time, event_queue, forecast, runway_availability)
            case "advanced":
                result = self.advanced(schedules, current_time, event_queue, forecast, runway_availability)
            case "rl":
                result = self.rl(schedules, current_time, event_queue, forecast, runway_availability)
            case _:
                result = self.greedy(schedules, current_time, event_queue, forecast, runway_availability)
        
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
    
    def _apply_ppo_action_with_constraints(self, schedule, action, current_time, 
                                         runway_availability, forecast):
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
            
            # 착륙 결정 적용
            if landing_decision == 1:  # go_around
                # go_around 이벤트 생성 (Event 객체로 생성)
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
        
        # 활주로 제약 조건 검사 (닫힘 여부만)
        if not self._check_runway_closure(runway_name, selected_time):
            return False
        
        # 스케줄 배정
        try:
            if schedule.is_takeoff:
                schedule.etd = selected_time
                schedule.eta = selected_time
            
            # Runway 객체 찾아서 할당
            runway_obj = next((r for r in self.sim.airport.runways if r.name == runway_name), None)
            if runway_obj:
                schedule.runway = runway_obj
            else:
                debug(f"활주로 객체를 찾을 수 없음: {runway_name}")
                return False
            
            return True
        except Exception as e:
            debug(f"스케줄 배정 실패: {e}")
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
                    next_available = runway_availability[runway.name]
                    runway_usage[runway.name] = max(runway_usage[runway.name], next_available)
                    runway_usage[runway.inverted_name] = max(runway_usage[runway.inverted_name], next_available)
                    debug(f"Runway {runway.name} next_available_time: {next_available} -> runway_usage[{runway.name}]: {runway_usage[runway.name]}")
        else:
            # runway_availability가 없으면 current_time으로 초기화
            for runway in self.sim.airport.runways:
                runway_usage[runway.name] = max(runway_usage[runway.name], current_time)
                runway_usage[runway.inverted_name] = max(runway_usage[runway.inverted_name], current_time)
        
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
    
    def rl(self, schedules, current_time, event_queue, forecast, runway_availability):
        """RL 알고리즘 - Using trained PPO agent"""
        debug("RL algorithm started")
        
        try:
            from rl.environment import AirportEnvironment
            
            # Initialize RL environment if not already done
            if not hasattr(self, 'rl_env'):
                self.rl_env = AirportEnvironment(self.sim)
            
            # simulation에서 설정된 rl_agent 사용
            if not self.sim.rl_agent:
                debug("RL 에이전트가 설정되지 않았습니다. Best 모델을 자동으로 로드합니다")
                try:
                    import os
                    from rl.agent import PPOAgent
                    
                    # MODEL PATH
                    model_path = "models/ppo_best.pth"
                    if os.path.exists(model_path):
                        # RL 환경과 에이전트 생성
                        rl_agent = PPOAgent(
                            observation_size=self.rl_env.observation_space_size,
                            action_size=self.rl_env.action_space_size
                        )
                        
                        # 훈련된 모델 로드
                        rl_agent.load_model(model_path)
                        self.sim.set_rl_agent(rl_agent)
                        debug(f"Best 모델을 로드했습니다: {model_path}")
                    else:
                        debug(f"모델 파일을 찾을 수 없습니다: {model_path}. Greedy 알고리즘으로 fallback합니다")
                        return self.greedy(schedules, current_time, event_queue, forecast, runway_availability)
                except Exception as e:
                    debug(f"모델 로드 중 오류 발생: {e}. Greedy 알고리즘으로 fallback합니다")
                    return self.greedy(schedules, current_time, event_queue, forecast, runway_availability)
            
            rl_agent = self.sim.rl_agent
            # Get current state observation
            state = self.rl_env._get_observation()
            
            # Get available schedules for RL
            available_schedules = [s for s in schedules if s.status in [FlightStatus.DORMANT, FlightStatus.WAITING]]
            num_schedules = len(available_schedules)
            
            if num_schedules == 0:
                debug("RL: 배정 가능한 스케줄이 없습니다")
                return {}
            
            # Select actions using RL agent
            actions, action_probs, value = rl_agent.select_action(state, num_schedules)
            
            # 액션과 확률, value 저장 (나중에 simulation에서 사용)
            self.last_actions = actions
            self.last_action_probs = action_probs
            self.last_value = value
            
            total_reward = 0.0
            assigned_count = 0
            result = {}
            
            # Apply actions to schedules
            for i, schedule in enumerate(available_schedules):
                if i < len(actions):
                    action = actions[i]
                    reward = self.rl_env._apply_single_schedule_action(schedule, action)
                    total_reward += reward
                    
                    if reward > 0:  # Successful assignment
                        assigned_count += 1
                        # Add to result for consistency with other algorithms
                        if schedule.is_takeoff and schedule.etd is not None:
                            result[schedule.flight.flight_id] = schedule.etd
                        elif not schedule.is_takeoff and schedule.eta is not None:
                            result[schedule.flight.flight_id] = schedule.eta
            
            debug(f"RL 액션 적용 완료: {assigned_count}/{num_schedules} 비행 배정, 총 보상: {total_reward:.2f}")
            
            return result
        except Exception as e:
            debug(f"RL 알고리즘 실행 중 오류 발생: {e}")
            debug("Greedy 알고리즘으로 fallback합니다")
            return self.greedy(schedules, current_time, event_queue, forecast, runway_availability)
    
    def _apply_rl_scheduling_action(self, action, schedules, current_time, event_queue, runway_availability):
        """RL 액션을 실제 스케줄링 결정으로 적용 - 환경과 동일한 로직"""
        # 대기 중인 스케줄들
        pending_schedules = [s for s in schedules if s.status == FlightStatus.DORMANT or s.status == FlightStatus.WAITING]
        
        if not pending_schedules:
            return False
        
        # 액션 해석: [runway_choice, time_choice, operation_type]
        runway_choice = (action // 6) % 3  # 0: 14L, 1: 14R, 2: wait
        time_choice = action % 6  # 0: -2, 1: -1, 2: 0, 3: +1, 4: +2, 5: go_around
        operation_type = (action // 18) % 2  # 0: takeoff, 1: landing
        
        # Use the first pending schedule (the environment will handle multiple schedules)
        schedule = pending_schedules[0]
        
        # Check if operation type matches schedule type
        if (operation_type == 0 and not schedule.is_takeoff) or (operation_type == 1 and schedule.is_takeoff):
            return False  # Operation type mismatch
        
        # 활주로 가용성 확인
        runway_usage = {}
        if runway_availability:
            runway_usage["14L"] = runway_availability.get("14L", current_time)
            runway_usage["14R"] = runway_availability.get("14R", current_time)
        else:
            runway_usage["14L"] = current_time
            runway_usage["14R"] = current_time
        
        if runway_choice == 0:  # 14L
            if schedule.is_takeoff:
                # 이륙: 14L 활주로 사용
                if self._is_runway_available_for_rl("14L", schedule, time_choice, runway_usage):
                    success = self._assign_takeoff_rl(schedule, "14L", time_choice, current_time)
                    return success
            else:
                # 착륙: 14L 활주로는 14R이 closed일 때만 사용 가능
                if self._is_14r_closed_rl(event_queue):
                    if self._is_runway_available_for_rl("14L", schedule, time_choice, runway_usage):
                        success = self._assign_landing_rl(schedule, "14L", time_choice, current_time)
                        return success
                        
        elif runway_choice == 1:  # 14R
            if not schedule.is_takeoff:
                # 착륙: 14R 활주로 사용
                if self._is_runway_available_for_rl("14R", schedule, time_choice, runway_usage):
                    success = self._assign_landing_rl(schedule, "14R", time_choice, current_time)
                    return success
            else:
                # 이륙: 14R 활주로는 14L이 closed일 때만 사용 가능
                if self._is_14l_closed_rl(event_queue):
                    if self._is_runway_available_for_rl("14R", schedule, time_choice, runway_usage):
                        success = self._assign_takeoff_rl(schedule, "14R", time_choice, current_time)
                        return success
        
        # runway_choice == 2: 대기 (아무것도 하지 않음)
        return False
    
    def _is_runway_available_for_rl(self, runway_name: str, schedule: Schedule, time_choice: int, runway_usage: dict) -> bool:
        """활주로 사용 가능 여부 확인 (RL용)"""
        # 시간 계산
        if schedule.is_takeoff:
            base_time = max(schedule.flight.etd, runway_usage.get(runway_name, 0))
        else:
            base_time = schedule.eta or runway_usage.get(runway_name, 0)
        
        # 시간 선택에 따른 실제 시간
        if time_choice == 0:  # -2
            actual_time = base_time - 2
        elif time_choice == 1:  # -1
            actual_time = base_time - 1
        elif time_choice == 2:  # 0
            actual_time = base_time
        elif time_choice == 3:  # +1
            actual_time = base_time + 1
        elif time_choice == 4:  # +2
            actual_time = base_time + 2
        else:  # go_around
            return True  # go_around는 항상 가능
        
        # 활주로 가용성 확인
        return actual_time >= runway_usage.get(runway_name, 0)
    
    def _assign_takeoff_rl(self, schedule: Schedule, runway_name: str, time_choice: int, current_time: int) -> bool:
        """이륙 배정 (RL용)"""
        if time_choice == 5:  # go_around
            return False
        
        # 시간 계산
        base_time = max(schedule.flight.etd, current_time)
        if time_choice == 0:  # -2
            actual_time = base_time - 2
        elif time_choice == 1:  # -1
            actual_time = base_time - 1
        elif time_choice == 2:  # 0
            actual_time = base_time
        elif time_choice == 3:  # +1
            actual_time = base_time + 1
        elif time_choice == 4:  # +2
            actual_time = base_time + 2
        
        # 활주로 배정
        for runway in self.sim.airport.runways:
            if runway.name == runway_name:
                schedule.runway = runway
                schedule.etd = actual_time
                debug(f"RL 이륙 배정: {schedule.flight.flight_id} -> {runway_name} {int_to_hhmm_colon(actual_time)}")
                return True
        
        return False
    
    def _assign_landing_rl(self, schedule: Schedule, runway_name: str, time_choice: int, current_time: int) -> bool:
        """착륙 배정 (RL용)"""
        if time_choice == 5:  # go_around
            # go_around 처리
            self.sim.event_handler._go_around(schedule.flight.flight_id)
            debug(f"RL go_around: {schedule.flight.flight_id}")
            return True
        
        # 시간 계산
        base_time = schedule.eta or current_time
        if time_choice == 0:  # -2
            actual_time = base_time - 2
        elif time_choice == 1:  # -1
            actual_time = base_time - 1
        elif time_choice == 2:  # 0
            actual_time = base_time
        elif time_choice == 3:  # +1
            actual_time = base_time + 1
        elif time_choice == 4:  # +2
            actual_time = base_time + 2
        
        # 활주로 배정
        for runway in self.sim.airport.runways:
            if runway.name == runway_name:
                schedule.runway = runway
                schedule.eta = actual_time
                debug(f"RL 착륙 배정: {schedule.flight.flight_id} -> {runway_name} {int_to_hhmm_colon(actual_time)}")
                return True
        
        return False
    
    def _is_14l_closed_rl(self, event_queue) -> bool:
        """14L 활주로 폐쇄 여부 확인 (RL용)"""
        if event_queue:
            for event in event_queue:
                if event.event_type == "RUNWAY_CLOSURE" and event.target == "14L":
                    return True
        return False
    
    def _is_14r_closed_rl(self, event_queue) -> bool:
        """14R 활주로 폐쇄 여부 확인 (RL용)"""
        if event_queue:
            for event in event_queue:
                if event.event_type == "RUNWAY_CLOSURE" and event.target == "14R":
                    return True
        return False

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