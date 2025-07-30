"""
RL Environment for Airport Scheduling
Provides the interface between the simulation and RL agent
"""

import numpy as np
from typing import Dict, Any, List
from sim.flight import FlightStatus

class AirportEnvironment:
    """Airport scheduling environment for RL"""
    
    def __init__(self, simulation):
        self.sim = simulation
        
        # Observation space size (calculated based on state features)
        # This should match the state size in simulation._get_current_state()
        self.observation_space_size = self._calculate_observation_size()
        
        # Action space size: 
        # 이륙: 281(시간 0~280분) * 2(14L, 14R) = 562개
        # 착륙: 6(시간 -2~+2, go_around) * 2(14L, 14R) = 12개
        # 총: 562 + 12 = 574개 액션
        self.action_space_size = 574
    
    def _calculate_observation_size(self) -> int:
        """Calculate the size of the observation space"""
        # 새로운 상태 벡터 크기 계산:
        # 1. 시간 정보: 1
        # 2. 활주로 상태: 2 (14L, 14R 다음 가용 시간)
        # 3. 날씨 예보: 24 * 2 = 48 (24개 시점 * 2개 특성)
        # 4. 스케줄 정보: 50 * 5 = 250 (50개 스케줄 * 5개 특성)
        # 5. 이벤트 정보: 10 * 3 = 30 (10개 이벤트 * 3개 특성)
        # 총합: 1 + 2 + 48 + 250 + 30 = 331
        
        size = 1 + 2 + 48 + 250 + 30  # = 331
        
        return size
    
    def _get_observation(self) -> np.ndarray:
        """Get current state observation"""
        # Get the full state from simulation
        full_state = self.sim._get_current_state()
        
        # The new model expects exactly 331 features
        # If we have more than 331, truncate; if less, pad with zeros
        if len(full_state) > 331:
            return full_state[:331]
        elif len(full_state) < 331:
            # Pad with zeros to reach 331
            padded_state = np.zeros(331)
            padded_state[:len(full_state)] = full_state
            return padded_state
        else:
            return full_state
    
    def _apply_single_schedule_action(self, schedule, action: int) -> float:
        """Apply a single schedule action and return reward"""
        # 이륙과 착륙에 따라 다른 액션 해석
        if schedule.is_takeoff:
            # 이륙 액션: 0~561 (562개)
            if action >= 562:
                return 0.0  # 잘못된 액션
            
            # 이륙 액션 해석
            delay_choice = action // 2  # 0~280 (281개)
            runway_choice = action % 2  # 0: 14L, 1: 14R
            
            # 시간 계산: ETD + delay
            actual_time = schedule.flight.etd + delay_choice
            runway_name = "14L" if runway_choice == 0 else "14R"
            
        else:
            # 착륙 액션: 562~573 (12개)
            if action < 562 or action >= 574:
                return 0.0  # 잘못된 액션
            
            # 착륙 액션 해석 (0~11로 변환)
            landing_action = action - 562
            
            time_choice = landing_action // 2  # 0: -2, 1: -1, 2: 0, 3: +1, 4: +2, 5: go_around
            runway_choice = landing_action % 2  # 0: 14L, 1: 14R
            
            # 시간 계산: ETA ±2분 또는 go_around
            if time_choice == 5:  # go_around
                actual_time = None
                runway_name = None
            else:
                time_offset = time_choice - 2  # -2, -1, 0, +1, +2
                actual_time = schedule.eta + time_offset
                runway_name = "14L" if runway_choice == 0 else "14R"
        
        # Get current runway availability
        runway_availability = {}
        for runway in self.sim.airport.runways:
            runway_availability[runway.name] = runway.next_available_time
        
        # 직접 액션 적용
        success = False
        if schedule.is_takeoff:
            if actual_time is not None and runway_name is not None:
                # 이륙 배정
                for runway in self.sim.airport.runways:
                    if runway.name == runway_name:
                        schedule.runway = runway
                        schedule.etd = actual_time
                        success = True
                        break
        else:
            if actual_time is None and runway_name is None:
                # go_around
                from sim.event import Event
                go_around_event = Event(
                    event_type="GO_AROUND",
                    target_type="",
                    target=schedule.flight.flight_id,
                    time=self.sim.time,
                    duration=0
                )
                self.sim.event_queue.append(go_around_event)
                success = True
            elif actual_time is not None and runway_name is not None:
                # 착륙 배정
                for runway in self.sim.airport.runways:
                    if runway.name == runway_name:
                        schedule.runway = runway
                        schedule.eta = actual_time
                        success = True
                        break
        
        # Simulation에서 보상 계산하므로 여기서는 성공/실패만 반환
        return 1.0 if success else 0.0
