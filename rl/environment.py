"""
RL Environment for Airport Scheduling
Provides the interface between the simulation and RL agent
"""

import numpy as np
from typing import Dict, Any, List, Tuple
from sim.flight import FlightStatus

class AirportEnvironment:
    """Airport scheduling environment for RL"""
    
    def __init__(self, simulation):
        self.sim = simulation
        
        # MultiDiscrete 액션 공간 정의
        # [time_choice, runway_choice]
        # time_choice: 0~180 (181개) - 지연 시간 (분)
        # runway_choice: 0~1 (2개) - 0: 14L, 1: 14R
        self.action_space = [181, 2]  # MultiDiscrete
        self.action_space_size = 181 * 2  # 총 액션 수 (호환성용)
        
        # Observation space size 계산
        self.observation_space_size = self._calculate_observation_size()
    
    def _calculate_observation_size(self) -> int:
        """Calculate the size of the observation space"""
        # 새로운 상태 벡터 크기 계산:
        # 1. 시간 정보: 1
        # 2. 활주로 상태: 2 (14L, 14R 다음 가용 시간)
        # 3. 날씨 예보: 24 * 2 = 48 (24개 시점 * 2개 특성)
        # 4. 스케줄 정보: 50 * 5 = 250 (50개 스케줄 * 5개 특성)
        # 5. 이벤트 정보: 10 * 3 = 30 (10개 이벤트 * 3개 특성)
        # 6. 현재 배정할 비행 정보: 6 (flight_id, etd/eta, priority, is_takeoff, status, runway)
        # 총합: 1 + 2 + 48 + 250 + 30 + 6 = 337
        
        size = 1 + 2 + 48 + 250 + 30 + 6  # = 337
        
        return size
    
    def _get_observation_for_schedule(self, schedule) -> np.ndarray:
        """Get current state observation with specific schedule information"""
        # 기본 상태 가져오기 (331차원)
        base_state = self.sim._get_current_state()
        
        # 현재 배정할 비행 정보 추가 (6차원)
        flight_info = self._get_flight_info(schedule)
        
        # 전체 observation 구성 (331 + 6 = 337차원)
        full_observation = np.concatenate([base_state, flight_info])
        
        # 크기 확인 (디버깅용)
        if len(full_observation) != self.observation_space_size:
            print(f"🚨 Observation 크기 불일치: {len(full_observation)} vs {self.observation_space_size}")
        
        return full_observation
    
    def _get_flight_info(self, schedule) -> np.ndarray:
        """Get information about the specific flight to be scheduled"""
        flight_info = np.zeros(6)
        
        # 0: flight_id (normalized)
        flight_info[0] = float(hash(schedule.flight.flight_id) % 1000) / 1000.0
        
        # 1: original ETD/ETA (normalized to 0-1)
        if schedule.is_takeoff:
            original_time = schedule.flight.etd
        else:
            original_time = schedule.eta
        
        if original_time is not None:
            # 시간을 0-1 범위로 정규화 (0600-2200 = 360-1320)
            flight_info[1] = (original_time - 360) / (1320 - 360)
        else:
            flight_info[1] = 0.5  # 기본값
        
        # 2: priority (normalized)
        flight_info[2] = schedule.priority / 64.0  # 최대 priority로 정규화
        
        # 3: is_takeoff (0 or 1)
        flight_info[3] = 1.0 if schedule.is_takeoff else 0.0
        
        # 4: status (normalized)
        status_mapping = {
            FlightStatus.DORMANT: 0.0,
            FlightStatus.WAITING: 0.2,
            FlightStatus.TAXI_TO_RUNWAY: 0.4,
            FlightStatus.TAKE_OFF: 0.6,
            FlightStatus.LANDING: 0.7,
            FlightStatus.TAXI_TO_GATE: 0.8,
            FlightStatus.DELAYED: 0.9,
            FlightStatus.CANCELLED: 1.0
        }
        flight_info[4] = status_mapping.get(schedule.status, 0.0)
        
        # 5: runway assignment (0: none, 1: 14L, 2: 14R)
        if schedule.runway is None:
            flight_info[5] = 0.0
        elif schedule.runway.name == "14L":
            flight_info[5] = 0.5
        else:  # 14R
            flight_info[5] = 1.0
        
        return flight_info
    
    def _apply_single_schedule_action(self, schedule, action: List[int]) -> float:
        """Apply a single schedule action and return reward"""
        # action: [time_choice, runway_choice]
        time_choice, runway_choice = action
        
        # 올바른 ETD/ETA 사용
        if schedule.is_takeoff:
            original_time = schedule.flight.etd
            if original_time is None:
                return 0.0  # ETD가 없으면 실패
            
            # 시간 계산: ETD + delay (0~180분)
            actual_time = original_time + time_choice
        else:
            # 착륙: original_eta를 기준으로 15분 간격으로 제한
            original_time = schedule.original_eta
            if original_time is None:
                return 0.0  # original_eta가 없으면 실패
            
            # time_choice를 0~12 범위로 변환 (// 15) 후 다시 15분 간격으로 변환 (* 15)
            normalized_choice = time_choice // 15
            actual_time = original_time + (normalized_choice * 15)
        
        # 활주로 선택
        runway_name = "14L" if runway_choice == 0 else "14R"
        
        # 활주로 선택
        runway = next((r for r in self.sim.airport.runways if r.name == runway_name), None)
        if runway is None:
            return 0.0
        
        # 스케줄 배정 (무조건 배정)
        try:
            if schedule.is_takeoff:
                schedule.etd = actual_time
            else:
                schedule.eta = actual_time
            
            schedule.runway = runway
            return 1.0  # 성공
        except Exception as e:
            return 0.0  # 실패
    
    def get_action_mask(self, schedule) -> List[List[bool]]:
        """Get action mask for the current schedule"""
        # [time_mask, runway_mask]
        time_mask = [True] * 181  # 모든 시간 선택 가능
        runway_mask = [True, True]  # 14L, 14R 모두 가능
        
        return [time_mask, runway_mask]
