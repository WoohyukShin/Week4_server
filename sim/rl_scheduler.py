"""
실제 시뮬레이션에서 사용하는 RL 스케줄러
"""

import torch
import numpy as np
from typing import List, Dict, Any, Tuple
from sim.schedule import Schedule
from sim.flight import FlightStatus
from utils.logger import debug
from rl.agent import PPOAgent
from rl.environment import AirportEnvironment

class RLScheduler:
    """실제 시뮬레이션에서 사용하는 RL 스케줄러"""
    
    def __init__(self, sim, model_path: str = None):
        self.sim = sim
        self.agent = None
        self.training_mode = False
        
        # RL 환경 (상태 관찰용)
        self.env = AirportEnvironment(sim)
        
        # 에이전트 초기화
        self._init_agent(model_path)
        
        # 경험 저장용
        self.current_episode_experiences = []
        self.episode_count = 0
        
    def _init_agent(self, model_path: str = None):
        """RL 에이전트 초기화"""
        observation_size = self.env._get_observation_space_size()
        action_space = self.env.action_space
        
        self.agent = PPOAgent(
            observation_size=observation_size,
            action_space=action_space
        )
        
        # 기존 모델 로드
        if model_path:
            try:
                self.agent.load_model(model_path)
                debug(f"RL 모델 로드 완료: {model_path}")
            except Exception as e:
                debug(f"모델 로드 실패: {e}")
    
    def do_action(self) -> bool:
        """실제 시뮬레이션에서 호출되는 스케줄링 액션"""
        # 사용 가능한 스케줄 확인
        available_schedules = [s for s in self.sim.schedules 
                              if s.status in [FlightStatus.DORMANT, FlightStatus.WAITING]]
        
        if not available_schedules:
            debug("RL Scheduler: 배정 가능한 스케줄이 없습니다.")
            return False
        
        # do_action() 시작 시 assigned 정보 초기화
        self._reset_assigned_info()
        
        # 각 스케줄별로 개별 observation 생성
        schedule_observations = []
        for schedule in available_schedules:
            observation = self.env._get_observation_for_schedule(schedule)
            schedule_observations.append(observation)
        
        # RL 에이전트가 액션 선택
        actions, action_probs, value = self.agent.select_action(schedule_observations)
        
        # 개별 액션별로 즉시 보상 계산 및 적용
        total_reward = 0.0
        assigned_count = 0
        
        for i, (schedule, action) in enumerate(zip(available_schedules, actions)):
            # 액션 적용
            reward = self._apply_single_schedule_action(schedule, action)
            
            # 개별 즉시 보상 계산 (이미 배정된 비행들과의 충돌 체크)
            immediate_reward = self._calculate_immediate_reward(schedule)
            total_action_reward = reward + immediate_reward
            
            total_reward += total_action_reward
            
            if reward >= 0:
                assigned_count += 1
                debug(f"RL Scheduler: {schedule.flight.flight_id} 배정 성공 (reward: {reward}, immediate: {immediate_reward})")
            else:
                debug(f"RL Scheduler: {schedule.flight.flight_id} 배정 실패 (reward: {reward})")
        
        # 경험 저장 (훈련 모드일 때만)
        if self.training_mode:
            self.current_episode_experiences.append({
                'state': schedule_observations,
                'actions': actions,
                'action_probs': action_probs,
                'reward': total_reward,
                'value': value,
                'done': False
            })
        
        debug(f"RL Scheduler: {assigned_count}개 스케줄 배정, 총 보상: {total_reward}")
        return assigned_count > 0
    
    def _apply_single_schedule_action(self, schedule: Schedule, action: int) -> float:
        """단일 스케줄에 액션 적용"""
        if not schedule or schedule.status not in [FlightStatus.DORMANT, FlightStatus.WAITING]:
            return 0.0
        
        # action을 활주로 선택 + 시간 선택으로 해석
        runway_choice = action % 2  # 0: 14L, 1: 14R
        time_choice = action // 2   # 0~143: ETD 기준 시간 오프셋
        
        runway_name = "14L" if runway_choice == 0 else "14R"
        
        # 비행의 원래 ETD/ETA를 기준으로 시간 선택
        if schedule.is_takeoff:
            original_etd = schedule.flight.etd
            if original_etd is None:
                return -5.0  # ETD 정보 없음
            
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
            
            # ETD 기준으로 -2~+142분 범위에서 선택 (총 144분)
            selected_time = etd_minutes - 2 + time_choice
            
        else:
            # 착륙: ETA 기준으로 시간 선택
            original_eta = schedule.flight.eta
            if original_eta is None:
                return -5.0  # ETA 정보 없음
            
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
            
            # ETA 기준으로 -2~+142분 범위에서 선택 (총 144분)
            selected_time = eta_minutes - 2 + time_choice
        
        # 시간 제약 조건 검사
        if selected_time < 0 or selected_time >= 1440:
            return -5.0  # 시간 제약 위반
        
        # 즉각적인 충돌 검사
        if self._check_conflict(schedule, runway_name, selected_time):
            return -10.0  # 즉시 충돌
        
        # 활주로 제약 조건 검사
        if not self._check_runway_constraints(schedule, runway_name):
            return -5.0  # 활주로 제약 위반
        
        # 배정 성공
        success = self._assign_schedule(schedule, runway_name, selected_time)
        if success:
            # assigned 정보 설정 (즉시 보상 계산용)
            schedule.assigned_time = selected_time
            schedule.assigned_runway_id = runway_name
            return 0.0  # 성공해도 즉시 보상 없음
        else:
            return -5.0   # 배정 실패
    
    def _check_conflict(self, schedule: Schedule, runway_name: str, selected_time: int) -> bool:
        """즉각적인 충돌 검사"""
        # 활주로 사용 시간 충돌 검사
        if runway_name in self.env.runway_usage:
            last_usage = self.env.runway_usage[runway_name]
            if abs(selected_time - last_usage) < 4:  # 4분 분리 규칙
                return True
        
        # 다른 스케줄과의 시간 충돌 검사
        for other_schedule in self.sim.schedules:
            if other_schedule == schedule:
                continue
            if other_schedule.etd == selected_time or other_schedule.eta == selected_time:
                return True
        
        return False
    
    def _check_runway_constraints(self, schedule: Schedule, runway_name: str) -> bool:
        """활주로 제약 조건 검사"""
        if schedule.is_takeoff:
            # 이륙: 14R은 14L이 closed일 때만 사용 가능
            if runway_name == "14R" and not self._is_14l_closed():
                return False
        else:
            # 착륙: 14L은 14R이 closed일 때만 사용 가능
            if runway_name == "14L" and not self._is_14r_closed():
                return False
        
        return True
    
    def _assign_schedule(self, schedule: Schedule, runway_name: str, selected_time: int) -> bool:
        """스케줄 배정"""
        try:
            # 스케줄 배정
            if schedule.is_takeoff:
                schedule.etd = selected_time
            else:
                schedule.eta = selected_time
            
            schedule.runway = runway_name
            
            # 활주로 사용 기록 업데이트
            self.env.runway_usage[runway_name] = selected_time
            
            return True
        except Exception as e:
            debug(f"스케줄 배정 실패: {e}")
            return False
    
    def _is_14l_closed(self) -> bool:
        """14L 활주로 폐쇄 여부 확인"""
        for runway in self.sim.airport.runways:
            if runway.name in ["14L", "32R"]:
                return runway.closed
        return False
    
    def _is_14r_closed(self) -> bool:
        """14R 활주로 폐쇄 여부 확인"""
        for runway in self.sim.airport.runways:
            if runway.name in ["14R", "32L"]:
                return runway.closed
        return False
    
    def end_episode(self, final_reward: float):
        """에피소드 종료 처리"""
        if not self.training_mode or not self.current_episode_experiences:
            return
        
        # 마지막 경험에 최종 보상 추가
        if self.current_episode_experiences:
            self.current_episode_experiences[-1]['done'] = True
            self.current_episode_experiences[-1]['reward'] += final_reward
        
        # 경험을 에이전트에 저장
        for exp in self.current_episode_experiences:
            self.agent.store_transition(
                exp['state'], exp['actions'], exp['action_probs'],
                exp['reward'], exp['value'], exp['done']
            )
        
        # 에이전트 업데이트
        self.agent.update(batch_size=32, epochs=4)
        
        # 경험 초기화
        self.current_episode_experiences = []
        self.episode_count += 1
        
        debug(f"RL 에피소드 {self.episode_count} 완료, 최종 보상: {final_reward}")
    
    def save_model(self, path: str):
        """모델 저장"""
        if self.agent:
            self.agent.save_model(path)
            debug(f"RL 모델 저장: {path}")
    
    def load_model(self, path: str):
        """모델 로드"""
        if self.agent:
            self.agent.load_model(path)
            debug(f"RL 모델 로드: {path}")
    
    def set_training_mode(self, training: bool):
        """훈련 모드 설정"""
        self.training_mode = training
        debug(f"RL 훈련 모드: {training}")
    
    def _reset_assigned_info(self):
        """do_action() 시작 시 assigned 정보 초기화"""
        for schedule in self.sim.schedules:
            if schedule.status in [FlightStatus.DORMANT, FlightStatus.WAITING]:
                # 이번 do_action()에서 배정할 비행들은 초기화
                schedule.assigned_time = None
                schedule.assigned_runway_id = None
            else:
                # 이미 배정된 비행들은 현재 배정 정보 유지
                if schedule.etd is not None:
                    schedule.assigned_time = schedule.etd
                elif schedule.eta is not None:
                    schedule.assigned_time = schedule.eta
                
                if schedule.runway is not None:
                    schedule.assigned_runway_id = schedule.runway.name
    
    def _calculate_immediate_reward(self, schedule: Schedule) -> float:
        """개별 스케줄 배정에 대한 즉시 보상 계산"""
        if not hasattr(schedule, 'assigned_time') or schedule.assigned_time is None:
            return 0.0
        
        immediate_reward = 0.0
        
        # 1. DELAY LOSS (지연 시간에 대한 보상)
        if schedule.is_takeoff:
            original_time = schedule.flight.etd
            if original_time is not None:
                delay = schedule.assigned_time - original_time
                if delay > 0:
                    # 지연 시간에 대한 보상 (priority 고려)
                    normalized_priority = schedule.priority / 32.0
                    immediate_reward -= delay * normalized_priority * 100
        else:
            original_time = schedule.original_eta
            if original_time is not None:
                delay = schedule.assigned_time - original_time
                if delay > 0:
                    # 지연 시간에 대한 보상 (priority 고려)
                    normalized_priority = schedule.priority / 32.0
                    immediate_reward -= delay * normalized_priority * 100
        
        # 2. RUNWAY OCCUPY LOSS
        for other_schedule in self.sim.schedules:
            if other_schedule == schedule:
                continue
            
            if (hasattr(other_schedule, 'assigned_time') and 
                other_schedule.assigned_time is not None and
                other_schedule.assigned_runway_id == schedule.assigned_runway_id):
                
                time_diff = abs(schedule.assigned_time - other_schedule.assigned_time)
                if time_diff == 0:
                    immediate_reward -= 30000 # 같은 활주로에서 동시 이착륙 => 최악의 경우
                elif time_diff < 4:
                    immediate_reward -= 3000 # Runway Occupy Loss
        
        # 3. SIMULTANEOUS OPS
        runway = next((r for r in self.sim.airport.runways if r.name == schedule.assigned_runway_id), None)
        if runway and schedule.assigned_time < runway.next_available_time:
            immediate_reward -= 5000 # CLOSED or occupied LOSS
        
        return immediate_reward