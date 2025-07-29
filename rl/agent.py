#!/usr/bin/env python3
"""
실제 시뮬레이션에서 사용하는 PPO 에이전트
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Any
from collections import deque
import random

class ActorCritic(nn.Module):
    """Actor-Critic 네트워크"""
    
    def __init__(self, observation_size: int, action_size: int):
        super(ActorCritic, self).__init__()
        
        # 공통 레이어
        self.shared = nn.Sequential(
            nn.Linear(observation_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # Actor (정책)
        self.actor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
            nn.Softmax(dim=-1)
        )
        
        # Critic (가치)
        self.critic = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, state):
        shared_features = self.shared(state)
        action_probs = self.actor(shared_features)
        value = self.critic(shared_features)
        return action_probs, value

class PPOAgent:
    """PPO 에이전트"""
    
    def __init__(self, observation_size: int, action_size: int, 
                 learning_rate: float = 0.0003, gamma: float = 0.99):
        self.observation_size = observation_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        
        # PPO 하이퍼파라미터
        self.clip_epsilon = 0.2
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.01
        
        # 네트워크
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor_critic = ActorCritic(observation_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        
        # 경험 저장
        self.states = []
        self.actions = []
        self.action_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        
        # 학습 통계
        self.training_count = 0
    
    def select_action(self, state: np.ndarray, num_schedules: int) -> Tuple[List[int], List[float], float]:
        """액션 선택"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_probs, value = self.actor_critic(state_tensor)
            action_probs = action_probs.squeeze(0)
            value = value.squeeze(0).item()
        
        # 각 스케줄에 대해 액션 선택
        actions = []
        probs = []
        
        for _ in range(num_schedules):
            # Categorical 분포에서 샘플링
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            prob = action_probs[action].item()
            
            actions.append(action.item())
            probs.append(prob)
        
        return actions, probs, value
    
    def store_transition(self, state: np.ndarray, actions: List[int], 
                        action_probs: List[float], reward: float, 
                        value: float, done: bool):
        """경험 저장 - 각 액션을 별도의 경험으로 저장"""
        for i, (action, prob) in enumerate(zip(actions, action_probs)):
            # 각 액션을 별도의 경험으로 저장
            self.states.append(state)
            self.actions.append([action])  # 단일 액션으로 변환
            self.action_probs.append([prob])  # 단일 확률로 변환
            self.rewards.append(reward)
            self.values.append(value)
            self.dones.append(done and i == len(actions) - 1)  # 마지막 액션만 done=True
    
    def update(self, batch_size: int = 32, epochs: int = 4):
        """PPO 업데이트"""
        if len(self.states) < batch_size:
            return
        
        # 경험을 텐서로 변환
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        old_action_probs = torch.FloatTensor(self.action_probs).to(self.device)
        rewards = torch.FloatTensor(self.rewards).to(self.device)
        values = torch.FloatTensor(self.values).to(self.device)
        dones = torch.FloatTensor(self.dones).to(self.device)
        
        # Advantage 계산
        advantages = self._compute_advantages(rewards, values, dones)
        
        # 여러 에포크에 걸쳐 업데이트
        for epoch in range(epochs):
            # 배치 샘플링
            indices = torch.randperm(len(states))
            
            for start_idx in range(0, len(states), batch_size):
                end_idx = min(start_idx + batch_size, len(states))
                batch_indices = indices[start_idx:end_idx]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_probs = old_action_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = rewards[batch_indices] + self.gamma * values[batch_indices]
                
                # 현재 정책으로 액션 확률 계산
                current_action_probs, current_values = self.actor_critic(batch_states)
                
                # 배치의 각 샘플에 대해 선택된 액션의 확률만 추출
                batch_size = batch_states.size(0)
                selected_probs = torch.zeros(batch_size).to(self.device)
                for i in range(batch_size):
                    action_idx = batch_actions[i][0]  # 단일 액션
                    selected_probs[i] = current_action_probs[i][action_idx]
                
                # PPO 손실 계산
                old_probs = batch_old_probs.squeeze()  # [batch_size]로 변환
                ratio = selected_probs / (old_probs + 1e-8)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(current_values.squeeze(), batch_returns)
                entropy_loss = -(current_action_probs * torch.log(current_action_probs + 1e-8)).mean()
                
                total_loss = actor_loss + self.value_loss_coef * critic_loss + self.entropy_coef * entropy_loss
                
                # 업데이트
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
        
        # 경험 초기화
        self.states = []
        self.actions = []
        self.action_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        
        self.training_count += 1
        
        if self.training_count % 10 == 0:
            print(f"PPO 업데이트: {self.training_count}회, 손실: {total_loss.item():.4f}")
    
    def _compute_advantages(self, rewards: torch.Tensor, values: torch.Tensor, 
                          dones: torch.Tensor) -> torch.Tensor:
        """Advantage 계산"""
        advantages = torch.zeros_like(rewards)
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * 0.95 * gae * (1 - dones[t])
            advantages[t] = gae
        
        return advantages
    
    def save_model(self, path: str):
        """모델 저장"""
        torch.save({
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_count': self.training_count
        }, path)
    
    def load_model(self, path: str):
        """모델 로드"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_count = checkpoint['training_count'] 