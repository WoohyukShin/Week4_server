#!/usr/bin/env python3
"""
실제 시뮬레이션을 사용한 RL 훈련 스크립트
"""

import os
import sys
import time
import numpy as np
import torch
from typing import List, Dict, Any

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import create_rkss_airport
from utils.scenario_loader import generate_random_scenario, load_scenario_from_dict
from sim.simulation import Simulation
from rl.agent import PPOAgent
from utils.logger import debug, set_training_mode

# matplotlib 한글 폰트 설정
try:
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm

    plt.rcParams['font.family'] = ['DejaVu Sans', 'Malgun Gothic', 'NanumGothic', 'AppleGothic']
    plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
    
    try:
        fm._rebuild()
    except AttributeError:
        pass
except ImportError:
    pass  # matplotlib이 설치되지 않은 경우 무시

def train_rl_with_real_simulation(episodes: int = 50, model_path: str = None):
    """실제 시뮬레이션을 사용한 RL 훈련"""
    print("=== 실제 시뮬레이션 기반 RL 훈련 시작 ===")
    
    # 훈련 모드 활성화
    set_training_mode(True)
    
    # RL 에이전트 초기화
    # 상태 크기 계산: 1(시간) + 2*3(활주로) + 24*2(날씨) + 20*5(스케줄) + 1(이벤트) + 4(통계) = 1 + 6 + 48 + 100 + 1 + 4 = 160
    observation_size = 160
    action_size = 288  # 2개 활주로 × 144개 시간 선택
    rl_agent = PPOAgent(observation_size=observation_size, action_size=action_size)
    
    # 기존 모델 로드 (있는 경우)
    models_dir = "models"
    model_path = None
    
    if os.path.exists(models_dir):
        model_files = [f for f in os.listdir(models_dir) if f.startswith("ppo_best_") and f.endswith(".pth")]
        if model_files:
            latest_model = sorted(model_files)[-1]
            model_path = os.path.join(models_dir, latest_model)
            rl_agent.load_model(model_path)
            print(f"기존 PPO 모델 로드: {model_path}")
    
    # 최고 성능 추적
    best_reward = float('-inf')
    best_model_path = None
    
    # Early stopping 설정
    patience = 30
    no_improvement_count = 0
    
    # 훈련 통계
    training_history = []
    
    for episode in range(episodes):
        print(f"\n--- 에피소드 {episode + 1}/{episodes} ---")
        
        # 새로운 시나리오 생성
        airport = create_rkss_airport()
        scenario_dict = generate_random_scenario(num_flights=25, num_events=5)
        schedules, landing_flights, events = load_scenario_from_dict(scenario_dict)
        
        # 시뮬레이션 생성
        sim = Simulation(
            airport=airport,
            schedules=schedules,
            landing_flights=landing_flights,
            events=events,
            mode="TRAINING"
        )
        
        # RL 에이전트 설정
        sim.set_rl_agent(rl_agent)
        sim.set_training_mode(True)
        sim.scheduler.algorithm = "rl"
        
        # 시뮬레이션 시작
        start_time = time.time()
        sim.start()
        
        # 시뮬레이션 완료까지 대기
        while sim.running:
            time.sleep(0.01)  # 10ms 대기
        
        # 시뮬레이션 완료
        end_time = time.time()
        simulation_time = end_time - start_time
        
        # 결과 계산
        final_reward = -sim.get_total_loss()  # 손실을 음수 보상으로 변환
        
        # 총 비행 수 계산 (완료 + 남은 + 취소)
        total_flights = len(sim.completed_schedules) + len(sim.schedules) + sim.cancelled_flights
        completion_rate = len(sim.completed_schedules) / total_flights if total_flights > 0 else 0
        
        # 통계 정보
        stats = sim.calculate_statistics()
        print("====================")
        print(f"  시뮬레이션 시간: {simulation_time:.2f}초")
        print(f"  최종 보상: {final_reward:.1f}")
        print(f"  완료율: {completion_rate:.1%}")
        print(f"  총 손실: {sim.get_total_loss():.1f}")
        print(f"  완료된 비행: {len(sim.completed_schedules)}")
        print(f"  남은 비행: {len(sim.schedules)}")
        print(f"  취소된 비행: {sim.cancelled_flights}")
        print(f"  총 비행: {total_flights}")
        
        # 기존 디버깅 형식으로 Loss 출력
        print("====================")
        print(f"TOTAL DELAY LOSS: {sim.total_delay_loss:.1f}")
        print(f"TOTAL SAFETY LOSS: {sim.total_safety_loss:.1f}")
        print(f"TOTAL SIMULTANEOUS OPS LOSS: {sim.total_simultaneous_ops_loss:.1f}")
        print(f"TOTAL RUNWAY OCCUPIED LOSS: {sim.total_runway_occupied_loss:.1f}")
        print("====================")
        
        # 훈련 히스토리 저장
        training_history.append({
            'episode': episode + 1,
            'reward': final_reward,
            'completion_rate': completion_rate,
            'total_loss': sim.get_total_loss(),
            'completed_flights': len(sim.completed_schedules),
            'remaining_flights': len(sim.schedules),
            'total_flights': total_flights,
            'cancelled_flights': sim.cancelled_flights,
            'delay_loss': sim.total_delay_loss,
            'safety_loss': sim.total_safety_loss,
            'simulation_time': simulation_time
        })
        
        # 최고 성능 모델 저장
        if final_reward > best_reward:
            best_reward = final_reward
            no_improvement_count = 0  # 개선됨
            timestamp = int(time.time())
            best_model_path = f"models/ppo_best_{timestamp}.pth"
            
            # 모델 저장
            os.makedirs("models", exist_ok=True)
            rl_agent.save_model(best_model_path)
            print(f"  🎉 새로운 최고 보상: {best_reward:.1f}")
            print(f"  모델 저장: {best_model_path}")
        else:
            no_improvement_count += 1
        
        # Early stopping 체크
        if no_improvement_count >= patience:
            print(f"\n🛑 Early stopping: {patience} 에피소드 동안 개선 없음")
            break
        
        # 진행 상황 출력
        if (episode + 1) % 10 == 0:
            recent_rewards = [h['reward'] for h in training_history[-10:]]
            avg_reward = np.mean(recent_rewards)
            print(f"\n📊 최근 10 에피소드 평균 보상: {avg_reward:.1f}")
    
    # 훈련 완료
    print(f"\n=== 훈련 완료 ===")
    print(f"총 에피소드: {episodes}")
    print(f"최고 보상: {best_reward:.1f}")
    print(f"최고 모델: {best_model_path}")
    
    # 전체 통계
    all_rewards = [h['reward'] for h in training_history]
    avg_reward = np.mean(all_rewards)
    std_reward = np.std(all_rewards)
    
    print(f"평균 보상: {avg_reward:.1f} ± {std_reward:.1f}")
    print(f"최저 보상: {min(all_rewards):.1f}")
    print(f"최고 보상: {max(all_rewards):.1f}")
    
    # 훈련 결과 그래프 생성
    if training_history:
        plot_training_results(training_history)
    
    return best_model_path, training_history

def plot_training_results(training_history):
    """훈련 결과 시각화"""
    try:
        import matplotlib.pyplot as plt
        
        episodes = [h['episode'] for h in training_history]
        rewards = [h['reward'] for h in training_history]
        total_losses = [h['total_loss'] for h in training_history]
        delay_losses = [h['delay_loss'] for h in training_history]
        safety_losses = [h['safety_loss'] for h in training_history]
        
        # 서브플롯 생성
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('RL 훈련 결과', fontsize=16)
        
        # 보상 그래프
        ax1.plot(episodes, rewards, 'b-', linewidth=2)
        ax1.set_title('에피소드별 보상')
        ax1.set_xlabel('에피소드')
        ax1.set_ylabel('보상')
        ax1.grid(True, alpha=0.3)
        
        # 총 손실 그래프
        ax2.plot(episodes, total_losses, 'r-', linewidth=2)
        ax2.set_title('에피소드별 총 손실')
        ax2.set_xlabel('에피소드')
        ax2.set_ylabel('총 손실')
        ax2.grid(True, alpha=0.3)
        
        # 지연 손실 그래프
        ax3.plot(episodes, delay_losses, 'g-', linewidth=2)
        ax3.set_title('에피소드별 지연 손실')
        ax3.set_xlabel('에피소드')
        ax3.set_ylabel('지연 손실')
        ax3.grid(True, alpha=0.3)
        
        # 안전 손실 그래프
        ax4.plot(episodes, safety_losses, 'orange', linewidth=2)
        ax4.set_title('에피소드별 안전 손실')
        ax4.set_xlabel('에피소드')
        ax4.set_ylabel('안전 손실')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 그래프 저장
        os.makedirs("plots", exist_ok=True)
        timestamp = int(time.time())
        plot_path = f"plots/training_results_{timestamp}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"훈련 결과 그래프 저장: {plot_path}")
        
        plt.show()
        
    except ImportError:
        print("matplotlib이 설치되지 않아 그래프를 생성할 수 없습니다.")
    except Exception as e:
        print(f"그래프 생성 중 오류 발생: {e}")

def main():
    """메인 함수"""
    # 훈련 설정
    episodes = 150  # 훈련 에피소드 수
    
    # 기존 모델 경로 (있는 경우)
    models_dir = "models"
    model_path = None
    
    if os.path.exists(models_dir):
        model_files = [f for f in os.listdir(models_dir) if f.startswith("ppo_best_") and f.endswith(".pth")]
        if model_files:
            latest_model = sorted(model_files)[-1]
            model_path = os.path.join(models_dir, latest_model)
            print(f"기존 모델 로드: {model_path}")
    
    # 훈련 실행
    best_model_path, training_history = train_rl_with_real_simulation(
        episodes=episodes,
        model_path=model_path
    )
    
    print(f"\n훈련 완료! 최고 모델: {best_model_path}")

if __name__ == "__main__":
    main() 