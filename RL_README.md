# 강화학습 (PPO) 모듈 사용법

## 개요

이 프로젝트는 공항 스케줄링 문제를 해결하기 위해 Proximal Policy Optimization (PPO) 알고리즘을 구현했습니다.

## 구조

```
backend/
├── rl/
│   ├── __init__.py
│   ├── environment.py      # 강화학습 환경
│   ├── agent.py           # PPO 에이전트
│   └── trainer.py         # 학습 관리자
├── models/                # 학습된 모델 저장
├── train_rl.py           # 독립 학습 스크립트
└── main.py               # 메인 실행 파일
```

## 설치

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. PyTorch 설치 (CUDA 지원)

```bash
# CPU 버전
pip install torch torchvision torchaudio

# CUDA 버전 (GPU 사용 시)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 사용법

### 1. 모델 학습

#### 방법 1: 독립 스크립트 사용
```bash
cd backend
python train_rl.py
```

#### 방법 2: main.py에서 학습
```python
# main.py에서 설정
train_rl = True  # 강화학습 모델 학습
use_rl = False   # 학습 중에는 사용하지 않음
```

### 2. 학습된 모델 사용

```python
# main.py에서 설정
train_rl = False  # 학습하지 않음
use_rl = True     # 강화학습 모델 사용
```

### 3. 스케줄러에서 직접 사용

```python
from sim.scheduler import Scheduler

scheduler = Scheduler("rl", sim)  # RL 알고리즘 사용
```

## 환경 (Environment)

### 상태 공간 (State Space)
- 스케줄 정보 (우선순위, ETD, ETA, 이륙/착륙 여부)
- 활주로 상태 (점유 여부, 다음 사용 가능 시간)
- 현재 시간

### 액션 공간 (Action Space)
- 0: 14L 활주로에 이륙 배정
- 1: 14R 활주로에 착륙 배정
- 2: 대기 (아무것도 하지 않음)

### 보상 함수 (Reward Function)
```
보상 = -(지연손실 + 안전손실) + 완료비행수 * 10
```

## 하이퍼파라미터

### PPO 에이전트
- `learning_rate`: 3e-4
- `gamma`: 0.99 (할인율)
- `epsilon`: 0.2 (클립 범위)
- `epochs`: 10 (업데이트 에포크)

### 학습 설정
- `max_episodes`: 1000
- `max_steps_per_episode`: 500
- `update_frequency`: 50

## 모델 저장/로드

### 모델 저장
```python
agent.save_model("models/ppo_airport.pth")
```

### 모델 로드
```python
agent.load_model("models/ppo_airport.pth")
```

## 성능 평가

학습 완료 후 자동으로 평가가 실행됩니다:

- 평균 보상
- 평균 에피소드 길이
- 평균 완료 비행 수
- 평균 지연 손실
- 평균 안전 손실

## 문제 해결

### 1. CUDA 메모리 부족
```python
# CPU 사용으로 변경
device = torch.device("cpu")
```

### 2. 학습이 수렴하지 않음
- 하이퍼파라미터 조정
- 보상 함수 수정
- 액션 공간 재설계

### 3. 모듈 import 오류
```bash
# PYTHONPATH 설정
export PYTHONPATH="${PYTHONPATH}:/path/to/backend"
```

## 향후 개선 사항

1. **더 복잡한 액션 공간**: 연속적인 시간 배정
2. **멀티 에이전트**: 각 활주로별 독립 에이전트
3. **메타러닝**: 다양한 시나리오에 대한 빠른 적응
4. **설명 가능성**: 의사결정 과정 시각화

## 참고 자료

- [PPO 논문](https://arxiv.org/abs/1707.06347)
- [PyTorch 공식 문서](https://pytorch.org/docs/stable/index.html)
- [OpenAI Gym 환경 설계](https://www.gymlibrary.dev/) 