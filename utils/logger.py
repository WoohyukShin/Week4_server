import inspect
import os

# Training 모드 전역 변수
TRAINING_MODE = False

def set_training_mode(enabled: bool):
    """Training 모드 설정"""
    global TRAINING_MODE
    TRAINING_MODE = enabled

def debug(message):
    # Training 모드일 때는 debug X
    if not TRAINING_MODE:
        frame = inspect.currentframe().f_back
        filename = os.path.basename(frame.f_code.co_filename)
        funcname = frame.f_code.co_name
        print(f"[DEBUG] {filename}/{funcname} : {message}") 