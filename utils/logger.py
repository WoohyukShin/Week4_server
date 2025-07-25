import inspect
import os

def debug(message):
    frame = inspect.currentframe().f_back
    filename = os.path.basename(frame.f_code.co_filename)
    funcname = frame.f_code.co_name
    print(f"[DEBUG] {filename}/{funcname} : {message}") 