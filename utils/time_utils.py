def int_to_hhmm(t):
    ''' 104 -> 144 '''
    hour = (t % 1440) // 60
    minute = (t % 1440) % 60
    return hour * 100 + minute 

def int_to_hhmm_str(t):
    ''' 104 -> "0144"'''
    hour = (t % 1440) // 60
    minute = (t % 1440) % 60
    return f"{hour:02d}{minute:02d}"

def int_to_hhmm_colon(t):
    ''' 104 -> "01:44 '''
    hour = (t % 1440) // 60
    minute = (t % 1440) % 60
    return f"{hour:02d}:{minute:02d}"

def hhmm_to_int(hhmm):
    ''' 0144 -> 104 '''
    if isinstance(hhmm, str):
        if ':' in hhmm:
            hour, minute = map(int, hhmm.split(':'))
        else:
            hour = int(hhmm) // 100
            minute = int(hhmm) % 100
    else:
        hour = hhmm // 100
        minute = hhmm % 100
    return hour * 60 + minute 