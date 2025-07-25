def int_to_hhmm(t):
    hour = (t % 1440) // 60
    minute = (t % 1440) % 60
    return hour * 100 + minute 