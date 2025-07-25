class Event:
    def __init__(self, event_type, target_type, target, time, duration):
        self.event_type = event_type
        self.target_type = target_type
        self.target = target
        self.time = time
        self.duration = duration 

'''
# "event_type", "targetType", "target", "duration"
#1. EMERGENCY_LANDING
"EMERGENCY_LANDING", "", "LJ517", 10 # target=flight_name , duration 안에 비상 착륙
#2. RUNWAY_CLOSURE
"RUNWAY_CLOSURE", "", "14L or 14R", 10 # duration 동안 해당 RWY CLOSED
#3. FLIGHT_CANCEL
"FLIGHT_CANCEL", "", "LJ517", 0 # 해당 비행 CNL'
#4. FLIGHT_DELAY
"FLIGHT_DELAY", "", "LJ517", 15 # 특정 TAKE-OFF SCHEDULE duration만큼 DELAY
#5. GO_AROUND
"GO_AROUND", "", "LJ517", 0 # 특정 LANDING SCHEDULE 15분 DELAY
#6. TAKEOFF_CRASH
"TAKEOFF_CRASH", "", "LJ517", 60 # 특정 비행 이륙 문제 발생. DURATION동안 이륙 활주로 폐쇄.
#7. LANDING_CRASH
"LANDING_CRASH", "", "LJ517", 60 # 특정 비행 착륙 문제 발생. DURATION동안 착륙 활주로 폐쇄. 
#8. RUNWAY_INVERT
"RUNWAY_INVERT", "", "", 0
# TAKEOFF(or LANDING)_RUNWAY: 14L->32R OR 32R->14L OR 14R->32L OR 32L->14R
# TAKEOFF_TAXIWAY: G2->B2 OR B2->G2
'''