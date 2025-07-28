import json
import random
import string
from sim.flight import Flight
from sim.schedule import Schedule, PRI_TAKEOFF_MIN, PRI_TAKEOFF_MAX, PRI_LANDING_MIN, PRI_LANDING_MAX
from sim.event import Event
from utils.time_utils import hhmm_to_int, int_to_hhmm
from utils.logger import debug

def generate_random_scenario(num_flights=50, num_events=5):
    """
    강화학습을 위한 랜덤 시나리오 생성
    
    Args:
        num_flights: 생성할 비행기 수 (기본값: 50)
        num_events: 생성할 이벤트 수 (기본값: 5)
    
    Returns:
        dict: JSON 형태의 시나리오 데이터
    """
    scenario = {
        "flights": [],
        "events": []
    }
    
    # 비행기 생성
    flight_ids = set()
    airlines = ["KE", "OZ", "7C", "TW", "ZE", "BX", "LJ", "VJ", "RS", "TZ"]
    
    for i in range(num_flights):
        # 고유한 flight_id 생성
        while True:
            airline = random.choice(airlines)
            flight_num = random.randint(100, 999)
            flight_id = f"{airline}{flight_num}"
            if flight_id not in flight_ids:
                flight_ids.add(flight_id)
                break
        
        # 출발/도착 공항 설정 (GMP는 반드시 포함)
        if random.random() < 0.6:  # 60% 확률로 GMP 출발
            dep_airport = "GMP"
            arr_airport = ''.join(random.choices(string.ascii_uppercase, k=3))
        else:  # 40% 확률로 GMP 도착
            dep_airport = ''.join(random.choices(string.ascii_uppercase, k=3))
            arr_airport = "GMP"
        
        # 시간 설정 (0600 ~ 0900, 즉 360 ~ 540)
        if dep_airport == "GMP":  # 이륙 비행
            etd = random.randint(360, 540)
            eta = None
        else:  # 착륙 비행
            etd = None
            eta = random.randint(360, 540)
        
        # Priority 설정 (상수 사용)
        if arr_airport == "GMP":  # 착륙 비행
            priority = random.randint(PRI_LANDING_MIN, PRI_LANDING_MAX)
        else:  # 이륙 비행
            priority = random.randint(PRI_TAKEOFF_MIN, PRI_TAKEOFF_MAX)
        
        flight = {
            "flight_id": flight_id,
            "etd": int_to_hhmm(etd) if etd else None,
            "eta": int_to_hhmm(eta) if eta else None,
            "dep_airport": dep_airport,
            "arr_airport": arr_airport,
            "airline": airline,
            "priority": priority
        }
        scenario["flights"].append(flight)
    
    # 이벤트 생성
    event_types = [
        "EMERGENCY_LANDING",
        "RUNWAY_CLOSURE", 
        "FLIGHT_CANCEL",
        "FLIGHT_DELAY",
        "GO_AROUND",
        "TAKEOFF_CRASH",
        "LANDING_CRASH",
        "RUNWAY_INVERT"
    ]
    
    # 이벤트별 가중치 (자주 발생하는 이벤트에 높은 가중치)
    event_weights = [0.1, 0.15, 0.1, 0.27, 0.15, 0.01, 0.02, 0.2]
    
    for i in range(num_events):
        event_type = random.choices(event_types, weights=event_weights)[0]
        event_time = random.randint(360, 540)  # 0600 ~ 0900
        
        # FLIGHT_DELAY는 이륙 항공편에만 적용
        if event_type == "FLIGHT_DELAY":
            # 이륙 항공편 중에서 선택
            takeoff_flights = [f for f in scenario["flights"] if f["dep_airport"] == "GMP"]
            if takeoff_flights:
                target_flight = random.choice(takeoff_flights)
                event_data = {
                    "type": event_type,
                    "time": event_time,
                    "flight_id": target_flight["flight_id"],
                    "duration": random.randint(10, 30)  # 10~30분 지연
                }
            else:
                # 이륙 항공편이 없으면 다른 이벤트로 변경
                event_type = random.choice([e for e in event_types if e != "FLIGHT_DELAY"])
                event_data = {
                    "type": event_type,
                    "time": event_time
                }
        else:
            event_data = {
                "type": event_type,
                "time": event_time
            }
        
        # 이벤트 타입별 설정
        if event_type in ["EMERGENCY_LANDING", "FLIGHT_CANCEL", "FLIGHT_DELAY", "GO_AROUND", "TAKEOFF_CRASH", "LANDING_CRASH"]:
            # 비행기 관련 이벤트
            target_flight = random.choice([f for f in scenario["flights"] if f["flight_id"] in flight_ids])
            target = target_flight["flight_id"]
            target_type = "flight"
            
            if event_type in ["EMERGENCY_LANDING", "TAKEOFF_CRASH", "LANDING_CRASH"]:
                duration = random.randint(10, 60)
            elif event_type == "FLIGHT_DELAY":
                duration = random.randint(5, 30)
            else:
                duration = 0
                
        elif event_type == "RUNWAY_CLOSURE":
            # 활주로 폐쇄 이벤트
            runways = ["14L", "14R", "32L", "32R"]
            target = random.choice(runways)
            target_type = "runway"
            duration = random.randint(10, 30)
            
        elif event_type == "RUNWAY_INVERT":
            # 활주로 방향 전환 이벤트
            target = ""
            target_type = ""
            duration = 0
        
        event = {
            "event_type": event_type,
            "target_type": target_type,
            "target": target,
            "time": int_to_hhmm(event_time),
            "duration": duration
        }
        scenario["events"].append(event)
    
    # 로그 출력
    debug("======= RANDOM SCENARIO GENERATED ========")
    debug("FLIGHTS:")
    
    # 비행기 정보를 시간 순으로 정렬
    sorted_flights = sorted(scenario["flights"], key=lambda x: (x["etd"] or x["eta"] or 0))
    
    for flight in sorted_flights:
        if flight["dep_airport"] == "GMP":  # 이륙
            debug(f"DEP {flight['flight_id']} {flight['etd']}")
        else:  # 착륙
            debug(f"ARR {flight['flight_id']} {flight['eta']}")
    
    debug("")
    debug("EVENTS:")
    # 이벤트 정보를 시간 순으로 정렬
    sorted_events = sorted(scenario["events"], key=lambda x: x["time"])
    
    for event in sorted_events:
        debug(f"{event['event_type']} {event['time']} {event['target']} {event['duration']}")
    
    debug("==========================================")
    
    return scenario

def save_random_scenario(scenario, filename="random_scenario.json"):
    """랜덤 시나리오를 JSON 파일로 저장"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(scenario, f, indent=2, ensure_ascii=False)
    debug(f"랜덤 시나리오가 {filename}에 저장되었습니다.")

def load_scenario_from_dict(data):
    """
    메모리에서 직접 시나리오 딕셔너리를 로드
    """
    # flight, event 시간을 Timestep으로 변환
    for flight_data in data.get('flights', []):
        if 'etd' in flight_data and flight_data['etd'] is not None:
            original_etd = flight_data['etd']
            flight_data['etd'] = hhmm_to_int(flight_data['etd'])
        if 'eta' in flight_data and flight_data['eta'] is not None:
            original_eta = flight_data['eta']
            flight_data['eta'] = hhmm_to_int(flight_data['eta'])
    for event_data in data.get('events', []):
        if 'time' in event_data and event_data['time'] is not None:
            original_time = event_data['time']
            event_data['time'] = hhmm_to_int(event_data['time'])
    
    # Flight 클래스에서 받을 수 있는 필드만 필터링 (priority 포함)
    flight_fields = ['flight_id', 'etd', 'eta', 'dep_airport', 'arr_airport', 'airline', 'priority']
    flights = []
    for fd in data.get('flights', []):
        filtered_fd = {k: v for k, v in fd.items() if k in flight_fields}
        flights.append(Flight(**filtered_fd))
    takeoff_flights = [f for f in flights if f.dep_airport == "GMP"]
    landing_flights = [f for f in flights if f.arr_airport == "GMP"]
    
    # Priority가 설정되지 않은 경우 기본값 할당
    schedules = []
    for f in takeoff_flights:
        schedule = Schedule(f, is_takeoff=True)
        schedules.append(schedule)
    events = [Event(e['event_type'], e['target_type'], e['target'], e['time'], e['duration']) for e in data.get('events', [])]
    return schedules, landing_flights, events

def load_scenario(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # flight, event 시간을 Timestep으로 변환
    for flight_data in data.get('flights', []):
        if 'etd' in flight_data and flight_data['etd'] is not None:
            original_etd = flight_data['etd']
            flight_data['etd'] = hhmm_to_int(flight_data['etd'])
        if 'eta' in flight_data and flight_data['eta'] is not None:
            original_eta = flight_data['eta']
            flight_data['eta'] = hhmm_to_int(flight_data['eta'])
    for event_data in data.get('events', []):
        if 'time' in event_data and event_data['time'] is not None:
            original_time = event_data['time']
            event_data['time'] = hhmm_to_int(event_data['time'])
            debug(f"이벤트 시간 변환: {original_time} -> {event_data['time']}")
    
    # Flight 클래스에서 받을 수 있는 필드만 필터링 (priority 포함)
    flight_fields = ['flight_id', 'etd', 'eta', 'dep_airport', 'arr_airport', 'airline', 'priority']
    flights = []
    for fd in data.get('flights', []):
        filtered_fd = {k: v for k, v in fd.items() if k in flight_fields}
        flights.append(Flight(**filtered_fd))
    takeoff_flights = [f for f in flights if f.dep_airport == "GMP"]
    landing_flights = [f for f in flights if f.arr_airport == "GMP"]
    
    # Priority가 설정되지 않은 경우 기본값 할당
    schedules = []
    for f in takeoff_flights:
        schedule = Schedule(f, is_takeoff=True)
        schedules.append(schedule)
        debug(f"Schedule created: {schedule.flight.flight_id} (priority {schedule.priority})")
    events = [Event(e['event_type'], e['target_type'], e['target'], e['time'], e['duration']) for e in data.get('events', [])]
    return schedules, landing_flights, events 