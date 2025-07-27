import json
from sim.flight import Flight
from sim.schedule import Schedule
from sim.event import Event
from utils.time_utils import hhmm_to_int
from utils.logger import debug

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
    
    # Flight 클래스에서 받을 수 있는 필드만 필터링
    flight_fields = ['flight_id', 'etd', 'eta', 'dep_airport', 'arr_airport', 'airline']
    flights = []
    for fd in data.get('flights', []):
        filtered_fd = {k: v for k, v in fd.items() if k in flight_fields}
        flights.append(Flight(**filtered_fd))
    takeoff_flights = [f for f in flights if f.dep_airport == "GMP"]
    landing_flights = [f for f in flights if f.arr_airport == "GMP"]
    schedules = [Schedule(f, is_takeoff=True) for f in takeoff_flights]
    # Priority 정보 출력
    for schedule in schedules:
        debug(f"Schedule created: {schedule.flight.flight_id} (priority {schedule.priority})")
    events = [Event(e['event_type'], e['target_type'], e['target'], e['time'], e['duration']) for e in data.get('events', [])]
    return schedules, landing_flights, events 