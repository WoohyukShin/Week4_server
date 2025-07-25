import json
from sim.flight import Flight
from sim.schedule import Schedule
from sim.event import Event

def load_scenario(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    flights = [Flight(**fd) for fd in data.get('flights', [])]
    takeoff_flights = [f for f in flights if f.dep_airport == "GMP"]
    landing_flights = [f for f in flights if f.arr_airport == "GMP"]
    schedules = [Schedule(f) for f in takeoff_flights]
    events = [Event(e['event_type'], e['target_type'], e['target'], e['time'], e['duration']) for e in data.get('events', [])]
    return schedules, landing_flights, events 