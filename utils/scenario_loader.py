import json
from sim.flight import Flight
from sim.event import Event

def load_scenario(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    flights = [Flight(**fd) for fd in data.get('flights', [])]
    events = [Event(e['event_type'], e['target_type'], e['target'], e['time'], e['duration']) for e in data.get('events', [])]
    return flights, events 