from enum import Enum


class FlightStatus(Enum):
    DORMANT = "dormant"
    TAXI_TO_RUNWAY = "taxiToRunway"
    WAITING = "waiting"
    TAKE_OFF = "takeOff"
    LANDING = "landing"
    TAXI_TO_GATE = "taxiToGate"
    DELAYED = "delayed"
    CANCELLED = "cancelled"


class Flight:
    def __init__(self, flight_id, etd, eta, dep_airport, arr_airport, airline, priority=None):
        self.flight_id = flight_id
        self.etd = etd
        self.eta = eta
        self.dep_airport = dep_airport
        self.arr_airport = arr_airport
        self.airline = airline
        self.priority = priority  # 시나리오에서 직접 설정