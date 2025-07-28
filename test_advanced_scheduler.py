#!/usr/bin/env python3
"""
Test script for the Advanced Scheduler
Verifies that all dependencies are installed and the scheduler works correctly
"""

import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_dependencies():
    """Test if all required dependencies are installed"""
    print("Testing dependencies...")
    
    try:
        import pulp
        print("✅ PuLP installed successfully")
    except ImportError as e:
        print(f"❌ PuLP not installed: {e}")
        return False
    
    try:
        import numpy
        print("✅ NumPy installed successfully")
    except ImportError as e:
        print(f"❌ NumPy not installed: {e}")
        return False
    
    try:
        import scipy
        print("✅ SciPy installed successfully")
    except ImportError as e:
        print(f"❌ SciPy not installed: {e}")
        return False
    
    return True

def test_advanced_scheduler():
    """Test if the advanced scheduler can be imported and initialized"""
    print("\nTesting Advanced Scheduler...")
    
    try:
        from sim.advanced_scheduler import AdvancedScheduler
        print("✅ AdvancedScheduler imported successfully")
        
        # Create a mock simulation object
        class MockSim:
            def __init__(self):
                self.time = 0
                self.weather = None
                self.airport = None
        
        mock_sim = MockSim()
        scheduler = AdvancedScheduler(mock_sim)
        print("✅ AdvancedScheduler initialized successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ AdvancedScheduler test failed: {e}")
        return False

def test_basic_optimization():
    """Test basic optimization functionality"""
    print("\nTesting basic optimization...")
    
    try:
        from sim.advanced_scheduler import AdvancedScheduler
        from sim.schedule import Schedule
        from sim.flight import Flight, FlightStatus
        from sim.airport import Airport, Runway
        
        # Create mock objects with realistic flight data matching random_scenario.json format
        # Takeoff flight (etd set, eta null)
        flight1 = Flight("OZ104", 1157, None, "GMP", "ZAX", "OZ", 32)
        # Landing flight (etd null, eta set)
        flight2 = Flight("OZ995", None, 1518, "HKT", "GMP", "OZ", 64)
        
        runway1 = Runway("14L", "32R")
        runway2 = Runway("14R", "32L")
        airport = Airport("GMP", "Gimpo", [runway1, runway2], [])
        
        schedule1 = Schedule(flight1, 1157, None, True, 32, runway1)  # Takeoff
        schedule2 = Schedule(flight2, None, 1518, False, 64, runway2)  # Landing
        
        class MockSim:
            def __init__(self):
                self.time = 0
                self.weather = None
                self.airport = airport
        
        mock_sim = MockSim()
        scheduler = AdvancedScheduler(mock_sim)
        
        # Test optimization with empty schedules
        result = scheduler.optimize([], 0)
        print("✅ Basic optimization test passed")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic optimization test failed: {e}")
        return False

def test_state_transitions():
    """Test that the advanced scheduler correctly handles state transitions and timing"""
    print("\nTesting state transitions...")
    
    try:
        from sim.scheduler import Scheduler
        from sim.schedule import Schedule
        from sim.flight import Flight, FlightStatus
        from sim.airport import Airport, Runway
        from sim.simulation import Simulation, SimulationMode
        import time
        
        # Create realistic flight data
        flight1 = Flight("OZ104", 1157, None, "GMP", "ZAX", "OZ", 32)  # Takeoff
        flight2 = Flight("OZ995", None, 1518, "HKT", "GMP", "OZ", 64)  # Landing
        
        runway1 = Runway("14L", "32R")
        runway2 = Runway("14R", "32L")
        airport = Airport("GMP", "Gimpo", [runway1, runway2], [])
        
        schedule1 = Schedule(flight1, is_takeoff=True, priority=32)  # Takeoff
        schedule1.runway = runway1  # Manually assign runway
        
        # Create simulation with advanced scheduler
        # Takeoff flight in schedules, landing flight in landing_flights
        sim = Simulation(airport, [schedule1], [flight2], mode=SimulationMode.HEADLESS)
        
        # Verify scheduler is using advanced algorithm
        assert isinstance(sim.scheduler, Scheduler), "Simulation should use Scheduler class"
        assert sim.scheduler.algorithm == "advanced", f"Scheduler should use 'advanced' algorithm, got {sim.scheduler.algorithm}"
        print("✅ Advanced scheduler correctly initialized in simulation")
        
        # Test initial states
        assert schedule1.status == FlightStatus.DORMANT, f"Takeoff flight should start as DORMANT, got {schedule1.status}"
        assert len(sim.schedules) == 1, f"Should have 1 schedule initially, got {len(sim.schedules)}"
        assert len(sim.landing_flights) == 1, f"Should have 1 landing flight, got {len(sim.landing_flights)}"
        print("✅ Initial flight states correct")
        
        # Test scheduler optimization with only takeoff flight
        result = sim.scheduler.optimize([schedule1], 1150, [], {})
        print(f"✅ Scheduler optimization completed: {len(result)} assignments")
        
        # Test state transition timing for takeoff
        print("\nTesting takeoff state transition timing...")
        
        # Simulate a few time steps to verify takeoff transitions
        test_times = [1150, 1155, 1157, 1158]  # Before taxi, taxi start, takeoff, after takeoff
        
        for test_time in test_times:
            sim.time = test_time
            sim.update_status()
            
            # Check takeoff flight state
            if test_time == 1150:
                assert schedule1.status == FlightStatus.DORMANT, f"At {test_time}, takeoff should be DORMANT"
            elif test_time == 1155:
                assert schedule1.status == FlightStatus.TAXI_TO_RUNWAY, f"At {test_time}, takeoff should be TAXI_TO_RUNWAY"
            elif test_time == 1157:
                assert schedule1.status == FlightStatus.TAKE_OFF, f"At {test_time}, takeoff should be TAKE_OFF"
            elif test_time == 1158:
                assert schedule1.status == FlightStatus.DORMANT, f"At {test_time}, takeoff should be DORMANT (completed)"
            
            print(f"  Time {test_time}: Takeoff flight status = {schedule1.status.value}")
        
        print("✅ Takeoff state transitions work correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ State transition test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_landing_state_transitions():
    """Test landing flight state transitions specifically"""
    print("\nTesting landing state transitions...")
    
    try:
        from sim.scheduler import Scheduler
        from sim.schedule import Schedule
        from sim.flight import Flight, FlightStatus
        from sim.airport import Airport, Runway
        from sim.simulation import Simulation, SimulationMode
        
        # Create landing flight
        flight = Flight("OZ995", None, 1518, "HKT", "GMP", "OZ", 64)
        runway = Runway("14R", "32L")
        airport = Airport("GMP", "Gimpo", [runway], [])
        
        # Create simulation with landing flight in landing_flights
        sim = Simulation(airport, [], [flight], mode=SimulationMode.HEADLESS)
        
        # Verify initial state
        assert len(sim.schedules) == 0, f"Should have 0 schedules initially, got {len(sim.schedules)}"
        assert len(sim.landing_flights) == 1, f"Should have 1 landing flight, got {len(sim.landing_flights)}"
        print("✅ Initial state correct - landing flight in landing_flights list")
        
        # Simulate LANDING_ANNOUNCE event at time 1498 (20 minutes before ETA)
        sim.time = 1498
        landing_announce_event = type('Event', (), {
            'event_type': 'LANDING_ANNOUNCE',
            'target': flight.flight_id,
            'duration': 20,
            'time': 1498
        })()
        
        sim.event_handler.handle(landing_announce_event, 1498)
        
        # Verify flight was added to schedules with WAITING status
        assert len(sim.schedules) == 1, f"Should have 1 schedule after LANDING_ANNOUNCE, got {len(sim.schedules)}"
        schedule = sim.schedules[0]
        assert schedule.flight.flight_id == "OZ995", f"Schedule should be for OZ995, got {schedule.flight.flight_id}"
        assert schedule.status == FlightStatus.WAITING, f"Landing flight should be WAITING after announce, got {schedule.status}"
        
        # Assign runway to the schedule (this normally happens during optimization)
        schedule.runway = runway
        print("✅ Landing flight added to schedules with WAITING status and runway assigned")
        
        # Test landing state transitions
        test_times = [1515, 1518, 1519, 1529]  # Before landing, landing start, after landing, after taxi to gate
        
        for test_time in test_times:
            sim.time = test_time
            sim.update_status()
            
            if test_time == 1515:
                assert schedule.status == FlightStatus.WAITING, f"At {test_time}, landing should be WAITING"
            elif test_time == 1518:
                assert schedule.status == FlightStatus.LANDING, f"At {test_time}, landing should be LANDING"
            elif test_time == 1519:
                assert schedule.status == FlightStatus.TAXI_TO_GATE, f"At {test_time}, landing should be TAXI_TO_GATE"
            elif test_time == 1529:
                assert schedule.status == FlightStatus.DORMANT, f"At {test_time}, landing should be DORMANT (completed)"
            
            print(f"  Time {test_time}: Landing flight status = {schedule.status.value}")
        
        print("✅ Landing state transitions work correctly")
        return True
        
    except Exception as e:
        print(f"❌ Landing state transition test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_state_update_messages():
    """Test that the advanced scheduler correctly sends state_update messages with proper flight status"""
    print("\nTesting state update messages...")
    
    try:
        from sim.scheduler import Scheduler
        from sim.schedule import Schedule
        from sim.flight import Flight, FlightStatus
        from sim.airport import Airport, Runway
        from sim.simulation import Simulation, SimulationMode
        
        # Create landing flight for testing
        flight = Flight("OZ995", None, 1518, "HKT", "GMP", "OZ", 64)
        runway = Runway("14R", "32L")
        airport = Airport("GMP", "Gimpo", [runway], [])
        
        # Create simulation with landing flight in landing_flights
        sim = Simulation(airport, [], [flight], mode=SimulationMode.HEADLESS)
        
        # Simulate LANDING_ANNOUNCE event to add flight to schedules
        sim.time = 1498
        landing_announce_event = type('Event', (), {
            'event_type': 'LANDING_ANNOUNCE',
            'target': flight.flight_id,
            'duration': 20,
            'time': 1498
        })()
        
        sim.event_handler.handle(landing_announce_event, 1498)
        schedule = sim.schedules[0]
        
        # Assign runway to the schedule (this normally happens during optimization)
        schedule.runway = runway
        
        # Test state update message format - landing flights start as WAITING after announce
        test_times = [1515, 1518, 1519, 1529]
        expected_statuses = ["waiting", "landing", "taxiToGate", "dormant"]
        
        for i, test_time in enumerate(test_times):
            sim.time = test_time
            sim.update_status()
            
            # Get state update message
            state = sim.get_state()
            
            # Verify message format
            assert state["type"] == "state_update", "State message should have type 'state_update'"
            assert "flights" in state, "State message should contain flights"
            assert "weather" in state, "State message should contain weather"
            assert "speed" in state, "State message should contain speed"
            
            # Verify flight status
            if state["flights"]:
                flight_data = state["flights"][0]
                assert flight_data["flight_id"] == "OZ995", f"Flight ID should be OZ995, got {flight_data['flight_id']}"
                assert flight_data["status"] == expected_statuses[i], f"At time {test_time}, status should be {expected_statuses[i]}, got {flight_data['status']}"
                
                print(f"  Time {test_time}: Flight status in state_update = {flight_data['status']}")
            
            # Verify weather data format
            weather_data = state["weather"][0]
            required_fields = ["condition", "visibility", "wind_speed", "wind_direction", "temperature", "pressure"]
            for field in required_fields:
                assert field in weather_data, f"Weather data should contain {field}"
        
        print("✅ State update messages correctly formatted")
        return True
        
    except Exception as e:
        print(f"❌ State update message test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🚀 Advanced Scheduler Test Suite")
    print("=" * 40)
    
    # Test dependencies
    if not test_dependencies():
        print("\n❌ Dependency test failed. Please install missing packages:")
        print("pip install -r requirements.txt")
        return False
    
    # Test scheduler import
    if not test_advanced_scheduler():
        print("\n❌ Scheduler import test failed.")
        return False
    
    # Test basic optimization
    if not test_basic_optimization():
        print("\n❌ Basic optimization test failed.")
        return False
    
    # Test state transitions
    if not test_state_transitions():
        print("\n❌ State transition test failed.")
        return False

    # Test landing state transitions
    if not test_landing_state_transitions():
        print("\n❌ Landing state transition test failed.")
        return False
    
    # Test state update messages
    if not test_state_update_messages():
        print("\n❌ State update message test failed.")
        return False
    
    print("\n🎉 All tests passed! Advanced Scheduler is ready to use.")
    print("\nTo use the advanced scheduler:")
    print("1. The simulation.py has been updated to use AdvancedScheduler")
    print("2. Run your simulation as usual")
    print("3. The scheduler will automatically use MILP optimization with weather awareness")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 