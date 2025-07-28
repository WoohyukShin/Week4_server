from pulp import *
import numpy as np
from typing import List, Dict, Tuple, Optional
from sim.schedule import Schedule
from sim.weather import Weather
from sim.event import Event
from sim.flight import FlightStatus
from utils.logger import debug
from utils.time_utils import int_to_hhmm_colon
import heapq

class AdvancedScheduler:
    """
    Advanced Aircraft Scheduling Algorithm based on research papers:
    - MINLP Aircraft Scheduling
    - Priority-based Aircraft Landing Scheduling (PALS)
    - Weather-aware optimization
    """
    
    def __init__(self, sim):
        self.sim = sim
        self.time_horizon = 120  # 2 hours ahead (in minutes)
        self.min_separation = 4  # 4 minutes between operations
        self.max_optimization_time = 30  # Maximum seconds for optimization
        
        # Cost weights for multi-objective optimization
        self.delay_weight = 1.0
        self.weather_risk_weight = 50.0
        self.safety_weight = 100.0
        self.priority_weight = 2.0
        
        # Weather risk thresholds
        self.high_risk_threshold = 3.0
        self.medium_risk_threshold = 2.0
        
    def optimize(self, schedules: List[Schedule], current_time: int, 
                event_queue: Optional[List[Event]] = None, 
                forecast: Optional[Dict] = None,
                runway_availability: Optional[Dict] = None) -> Dict:
        """
        Main optimization method - replaces the greedy algorithm
        """
        debug(f"Advanced scheduler started at {int_to_hhmm_colon(current_time)}")
        
        if not schedules:
            return {}
            
        # Preprocess data
        runway_constraints = self._analyze_runway_constraints(event_queue or [], current_time, runway_availability)
        weather_risks = self._calculate_weather_risks(forecast or {}, current_time)
        
        # Separate takeoff and landing schedules
        takeoff_schedules = [s for s in schedules if s.is_takeoff and s.status == FlightStatus.DORMANT]
        landing_schedules = [s for s in schedules if not s.is_takeoff and s.status == FlightStatus.WAITING]
        
        debug(f"Takeoff schedules: {len(takeoff_schedules)}, Landing schedules: {len(landing_schedules)}")
        
        # Run optimization
        try:
            optimized_schedule = self._run_milp_optimization(
                takeoff_schedules, landing_schedules, current_time, 
                runway_constraints, weather_risks
            )
            
            # Fallback to heuristic if MILP fails
            if not optimized_schedule:
                debug("MILP optimization failed, using heuristic fallback")
                optimized_schedule = self._heuristic_optimization(
                    takeoff_schedules, landing_schedules, current_time,
                    runway_constraints, weather_risks, runway_availability
                )
                
        except Exception as e:
            debug(f"Optimization error: {e}, using heuristic fallback")
            optimized_schedule = self._heuristic_optimization(
                takeoff_schedules, landing_schedules, current_time,
                runway_constraints, weather_risks, runway_availability
            )
        
        # Log results
        self._log_optimization_results(optimized_schedule, current_time)
        
        # Verify separation in results
        if optimized_schedule:
            self._verify_separation(optimized_schedule, current_time)
        
        return optimized_schedule
    
    def _analyze_runway_constraints(self, event_queue: List[Event], current_time: int, runway_availability: Optional[Dict]) -> Dict:
        """
        Analyze runway constraints from events
        """
        constraints = {
            'closures': [],
            'inversions': [],
            'runway_availability': {}
        }
        
        # Initialize runway availability for all possible runway directions
        for runway in self.sim.airport.runways:
            # Initialize for both normal and inverted names
            constraints['runway_availability'][runway.name] = []
            constraints['runway_availability'][runway.inverted_name] = []
            
            # Mark current availability for both directions using next_available_time
            for t in range(self.time_horizon):
                time_step = current_time + t
                
                # Use runway_availability if provided, otherwise fall back to runway state
                if runway_availability and runway.name in runway_availability:
                    next_available = runway_availability[runway.name]
                    available = not runway.closed and next_available <= time_step
                else:
                    available = not runway.closed and runway.next_available_time <= time_step
                
                # Both normal and inverted names get the same availability initially
                constraints['runway_availability'][runway.name].append(available)
                constraints['runway_availability'][runway.inverted_name].append(available)
        
        # Process events
        for event in event_queue:
            if event.event_type == "RUNWAY_CLOSURE":
                constraints['closures'].append({
                    'runway': event.target,
                    'start_time': event.time,
                    'end_time': event.time + event.duration
                })
                
                # Update availability for the specific runway
                if event.target in constraints['runway_availability']:
                    for t in range(self.time_horizon):
                        time_step = current_time + t
                        if event.time <= time_step <= event.time + event.duration:
                            constraints['runway_availability'][event.target][t] = False
                        
            elif event.event_type == "RUNWAY_INVERT":
                constraints['inversions'].append(event.time)
                
        return constraints
    
    def _calculate_weather_risks(self, forecast: Dict, current_time: int) -> Dict:
        """
        Calculate weather risks for each time step
        """
        risks = {}
        
        for t in range(self.time_horizon):
            time_step = current_time + t
            
            # Get weather forecast for this time step
            if time_step in forecast:
                weather_data = forecast[time_step]
            else:
                # Use current weather if no forecast available
                weather_data = self.sim.weather.get_weather_info()
            
            # Calculate landing risk
            landing_risk = 1.0
            if weather_data.get('visibility', 10) < 3:
                landing_risk *= 2.5
            if weather_data.get('wind_speed', 0) > 15:
                landing_risk *= 1.8
            if weather_data.get('condition') in ['storm', 'heavy_fog']:
                landing_risk *= 3.0
            elif weather_data.get('condition') in ['snow', 'moderate_fog']:
                landing_risk *= 2.0
            elif weather_data.get('condition') in ['rain', 'light_fog']:
                landing_risk *= 1.5
                
            # Calculate takeoff risk (generally lower than landing)
            takeoff_risk = 1.0
            if weather_data.get('visibility', 10) < 1:
                takeoff_risk *= 2.0
            if weather_data.get('wind_speed', 0) > 20:
                takeoff_risk *= 1.5
            if weather_data.get('condition') in ['storm']:
                takeoff_risk *= 2.5
            elif weather_data.get('condition') in ['heavy_fog', 'snow']:
                takeoff_risk *= 1.8
            elif weather_data.get('condition') in ['rain', 'moderate_fog']:
                takeoff_risk *= 1.3
                
            risks[t] = {
                'landing_risk': landing_risk,
                'takeoff_risk': takeoff_risk,
                'weather_data': weather_data
            }
        
        return risks
    
    def _run_milp_optimization(self, takeoff_schedules: List[Schedule], 
                              landing_schedules: List[Schedule], current_time: int,
                              runway_constraints: Dict, weather_risks: Dict) -> Dict:
        """
        Run Mixed Integer Linear Programming optimization
        """
        all_schedules = takeoff_schedules + landing_schedules
        if not all_schedules:
            return {}
            
        # Create optimization problem
        prob = LpProblem("Aircraft_Scheduling", LpMinimize)
        
        # Decision variables: x[i][t] = 1 if schedule i is assigned to time t
        x = {}
        for i, schedule in enumerate(all_schedules):
            for t in range(self.time_horizon):
                x[i, t] = LpVariable(f"x_{i}_{t}", 0, 1, LpBinary)
        
        # Objective function: minimize total cost
        objective = 0
        
        # Delay cost (weighted by priority)
        for i, schedule in enumerate(all_schedules):
            for t in range(self.time_horizon):
                original_time = schedule.etd if schedule.is_takeoff else schedule.eta
                delay = max(0, (current_time + t) - original_time)
                
                if schedule.is_takeoff:
                    objective += (schedule.priority * self.delay_weight * delay * x[i, t])
                else:
                    objective += (schedule.priority * self.delay_weight * delay * 1.2 * x[i, t])  # Landing delays cost more
        
        # Weather risk cost
        for i, schedule in enumerate(all_schedules):
            for t in range(self.time_horizon):
                risk = weather_risks[t]['takeoff_risk'] if schedule.is_takeoff else weather_risks[t]['landing_risk']
                objective += (risk * self.weather_risk_weight * x[i, t])
        
        prob += objective
        
        # Constraints
        
        # 1. Each schedule must be assigned exactly once
        for i, schedule in enumerate(all_schedules):
            prob += lpSum(x[i, t] for t in range(self.time_horizon)) == 1
        
        # 2. Separation constraints (4 minutes between operations on same runway)
        for t in range(self.time_horizon - self.min_separation):
            for runway in self.sim.airport.runways:
                runway_direction = runway.get_current_direction()
                
                # Count operations on this runway in time window
                operations_in_window = lpSum(
                    x[i, t_prime] 
                    for i, schedule in enumerate(all_schedules)
                    for t_prime in range(t, min(t + self.min_separation, self.time_horizon))
                    if self._get_assigned_runway(schedule, runway_direction) == runway_direction
                )
                
                prob += operations_in_window <= 1
        
        # 2b. Additional separation constraint: No two operations at exactly the same time
        for t in range(self.time_horizon):
            for runway in self.sim.airport.runways:
                runway_direction = runway.get_current_direction()
                
                # Count operations on this runway at this exact time
                operations_at_time = lpSum(
                    x[i, t] 
                    for i, schedule in enumerate(all_schedules)
                    if self._get_assigned_runway(schedule, runway_direction) == runway_direction
                )
                
                prob += operations_at_time <= 1
        
        # 3. Runway availability constraints
        for i, schedule in enumerate(all_schedules):
            for t in range(self.time_horizon):
                assigned_runway = self._get_assigned_runway(schedule, None)
                if assigned_runway in runway_constraints['runway_availability']:
                    if not runway_constraints['runway_availability'][assigned_runway][t]:
                        prob += x[i, t] == 0
        
        # 4. Priority constraints (higher priority flights should not be delayed more than necessary)
        for i, schedule_i in enumerate(all_schedules):
            for j, schedule_j in enumerate(all_schedules):
                if i != j and schedule_i.priority > schedule_j.priority:
                    # Higher priority flights should not be significantly delayed if lower priority flights can be delayed instead
                    for t in range(self.time_horizon):
                        original_time_i = schedule_i.etd if schedule_i.is_takeoff else schedule_i.eta
                        original_time_j = schedule_j.etd if schedule_j.is_takeoff else schedule_j.eta
                        
                        delay_i = max(0, (current_time + t) - original_time_i)
                        delay_j = max(0, (current_time + t) - original_time_j)
                        
                        if delay_i > delay_j + 10:  # Allow some flexibility
                            prob += x[i, t] <= 0.5  # Soft constraint
        
        # 5. ETA/ETD constraints (planes cannot land/take off earlier than scheduled)
        for i, schedule in enumerate(all_schedules):
            original_time = schedule.etd if schedule.is_takeoff else schedule.eta
            if original_time is not None:
                for t in range(self.time_horizon):
                    proposed_time = current_time + t
                    if proposed_time < original_time:
                        # Cannot assign to time earlier than ETA/ETD
                        prob += x[i, t] == 0
        
        # Solve the problem
        prob.solve(PULP_CBC_CMD(msg=0, timeLimit=self.max_optimization_time))
        
        # Extract solution
        if prob.status == 1:  # Optimal solution found
            solution = {}
            for i, schedule in enumerate(all_schedules):
                for t in range(self.time_horizon):
                    if x[i, t].value() == 1:
                        solution[schedule.flight.flight_id] = current_time + t
                        break
            return solution
        else:
            debug(f"MILP optimization failed with status: {prob.status}")
            return {}
    
    def _heuristic_optimization(self, takeoff_schedules: List[Schedule], 
                               landing_schedules: List[Schedule], current_time: int,
                               runway_constraints: Dict, weather_risks: Dict,
                               runway_availability: Optional[Dict] = None) -> Dict:
        """
        Heuristic fallback optimization based on PALS algorithm
        """
        solution = {}
        
        # Sort schedules by priority and original time
        takeoff_schedules.sort(key=lambda s: (-s.priority, s.etd))
        landing_schedules.sort(key=lambda s: (-s.priority, s.eta))
        
        # Track runway usage with proper separation
        runway_usage = {
            '14L': 0, '14R': 0, '32L': 0, '32R': 0
        }
        
        # Initialize runway usage with actual next_available_time if provided
        if runway_availability:
            for runway in self.sim.airport.runways:
                if runway.name in runway_availability:
                    runway_usage[runway.name] = max(runway_usage[runway.name], runway_availability[runway.name])
                    runway_usage[runway.inverted_name] = max(runway_usage[runway.inverted_name], runway_availability[runway.name])
        
        # Process takeoff schedules
        for schedule in takeoff_schedules:
            assigned_time = self._assign_takeoff_time(
                schedule, current_time, runway_usage, runway_constraints, weather_risks
            )
            if assigned_time is not None:
                solution[schedule.flight.flight_id] = assigned_time
                assigned_runway = self._get_assigned_runway(schedule, None)
                # Ensure 4-minute separation
                runway_usage[assigned_runway] = assigned_time + self.min_separation
        
        # Process landing schedules
        for schedule in landing_schedules:
            assigned_time = self._assign_landing_time(
                schedule, current_time, runway_usage, runway_constraints, weather_risks
            )
            if assigned_time is not None:
                solution[schedule.flight.flight_id] = assigned_time
                assigned_runway = self._get_assigned_runway(schedule, None)
                # Ensure 4-minute separation
                runway_usage[assigned_runway] = assigned_time + self.min_separation
        
        return solution
    
    def _assign_takeoff_time(self, schedule: Schedule, current_time: int, 
                           runway_usage: Dict, runway_constraints: Dict, 
                           weather_risks: Dict) -> Optional[int]:
        """
        Assign takeoff time for a schedule
        """
        original_time = max(current_time, schedule.etd)  # Cannot take off earlier than ETD
        assigned_runway = self._get_assigned_runway(schedule, None)
        
        # Find earliest available time considering constraints
        for t in range(self.time_horizon):
            proposed_time = current_time + t
            
            # Constraint 1: Cannot take off earlier than ETD
            if proposed_time < schedule.etd:
                continue
                
            # Constraint 2: Runway availability (must be after next_available_time)
            if proposed_time < runway_usage.get(assigned_runway, 0):
                continue
                
            # Constraint 3: Runway closure constraints
            if not self._is_runway_available(assigned_runway, proposed_time, runway_constraints):
                continue
                
            # Constraint 4: Weather risk
            if weather_risks[t]['takeoff_risk'] > self.high_risk_threshold:
                continue  # Skip high-risk time slots
                
            return proposed_time
        
        return None
    
    def _assign_landing_time(self, schedule: Schedule, current_time: int, 
                           runway_usage: Dict, runway_constraints: Dict, 
                           weather_risks: Dict) -> Optional[int]:
        """
        Assign landing time for a schedule
        """
        original_time = max(current_time, schedule.eta)  # Cannot land earlier than ETA
        assigned_runway = self._get_assigned_runway(schedule, None)
        
        # Find earliest available time considering constraints
        for t in range(self.time_horizon):
            proposed_time = current_time + t
            
            # Constraint 1: Cannot land earlier than ETA
            if proposed_time < schedule.eta:
                continue
                
            # Constraint 2: Runway availability (must be after next_available_time)
            if proposed_time < runway_usage.get(assigned_runway, 0):
                continue
                
            # Constraint 3: Runway closure constraints
            if not self._is_runway_available(assigned_runway, proposed_time, runway_constraints):
                continue
                
            # Constraint 4: Weather risk
            if weather_risks[t]['landing_risk'] > self.high_risk_threshold:
                continue  # Skip high-risk time slots
                
            return proposed_time
        
        return None
    
    def _get_assigned_runway(self, schedule: Schedule, preferred_runway: Optional[str] = None) -> str:
        """
        Get the assigned runway for a schedule
        """
        if preferred_runway:
            return preferred_runway
            
        # Default runway assignment
        if schedule.is_takeoff:
            # Check if 14L is available for takeoff
            for runway in self.sim.airport.runways:
                if runway.get_current_direction() in ["14L", "32R"] and not runway.closed:
                    return runway.get_current_direction()
            # Fallback to 14R/32L
            for runway in self.sim.airport.runways:
                if runway.get_current_direction() in ["14R", "32L"] and not runway.closed:
                    return runway.get_current_direction()
        else:
            # Landing: prefer 14R/32L
            for runway in self.sim.airport.runways:
                if runway.get_current_direction() in ["14R", "32L"] and not runway.closed:
                    return runway.get_current_direction()
            # Fallback to 14L/32R
            for runway in self.sim.airport.runways:
                if runway.get_current_direction() in ["14L", "32R"] and not runway.closed:
                    return runway.get_current_direction()
        
        return "14L"  # Default fallback
    
    def _is_runway_available(self, runway_direction: str, time: int, 
                           runway_constraints: Dict) -> bool:
        """
        Check if runway is available at given time
        """
        # First check if runway exists in constraints
        if runway_direction not in runway_constraints['runway_availability']:
            return True
            
        # Check constraints-based availability
        time_index = time - self.sim.time
        if 0 <= time_index < len(runway_constraints['runway_availability'][runway_direction]):
            if not runway_constraints['runway_availability'][runway_direction][time_index]:
                return False
        
        # Also check actual runway state using can_handle_operation
        for runway in self.sim.airport.runways:
            if (runway.get_current_direction() == runway_direction or 
                runway.name == runway_direction or 
                runway.inverted_name == runway_direction):
                return runway.can_handle_operation(time)
        
        return True
    
    def _log_optimization_results(self, solution: Dict, current_time: int):
        """
        Log optimization results
        """
        debug("===== ADVANCED SCHEDULER RESULTS =====")
        for flight_id, assigned_time in solution.items():
            debug(f"{flight_id}: {int_to_hhmm_colon(assigned_time)}")
        debug("=====================================")
    
    def _verify_separation(self, solution: Dict, current_time: int):
        """Verify that proper separation is maintained in the solution"""
        debug("===== VERIFYING SEPARATION =====")
        
        # Group operations by runway
        runway_operations = {'14L': [], '14R': [], '32L': [], '32R': []}
        
        for flight_id, assigned_time in solution.items():
            # Find the schedule for this flight
            schedule = None
            for s in self.sim.schedules:
                if s.flight.flight_id == flight_id:
                    schedule = s
                    break
            
            if schedule:
                assigned_runway = self._get_assigned_runway(schedule, None)
                runway_operations[assigned_runway].append((flight_id, assigned_time))
        
        # Check separation for each runway
        for runway, operations in runway_operations.items():
            if len(operations) > 1:
                operations.sort(key=lambda x: x[1])  # Sort by time
                for i in range(len(operations) - 1):
                    time_diff = operations[i+1][1] - operations[i][1]
                    if time_diff < self.min_separation:
                        debug(f"⚠️  SEPARATION VIOLATION: {operations[i][0]} and {operations[i+1][0]} on {runway} only {time_diff} minutes apart!")
                    else:
                        debug(f"✅ Separation OK: {operations[i][0]} and {operations[i+1][0]} on {runway} {time_diff} minutes apart")
        
        debug("================================")
    
    def adaptive_schedule_update(self, current_schedules: List[Schedule], 
                               new_events: List[Event], weather_update: Dict) -> Dict:
        """
        Real-time schedule adaptation
        """
        debug("Adaptive schedule update triggered")
        
        # Identify affected time window
        affected_window = self._identify_affected_window(new_events, weather_update)
        
        # Get committed operations (within next 10 minutes)
        committed_operations = self._get_committed_operations(current_schedules, self.sim.time + 10)
        
        # Re-optimize remaining operations
        remaining_schedules = self._get_remaining_schedules(current_schedules, committed_operations)
        
        # Run optimization for remaining schedules
        updated_schedule = self.optimize(remaining_schedules, self.sim.time, new_events, weather_update)
        
        # Merge with committed operations
        return {**committed_operations, **updated_schedule}
    
    def _identify_affected_window(self, new_events: List[Event], weather_update: Dict) -> Tuple[int, int]:
        """
        Identify time window affected by new events/weather
        """
        start_time = self.sim.time
        end_time = self.sim.time + self.time_horizon
        
        for event in new_events:
            if event.event_type == "RUNWAY_CLOSURE":
                start_time = min(start_time, event.time)
                end_time = max(end_time, event.time + event.duration)
        
        return start_time, end_time
    
    def _get_committed_operations(self, schedules: List[Schedule], cutoff_time: int) -> Dict:
        """
        Get operations that are already committed (within cutoff time)
        """
        committed = {}
        for schedule in schedules:
            if schedule.is_takeoff and schedule.etd <= cutoff_time:
                committed[schedule.flight.flight_id] = schedule.etd
            elif not schedule.is_takeoff and schedule.eta <= cutoff_time:
                committed[schedule.flight.flight_id] = schedule.eta
        return committed
    
    def _get_remaining_schedules(self, schedules: List[Schedule], committed_operations: Dict) -> List[Schedule]:
        """
        Get schedules that are not yet committed
        """
        committed_ids = set(committed_operations.keys())
        return [s for s in schedules if s.flight.flight_id not in committed_ids] 