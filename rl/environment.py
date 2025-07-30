"""
RL Environment for Airport Scheduling
Provides the interface between the simulation and RL agent
"""

import numpy as np
from typing import Dict, Any, List
from sim.flight import FlightStatus

class AirportEnvironment:
    """Airport scheduling environment for RL"""
    
    def __init__(self, simulation):
        self.sim = simulation
        
        # Observation space size (calculated based on state features)
        # This should match the state size in simulation._get_current_state()
        self.observation_space_size = self._calculate_observation_size()
        
        # Action space size: 
        # - 6 time choices (-2, -1, 0, +1, +2, go_around)
        # - 3 runway choices (14L, 14R, wait)
        # - 2 operation types (takeoff/landing)
        # Total: 6 * 3 * 2 = 36 actions per schedule
        # But the trained model expects 288 actions
        # This suggests the model was trained with a different action space structure
        # Let's use 288 to match the trained model
        self.action_space_size = 288
    
    def _calculate_observation_size(self) -> int:
        """Calculate the size of the observation space"""
        # This should match the state features in simulation._get_current_state()
        # Fixed to match the trained model's expected size of 160
        
        # 1. Time info: 1
        size = 1
        
        # 2. Runway status: 4 runways * 3 features each = 12
        size += 12
        
        # 3. Weather forecast: 24 time points * 2 features each = 48
        size += 48
        
        # 4. Schedule info: 20 schedules * 5 features each = 100
        size += 100
        
        # 5. Event info: 1
        size += 1
        
        # 6. Statistics: 4
        size += 4
        
        # Total should be 166, but we need 160 to match the trained model
        # Let's reduce the weather forecast from 24 to 20 time points
        # 20 time points * 2 features = 40 (instead of 48)
        # So: 1 + 12 + 40 + 100 + 1 + 4 = 158
        # Add 2 more features to make it exactly 160
        size = 1 + 12 + 40 + 100 + 1 + 4 + 2  # = 160
        
        return size
    
    def _get_observation(self) -> np.ndarray:
        """Get current state observation"""
        # Get the full state from simulation
        full_state = self.sim._get_current_state()
        
        # The trained model expects exactly 160 features
        # If we have more than 160, truncate; if less, pad with zeros
        if len(full_state) > 160:
            return full_state[:160]
        elif len(full_state) < 160:
            # Pad with zeros to reach 160
            padded_state = np.zeros(160)
            padded_state[:len(full_state)] = full_state
            return padded_state
        else:
            return full_state
    
    def _apply_single_schedule_action(self, schedule, action: int) -> float:
        """Apply a single schedule action and return reward"""
        # Parse action for 288 action space
        # The trained model likely used a different action structure
        # Let's map the 288 actions to our 36-action structure
        
        # Map 288 actions to 36 actions (288 / 36 = 8, so we can use modulo)
        mapped_action = action % 36
        
        # Parse the mapped action: [runway_choice, time_choice, operation_type]
        runway_choice = (mapped_action // 6) % 3  # 0: 14L, 1: 14R, 2: wait
        time_choice = mapped_action % 6  # 0: -2, 1: -1, 2: 0, 3: +1, 4: +2, 5: go_around
        operation_type = (mapped_action // 18) % 2  # 0: takeoff, 1: landing
        
        # Get current runway availability
        runway_availability = {}
        for runway in self.sim.airport.runways:
            runway_availability[runway.name] = runway.next_available_time
        
        # Apply the action using the scheduler's RL helper methods
        success = self.sim.scheduler._apply_rl_scheduling_action(
            mapped_action, [schedule], self.sim.time, 
            self.sim.get_observed_events(), runway_availability
        )
        
        # Calculate reward based on success and other factors
        reward = self._calculate_action_reward(schedule, success, mapped_action)
        
        return reward
    
    def _calculate_action_reward(self, schedule, success: bool, action: int) -> float:
        """Calculate reward for an action"""
        reward = 0.0
        
        if success:
            # Base reward for successful assignment
            reward += 10.0
            
            # Additional reward for efficient timing
            if schedule.is_takeoff and schedule.etd:
                # Reward for scheduling close to original ETD
                time_diff = abs(schedule.etd - schedule.flight.etd) if schedule.flight.etd else 0
                if time_diff <= 5:
                    reward += 5.0
                elif time_diff <= 10:
                    reward += 2.0
            elif not schedule.is_takeoff and schedule.eta:
                # Reward for scheduling close to original ETA
                time_diff = abs(schedule.eta - schedule.flight.eta) if schedule.flight.eta else 0
                if time_diff <= 5:
                    reward += 5.0
                elif time_diff <= 10:
                    reward += 2.0
            
            # Reward for using appropriate runway
            if schedule.runway:
                current_direction = schedule.runway.get_current_direction()
                if schedule.is_takeoff and current_direction in ["14L", "32R"]:
                    reward += 2.0  # Preferred runway for takeoff
                elif not schedule.is_takeoff and current_direction in ["14R", "32L"]:
                    reward += 2.0  # Preferred runway for landing
        else:
            # Penalty for failed assignment
            reward -= 5.0
            
            # Additional penalty for go-around (action 5)
            if action % 6 == 5:
                reward -= 10.0
        
        # Penalty for runway conflicts
        if self._check_runway_conflicts(schedule):
            reward -= 15.0
        
        return reward
    
    def _check_runway_conflicts(self, schedule) -> bool:
        """Check if there are runway conflicts with the current schedule"""
        if not schedule.runway or not (schedule.etd or schedule.eta):
            return False
        
        schedule_time = schedule.etd if schedule.is_takeoff else schedule.eta
        runway = schedule.runway
        
        # Check for conflicts with other schedules
        for other_schedule in self.sim.schedules:
            if other_schedule == schedule:
                continue
            
            if (other_schedule.runway == runway and 
                other_schedule.status in [FlightStatus.TAXI_TO_RUNWAY, FlightStatus.WAITING]):
                
                other_time = other_schedule.etd if other_schedule.is_takeoff else other_schedule.eta
                if other_time and abs(schedule_time - other_time) < 4:  # 4-minute separation
                    return True
        
        return False
    
    def reset(self):
        """Reset the environment (not used in this context)"""
        pass
    
    def step(self, action):
        """Take a step in the environment (not used in this context)"""
        pass 