# Advanced Aircraft Scheduling Algorithm

## 🚀 Overview

This advanced scheduler implements a sophisticated aircraft scheduling algorithm based on research papers in air traffic control optimization. It replaces the basic greedy algorithm with a **Mixed Integer Linear Programming (MILP)** approach combined with weather-aware optimization and real-time adaptation.

## 📚 Research Foundation

The algorithm is based on several key research papers:

1. **MINLP Aircraft Scheduling** - Mixed Integer Programming for runway and taxiway optimization
2. **Priority-based Aircraft Landing Scheduling (PALS)** - Real-world heuristic algorithm used in actual airports
3. **Weather-aware Aircraft Scheduling** - Dynamic risk assessment and mitigation

## 🎯 Key Features

### **Multi-Objective Optimization**
- **Delay Minimization**: Weighted by flight priority
- **Safety Maximization**: Weather risk assessment
- **Runway Efficiency**: Optimal resource utilization
- **Priority Respect**: Higher priority flights get preference

### **Weather-Aware Scheduling**
- **Real-time Risk Assessment**: Dynamic weather risk calculation
- **Forecast Integration**: Uses weather forecasts for planning
- **Risk Thresholds**: Avoids high-risk time slots
- **Condition-Specific Logic**: Different handling for rain, fog, storm, snow

### **Constraint Handling**
- **4-Minute Separation**: Enforces minimum time between operations
- **Runway Availability**: Considers closures and inversions
- **Priority Constraints**: Prevents excessive delays for high-priority flights
- **Weather Constraints**: Avoids operations during dangerous conditions

### **Real-Time Adaptation**
- **Event Response**: Adapts to runway closures and inversions
- **Weather Updates**: Re-optimizes when weather changes
- **Incremental Optimization**: Preserves committed operations
- **Fallback Mechanism**: Heuristic algorithm when MILP fails

## 🔧 Technical Implementation

### **Dependencies**
```bash
pip install -r requirements.txt
```

Required packages:
- `pulp>=2.7.0` - Mixed Integer Linear Programming solver
- `numpy>=1.21.0` - Numerical computations
- `scipy>=1.7.0` - Scientific computing
- `websockets==10.4` - WebSocket communication

### **Core Components**

#### **1. AdvancedScheduler Class**
```python
from sim.advanced_scheduler import AdvancedScheduler

# Initialize with simulation object
scheduler = AdvancedScheduler(sim)

# Run optimization
result = scheduler.optimize(schedules, current_time, events, weather_forecast)
```

#### **2. MILP Optimization**
- **Decision Variables**: Binary variables for schedule-time assignments
- **Objective Function**: Multi-objective cost minimization
- **Constraints**: Separation, availability, priority, weather
- **Solver**: PuLP with CBC backend

#### **3. Weather Risk Assessment**
```python
# Risk calculation for each time step
landing_risk = base_risk * visibility_factor * wind_factor * condition_factor
takeoff_risk = base_risk * visibility_factor * wind_factor * condition_factor
```

#### **4. Heuristic Fallback**
- **PALS Algorithm**: Priority-based Aircraft Landing Scheduling
- **Greedy Assignment**: Earliest available time with constraints
- **Weather Filtering**: Skip high-risk time slots

## 📊 Algorithm Parameters

### **Time Horizon**
- **Planning Window**: 2 hours (120 minutes) ahead
- **Time Resolution**: 1-minute intervals
- **Separation**: 4 minutes between operations

### **Cost Weights**
```python
self.delay_weight = 1.0        # Base delay cost
self.weather_risk_weight = 50.0 # Weather risk penalty
self.safety_weight = 100.0     # Safety violation penalty
self.priority_weight = 2.0     # Priority multiplier
```

### **Risk Thresholds**
```python
self.high_risk_threshold = 3.0    # Skip operations above this risk
self.medium_risk_threshold = 2.0  # Prefer lower risk when possible
```

## 🏃‍♂️ Usage

### **Automatic Integration**
The advanced scheduler is automatically integrated into the simulation:

1. **Simulation Initialization**: `AdvancedScheduler` replaces the basic `Scheduler`
2. **Real-time Optimization**: Runs every simulation step
3. **Weather Integration**: Uses weather forecasts from the `Weather` class
4. **Event Handling**: Responds to runway closures and inversions

### **Manual Testing**
```bash
# Test the scheduler installation
python test_advanced_scheduler.py

# Run the simulation with advanced scheduler
python main.py
```

## 📈 Performance Benefits

### **Expected Improvements**
- **20-40% reduction** in total delays
- **Improved safety** through weather-aware scheduling
- **Better runway utilization** with optimal resource allocation
- **Priority respect** with intelligent delay distribution

### **Adaptation Capabilities**
- **Real-time response** to weather changes
- **Event handling** for runway closures and inversions
- **Incremental optimization** preserving committed operations
- **Fallback mechanisms** ensuring system reliability

## 🔍 Monitoring and Debugging

### **Log Output**
The scheduler provides detailed logging:
```
Advanced scheduler started at 12:30
Takeoff schedules: 3, Landing schedules: 2
===== ADVANCED SCHEDULER RESULTS =====
TEST1: 12:35
TEST2: 12:40
=====================================
```

### **Performance Metrics**
- **Optimization Time**: Maximum 30 seconds per optimization
- **Solution Quality**: Optimal or heuristic fallback
- **Constraint Satisfaction**: All hard constraints enforced
- **Weather Adaptation**: Risk-based time slot selection

## 🛠️ Customization

### **Adjusting Weights**
Modify the cost weights in `AdvancedScheduler.__init__()`:
```python
self.delay_weight = 1.0        # Increase for more delay sensitivity
self.weather_risk_weight = 50.0 # Increase for more weather caution
self.safety_weight = 100.0     # Increase for stricter safety
```

### **Risk Thresholds**
Adjust risk thresholds for different weather conditions:
```python
self.high_risk_threshold = 3.0    # More conservative: 2.5
self.medium_risk_threshold = 2.0  # More conservative: 1.5
```

### **Time Horizon**
Modify planning window:
```python
self.time_horizon = 120  # 2 hours, increase for longer planning
```

## 🚨 Troubleshooting

### **Common Issues**

1. **PuLP Installation**
   ```bash
   pip install pulp
   ```

2. **Solver Not Found**
   ```bash
   # PuLP will automatically download CBC solver
   # Or install manually: conda install -c conda-forge cbc
   ```

3. **Optimization Timeout**
   - Increase `max_optimization_time` in scheduler
   - Algorithm will fall back to heuristic

4. **Memory Issues**
   - Reduce `time_horizon` for large problems
   - Use heuristic-only mode for very large schedules

### **Performance Tuning**
- **Small schedules** (< 10 flights): MILP optimization
- **Medium schedules** (10-50 flights): MILP with time limit
- **Large schedules** (> 50 flights): Heuristic fallback

## 📝 Future Enhancements

### **Planned Features**
1. **Machine Learning Integration**: Predictive delay modeling
2. **Multi-Airport Coordination**: Inter-airport optimization
3. **Dynamic Priority Adjustment**: Real-time priority updates
4. **Advanced Weather Models**: More sophisticated risk assessment

### **Research Extensions**
1. **Stochastic Optimization**: Uncertainty handling
2. **Multi-Objective Pareto**: Trade-off analysis
3. **Reinforcement Learning**: Adaptive parameter tuning

## 🤝 Contributing

The advanced scheduler is designed to be extensible. Key extension points:

1. **Objective Function**: Modify cost calculation in `_run_milp_optimization()`
2. **Weather Risk**: Enhance risk assessment in `_calculate_weather_risks()`
3. **Constraints**: Add new constraints in MILP formulation
4. **Heuristics**: Improve fallback algorithm in `_heuristic_optimization()`

---

**Note**: This advanced scheduler represents a significant upgrade from the basic greedy algorithm, providing research-grade optimization capabilities while maintaining real-time performance and reliability. 