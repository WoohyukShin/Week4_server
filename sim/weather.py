import random
from enum import Enum
from utils.logger import debug

class WeatherCondition(Enum):
    CLEAR = "clear"           # 맑음 (시정 > 10km)
    LIGHT_FOG = "light_fog"   # 연무 (시정 5-10km)
    MODERATE_FOG = "moderate_fog"  # 중간 안개 (시정 1-5km)
    HEAVY_FOG = "heavy_fog"   # 짙은 안개 (시정 < 1km)
    RAIN = "rain"             # 비
    SNOW = "snow"             # 눈
    STORM = "storm"           # 폭풍

class Weather:
    def __init__(self, initial_condition=None):
        self.condition = initial_condition or WeatherCondition.CLEAR
        self.visibility = self._get_visibility_for_condition(self.condition)
        self.wind_speed = random.uniform(0, 20)  # 0-20 m/s
        self.wind_direction = random.uniform(0, 360)  # 0-360도
        self.temperature = random.uniform(-10, 35)  # -10~35도
        self.pressure = random.uniform(1000, 1020)  # hPa
        
        # Safety risk factors
        self.landing_risk_multiplier = self._calculate_landing_risk()
        self.takeoff_risk_multiplier = self._calculate_takeoff_risk()
        
        # Weather forecast (2시간 예보)
        self.forecast = self._generate_forecast()
        
    def _get_visibility_for_condition(self, condition):
        """날씨 조건에 따른 시정 반환 (km)"""
        visibility_ranges = {
            WeatherCondition.CLEAR: (10, 50),
            WeatherCondition.LIGHT_FOG: (5, 10),
            WeatherCondition.MODERATE_FOG: (1, 5),
            WeatherCondition.HEAVY_FOG: (0.1, 1),
            WeatherCondition.RAIN: (2, 8),
            WeatherCondition.SNOW: (0.5, 3),
            WeatherCondition.STORM: (0.1, 2)
        }
        min_vis, max_vis = visibility_ranges[condition]
        return random.uniform(min_vis, max_vis)
    
    def _calculate_landing_risk(self):
        """착륙 위험도 계산"""
        base_risk = 1.0
        
        # 시정에 따른 위험도
        if self.visibility < 1:
            base_risk *= 3.0  # 짙은 안개
        elif self.visibility < 3:
            base_risk *= 2.0  # 중간 안개/눈
        elif self.visibility < 5:
            base_risk *= 1.5  # 연무/비
        
        # 바람에 따른 위험도
        if self.wind_speed > 15:
            base_risk *= 2.0  # 강풍
        elif self.wind_speed > 10:
            base_risk *= 1.3  # 중간 바람
        
        # 날씨 조건에 따른 추가 위험도
        if self.condition == WeatherCondition.STORM:
            base_risk *= 4.0
        elif self.condition == WeatherCondition.SNOW:
            base_risk *= 2.5
        elif self.condition == WeatherCondition.HEAVY_FOG:
            base_risk *= 2.0
            
        return base_risk
    
    def _calculate_takeoff_risk(self):
        """이륙 위험도 계산"""
        base_risk = 1.0
        
        # 시정에 따른 위험도 (이륙은 착륙보다 덜 민감)
        if self.visibility < 1:
            base_risk *= 2.0
        elif self.visibility < 3:
            base_risk *= 1.5
        elif self.visibility < 5:
            base_risk *= 1.2
        
        # 바람에 따른 위험도
        if self.wind_speed > 15:
            base_risk *= 1.8
        elif self.wind_speed > 10:
            base_risk *= 1.2
        
        # 날씨 조건에 따른 추가 위험도
        if self.condition == WeatherCondition.STORM:
            base_risk *= 3.0
        elif self.condition == WeatherCondition.SNOW:
            base_risk *= 2.0
        elif self.condition == WeatherCondition.HEAVY_FOG:
            base_risk *= 1.5
            
        return base_risk
    
    def update_weather(self, time_step):
        """시간에 따른 날씨 변화"""
        # 더 자연스러운 날씨 변화: 15-25분 간격으로 랜덤하게 변화
        if not hasattr(self, 'next_weather_change'):
            self.next_weather_change = time_step + random.randint(15, 25)
        
        if time_step >= self.next_weather_change:
            if random.random() < 0.4:  # 40% 확률로 날씨 변화
                self._random_weather_change()
            # 다음 변화 시간 설정
            self.next_weather_change = time_step + random.randint(15, 25)
        
        # 시정, 바람 등 세부 조건은 매분 약간씩 변화 (점진적 변화)
        self.visibility += random.uniform(-0.1, 0.1)  # 더 작은 변화
        self.visibility = max(0.1, min(50, self.visibility))  # 0.1-50km 범위 제한
        
        self.wind_speed += random.uniform(-0.5, 0.5)  # 더 작은 변화
        self.wind_speed = max(0, min(30, self.wind_speed))  # 0-30 m/s 범위 제한
        
        # 위험도 재계산
        self.landing_risk_multiplier = self._calculate_landing_risk()
        self.takeoff_risk_multiplier = self._calculate_takeoff_risk()
        
        # 예보는 30분마다 업데이트 (더 안정적인 예보)
        if time_step % 30 == 0:
            self.forecast = self._generate_forecast()
            debug(f"Weather forecast updated at {time_step}: {self.condition.value}, visibility: {self.visibility:.1f}km, wind: {self.wind_speed:.1f}m/s")
        else:
            debug(f"Weather update: {self.condition.value}, visibility: {self.visibility:.1f}km, wind: {self.wind_speed:.1f}m/s")
    
    def _generate_forecast(self):
        """2시간 예보 생성"""
        forecast = []
        current_condition = self.condition
        current_visibility = self.visibility
        current_wind = self.wind_speed
        
        for hour in range(1, 3):  # 1시간 후, 2시간 후
            # 예보 정확도 (시간이 지날수록 부정확하지만 더 현실적)
            accuracy = 1.0 - (hour * 0.15)  # 1시간 후 85%, 2시간 후 70% 정확도
            
            if random.random() < accuracy:
                # 현재 조건에서 점진적 변화 (더 현실적)
                forecast_condition = current_condition
                forecast_visibility = current_visibility + random.uniform(-1.0, 1.0)
                forecast_wind = current_wind + random.uniform(-3, 3)
            else:
                # 다른 조건으로 변화 (더 자연스러운 전환)
                forecast_condition = self._get_natural_transition(current_condition)
                forecast_visibility = self._get_visibility_for_condition(forecast_condition)
                forecast_wind = current_wind + random.uniform(-5, 5)
            
            # 범위 제한
            forecast_visibility = max(0.1, min(50, forecast_visibility))
            forecast_wind = max(0, min(30, forecast_wind))
            
            # 위험도 계산
            forecast_landing_risk = self._calculate_forecast_risk(forecast_condition, forecast_visibility, forecast_wind, "landing")
            forecast_takeoff_risk = self._calculate_forecast_risk(forecast_condition, forecast_visibility, forecast_wind, "takeoff")
            
            forecast.append({
                "hour": hour,
                "condition": forecast_condition.value,
                "visibility": round(forecast_visibility, 1),
                "wind_speed": round(forecast_wind, 1),
                "landing_risk": round(forecast_landing_risk, 2),
                "takeoff_risk": round(forecast_takeoff_risk, 2)
            })
            
            # 다음 시간을 위한 현재 조건 업데이트
            current_condition = forecast_condition
            current_visibility = forecast_visibility
            current_wind = forecast_wind
        
        return forecast
    
    def _get_natural_transition(self, current_condition):
        """현재 조건에서 자연스러운 전환 (현실적인 확률)"""
        # 각 조건별 전환 확률 (가중치 기반)
        transition_weights = {
            WeatherCondition.CLEAR: {
                WeatherCondition.CLEAR: 70,      # 맑음 유지 (70%)
                WeatherCondition.LIGHT_FOG: 20,  # 연무로 변화 (20%)
                WeatherCondition.RAIN: 10        # 비로 변화 (10%)
            },
            WeatherCondition.LIGHT_FOG: {
                WeatherCondition.CLEAR: 60,      # 맑음으로 회복 (60%)
                WeatherCondition.LIGHT_FOG: 25,  # 연무 유지 (25%)
                WeatherCondition.MODERATE_FOG: 15  # 중간 안개로 악화 (15%)
            },
            WeatherCondition.MODERATE_FOG: {
                WeatherCondition.LIGHT_FOG: 50,  # 연무로 개선 (50%)
                WeatherCondition.MODERATE_FOG: 30,  # 중간 안개 유지 (30%)
                WeatherCondition.HEAVY_FOG: 20   # 짙은 안개로 악화 (20%)
            },
            WeatherCondition.HEAVY_FOG: {
                WeatherCondition.MODERATE_FOG: 60,  # 중간 안개로 개선 (60%)
                WeatherCondition.HEAVY_FOG: 25,     # 짙은 안개 유지 (25%)
                WeatherCondition.CLEAR: 15          # 맑음으로 급격한 개선 (15%)
            },
            WeatherCondition.RAIN: {
                WeatherCondition.CLEAR: 50,      # 맑음으로 개선 (50%)
                WeatherCondition.RAIN: 35,       # 비 유지 (35%)
                WeatherCondition.STORM: 15       # 폭풍으로 악화 (15%)
            },
            WeatherCondition.SNOW: {
                WeatherCondition.CLEAR: 60,      # 맑음으로 개선 (60%)
                WeatherCondition.SNOW: 25,       # 눈 유지 (25%)
                WeatherCondition.STORM: 15       # 폭풍으로 악화 (15%)
            },
            WeatherCondition.STORM: {
                WeatherCondition.RAIN: 70,       # 비로 개선 (70%)
                WeatherCondition.STORM: 20,      # 폭풍 유지 (20%)
                WeatherCondition.CLEAR: 10       # 맑음으로 급격한 개선 (10%)
            }
        }
        
        weights = transition_weights.get(current_condition, {WeatherCondition.CLEAR: 100})
        conditions = list(weights.keys())
        weight_values = list(weights.values())
        
        return random.choices(conditions, weights=weight_values)[0]
    
    def _get_random_condition(self):
        """랜덤 날씨 조건 반환"""
        return random.choice(list(WeatherCondition))
    
    def _calculate_forecast_risk(self, condition, visibility, wind_speed, operation_type):
        """예보 조건에 따른 위험도 계산"""
        base_risk = 1.0
        
        # 시정에 따른 위험도
        if visibility < 1:
            base_risk *= 3.0 if operation_type == "landing" else 2.0
        elif visibility < 3:
            base_risk *= 2.0 if operation_type == "landing" else 1.5
        elif visibility < 5:
            base_risk *= 1.5 if operation_type == "landing" else 1.2
        
        # 바람에 따른 위험도
        if wind_speed > 15:
            base_risk *= 2.0 if operation_type == "landing" else 1.8
        elif wind_speed > 10:
            base_risk *= 1.3 if operation_type == "landing" else 1.2
        
        # 날씨 조건에 따른 추가 위험도
        if condition == WeatherCondition.STORM:
            base_risk *= 4.0 if operation_type == "landing" else 3.0
        elif condition == WeatherCondition.SNOW:
            base_risk *= 2.5 if operation_type == "landing" else 2.0
        elif condition == WeatherCondition.HEAVY_FOG:
            base_risk *= 2.0 if operation_type == "landing" else 1.5
            
        return base_risk
    
    def _random_weather_change(self):
        """랜덤 날씨 변화"""
        weather_transitions = {
            WeatherCondition.CLEAR: [WeatherCondition.LIGHT_FOG, WeatherCondition.RAIN],
            WeatherCondition.LIGHT_FOG: [WeatherCondition.CLEAR, WeatherCondition.MODERATE_FOG],
            WeatherCondition.MODERATE_FOG: [WeatherCondition.LIGHT_FOG, WeatherCondition.HEAVY_FOG],
            WeatherCondition.HEAVY_FOG: [WeatherCondition.MODERATE_FOG, WeatherCondition.CLEAR],
            WeatherCondition.RAIN: [WeatherCondition.CLEAR, WeatherCondition.STORM],
            WeatherCondition.SNOW: [WeatherCondition.CLEAR, WeatherCondition.STORM],
            WeatherCondition.STORM: [WeatherCondition.RAIN, WeatherCondition.CLEAR]
        }
        
        possible_transitions = weather_transitions.get(self.condition, [WeatherCondition.CLEAR])
        self.condition = random.choice(possible_transitions)
        self.visibility = self._get_visibility_for_condition(self.condition)
        
        debug(f"Weather condition changed to: {self.condition.value}")
    
    def get_weather_info(self):
        """현재 날씨 정보 반환 (Agent용)"""
        return {
            "landing_risk": round(self.landing_risk_multiplier, 2),
            "takeoff_risk": round(self.takeoff_risk_multiplier, 2),
            "forecast": self.forecast
        }
    
    def get_detailed_weather_info(self):
        """상세 날씨 정보 반환 (프론트엔드용)"""
        return {
            "condition": self.condition.value,
            "visibility": round(self.visibility, 1),
            "wind_speed": round(self.wind_speed, 1),
            "wind_direction": round(self.wind_direction, 1),
            "temperature": round(self.temperature, 1),
            "pressure": round(self.pressure, 1),
            "landing_risk": round(self.landing_risk_multiplier, 2),
            "takeoff_risk": round(self.takeoff_risk_multiplier, 2),
            "forecast": self.forecast
        } 