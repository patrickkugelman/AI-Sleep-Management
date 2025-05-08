import numpy as np
import pandas as pd

class SleepEnvironmentSimulator:
    """
    Simulate sleep data in harsh environments with various stressors.
    """
    def __init__(self, num_samples=1000):
        self.num_samples = num_samples
        
    def generate_dataset(self, advanced_simulation=True):
        """Generate advanced synthetic sleep monitoring dataset with complex environmental interactions.
        
        Columns:
        - noise_level: Decibel levels (0-100)
        - temperature: Celsius (-10 to 50)
        - light_intensity: Lux (0-10000)
        - air_quality_index: AQI (0-500)
        - humidity: Relative humidity (%)
        - circadian_disruption: Circadian rhythm disturbance score
        - sleep_quality_score: 0-100 scale
        - cognitive_performance: 0-100 scale
        - physical_performance: 0-100 scale
        """
        np.random.seed(42)
        
        # Advanced environmental stressor generation
        def generate_correlated_features(base_mean, base_std, correlation_factor=0.3):
            base = np.random.normal(base_mean, base_std, self.num_samples)
            correlated_noise = np.random.normal(0, base_std * correlation_factor, self.num_samples)
            return base + correlated_noise
        
        # Generate correlated environmental features
        noise_level = generate_correlated_features(60, 20)
        temperature = generate_correlated_features(20, 10)
        light_intensity = generate_correlated_features(500, 300)
        air_quality_index = generate_correlated_features(50, 50)
        humidity = generate_correlated_features(50, 15)
        
        # Simulate circadian disruption
        time_of_day = np.random.uniform(0, 24, self.num_samples)
        circadian_disruption = np.abs(time_of_day - 12) / 12 * 100
        
        if advanced_simulation:
            # Complex non-linear interactions with advanced feature engineering
            sleep_quality_score = (
                100 
                - 0.6 * np.abs(noise_level - 50)**1.2  # Non-linear noise impact
                - 0.4 * np.abs(temperature - 20)**1.1  # Non-linear temperature impact
                - 0.3 * np.abs(light_intensity - 100)**1.3  # Non-linear light impact
                - 0.2 * air_quality_index
                - 0.1 * np.abs(humidity - 50)
                - 0.2 * circadian_disruption
            ).clip(0, 100)
        else:
            # Original linear simulation
            sleep_quality_score = (
                100 - 0.5 * np.abs(noise_level - 50) 
                - 0.3 * np.abs(temperature - 20) 
                - 0.2 * np.abs(light_intensity - 100) 
                - 0.1 * air_quality_index
            ).clip(0, 100)
        
        # Advanced performance metrics with more complex relationships
        cognitive_performance = (
            sleep_quality_score * 0.8 
            - 0.2 * circadian_disruption
        ).clip(0, 100)
        
        physical_performance = (
            sleep_quality_score * 0.7 
            - 0.15 * circadian_disruption
        ).clip(0, 100)
        
        # Create DataFrame
        df = pd.DataFrame({
            'noise_level': noise_level,
            'temperature': temperature,
            'light_intensity': light_intensity,
            'air_quality_index': air_quality_index,
            'sleep_quality_score': sleep_quality_score,
            'cognitive_performance': cognitive_performance,
            'physical_performance': physical_performance
        })
        
        return df
    
    def save_dataset(self, filepath='sleep_environment_data.csv'):
        """Save generated dataset to CSV"""
        df = self.generate_dataset()
        df.to_csv(filepath, index=False)
        print(f"Dataset saved to {filepath}")

if __name__ == "__main__":
    simulator = SleepEnvironmentSimulator(num_samples=5000)
    simulator.save_dataset()
