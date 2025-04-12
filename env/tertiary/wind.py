# env/tertiary/wind.py
import numpy as np


class Wind:
    def __init__(self, config, wind_id=0):
        self.wind_id = wind_id
        self.config = config
        self.base_output = getattr(config, 'wind_base_output', 0.007)  # MW (default)
        self.variability = getattr(config, 'wind_variability', 0.2)  # ±20% variability as default
        self.current_output = self.base_output
        self.time_step = 0

        # Optionally, a wind profile function to simulate temporal fluctuations (e.g., gustiness).
        self.wind_profile = getattr(config, 'wind_profile', None)

        if hasattr(config, 'seed'):
            np.random.seed(config.seed)

    def reset(self):
        self.current_output = self.base_output
        self.time_step = 0

    def update(self):
        if self.wind_profile:
            # Use the provided wind_profile function (e.g., periodic or stochastic profile)
            profile_multiplier = self.wind_profile(self.time_step)
        else:
            # Without a profile, apply random variation within ±variability.
            profile_multiplier = 1.0 + np.random.uniform(-self.variability, self.variability)

        self.current_output = self.base_output * profile_multiplier
        self.time_step += 1
        return self.current_output

    def get_current_output(self):
        return self.current_output

    def get_state(self):
        return {
            'wind_id': self.wind_id,
            'current_output': self.current_output,
            'time_step': self.time_step
        }


if __name__ == "__main__":
    # Example usage and simple test scenario.
    class Config:
        wind_base_output = 0.007  # MW
        wind_variability = 0.2  # ±20% variability
        seed = 42

        # Example wind profile function: simulate a periodic pattern to mimic gustiness.
        @staticmethod
        def wind_profile(time_step):
            # For simplicity, use a sine function with a period of 12 time steps.
            return 0.8 + 0.2 * np.sin(2 * np.pi * (time_step % 12) / 12)


    config = Config()
    wind_unit = Wind(config, wind_id=1)
    wind_unit.reset()
    print("Initial Wind State:", wind_unit.get_state())
    for _ in range(5):
        output = wind_unit.update()
        print("Updated Wind Output (MW):", output)
        print("State:", wind_unit.get_state())