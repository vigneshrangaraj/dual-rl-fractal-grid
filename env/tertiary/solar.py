# env/tertiary/solar.py
import numpy as np


class Solar:
    def __init__(self, config, solar_id=0):
        self.solar_id = solar_id
        self.config = config
        self.base_output = getattr(config, 'solar_base_output', 0.005)  # MW (default)
        self.variability = getattr(config, 'solar_variability', 0.1)
        self.current_output = self.base_output
        self.time_step = 0

        # Optional daily profile function: expects a function of time_step returning a multiplier
        self.daily_profile = getattr(config, 'solar_daily_profile', None)

        if hasattr(config, 'seed'):
            np.random.seed(config.seed)

    def reset(self):
        self.current_output = self.base_output
        self.time_step = 0

    def update(self):
        # Use a diurnal profile if provided, else apply random variability.
        if self.daily_profile:
            # daily_profile is a function of time_step, e.g., a sinusoid to mimic daylight.
            profile_multiplier = self.daily_profile(self.time_step)
        else:
            # Without a diurnal profile, use random variation within ±variability.
            profile_multiplier = 1.0 + np.random.uniform(-self.variability, self.variability)

        self.current_output = self.base_output * profile_multiplier
        self.time_step += 1
        return self.current_output

    def get_current_output(self):
        return self.current_output

    def get_state(self):
        return {
            'solar_id': self.solar_id,
            'current_output': self.current_output,
            'time_step': self.time_step
        }


if __name__ == "__main__":
    # Example usage and simple test scenario.
    class Config:
        solar_base_output = 0.005  # MW
        solar_variability = 0.1  # ±10% variability
        seed = 42

        # Example diurnal profile: sinusoidal function peaking at noon.
        @staticmethod
        def solar_daily_profile(time_step):
            # For simplicity, assume one cycle every 24 steps.
            return 0.5 * (1 + np.sin(2 * np.pi * (time_step % 24) / 24))


    config = Config()
    solar_unit = Solar(config, solar_id=1)
    solar_unit.reset()
    print("Initial Solar State:", solar_unit.get_state())
    for _ in range(5):
        output = solar_unit.update()
        print("Updated Solar Output (MW):", output)
        print("State:", solar_unit.get_state())