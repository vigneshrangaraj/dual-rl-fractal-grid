# env/secondary/inverter.py
import numpy as np


class Inverter:
    def __init__(self, config, inverter_id=0, mg_id=0):
        self.inverter_id = inverter_id
        self.V_nom = getattr(config, "V_nom", 1.0)
        self.V = self.V_nom  # initial voltage in per unit
        self.delta = 0.0  # phase angle (radians), starting at 0
        self.i_d = 0.0
        self.i_q = 0.0
        self.mg_id = mg_id  # microgrid ID

        self.droop_coefficient = getattr(config, "droop_coefficient", 10.0)
        self.tau = getattr(config, "inverter_time_constant", 0.1)
        self.noise_std = getattr(config, "secondary_noise_std", 0.002)
        
        # Add voltage control parameters
        self.k_p = getattr(config, "voltage_proportional_gain", 0.1)  # Proportional gain
        self.k_i = getattr(config, "voltage_integral_gain", 0.01)    # Integral gain
        self.error_integral = 0.0  # Integral of voltage error

        # Simulation parameters
        self.frequency = 50.0  # Hz
        self.omega = 2 * np.pi * self.frequency
        self.t = 0.0  # simulation time
        self.dt = 1e-3  # simulation timestep in seconds

        self.measured_voltage = self.V

    def reset(self):
        self.V = self.V_nom
        self.delta = 0.0
        self.i_d = 0.0
        self.i_q = 0.0
        self.t = 0.0
        self.error_integral = 0.0

    def simulate_phase_currents(self):
        Ia = self.V * np.sin(self.omega * self.t + self.delta)
        Ib = self.V * np.sin(self.omega * self.t + self.delta - 2 * np.pi / 3)
        Ic = self.V * np.sin(self.omega * self.t + self.delta + 2 * np.pi / 3)
        return Ia, Ib, Ic

    def update(self, V_ref, measured_voltage):
        # Calculate voltage error
        error = V_ref - measured_voltage
        
        # Update integral of error
        self.error_integral += error * self.dt
        
        # Calculate voltage correction using PI control
        voltage_correction = (self.k_p * error + 
                            self.k_i * self.error_integral)
        
        # Add noise
        noise = np.random.normal(0, self.noise_std)
        
        # Update voltage with bounded change
        max_voltage_change = 0.05  # Maximum voltage change per step
        voltage_change = voltage_correction + noise
        voltage_change = np.clip(voltage_change, -max_voltage_change, max_voltage_change)
        
        self.V = measured_voltage + voltage_change
        
        # Update phase angle using droop control
        phase_correction = error * self.dt / self.droop_coefficient
        self.delta += phase_correction
        
        # Normalize phase angle to [-pi, pi]
        self.delta = np.arctan2(np.sin(self.delta), np.cos(self.delta))

        # Advance simulation time
        self.t += self.dt

        # Simulate 3-phase currents and apply Park transformation
        Ia, Ib, Ic = self.simulate_phase_currents()
        self.i_d, self.i_q = self.park_transform(Ia, Ib, Ic, self.delta)

        return self.get_state()

    @staticmethod
    def park_transform(Ia, Ib, Ic, delta):
        i_d = (2 / 3) * (Ia * np.cos(delta) +
                         Ib * np.cos(delta - 2 * np.pi / 3) +
                         Ic * np.cos(delta + 2 * np.pi / 3))
        i_q = -(2 / 3) * (Ia * np.sin(delta) +
                          Ib * np.sin(delta - 2 * np.pi / 3) +
                          Ic * np.sin(delta + 2 * np.pi / 3))
        return i_d, i_q

    def get_state(self):
        """Return current inverter state."""
        return {
            "inverter_id": self.inverter_id,
            "V": self.V,
            "delta": self.delta,
            "i_d": self.i_d,
            "i_q": self.i_q
        }


# --- Testing the Inverter module ---
if __name__ == "__main__":
    class Config:
        V_nom = 1.0
        droop_coefficient = 10.0
        inverter_time_constant = 0.1
        secondary_noise_std = 0.002

    config = Config()
    inv = Inverter(config, inverter_id=0)
    print("Initial Inverter State:", inv.get_state())
    for _ in range(10):
        state = inv.update(V_ref=1.02, measured_voltage=inv.V)
        print("Updated Inverter State:", state)