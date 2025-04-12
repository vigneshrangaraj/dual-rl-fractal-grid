# env/tertiary/fractal_grid_env.py

import numpy as np
import logging

from env.tertiary.microgrid import MicroGrid
from env.tertiary.panda_power_wrapper import PandaPowerWrapper


class FractalGridEnv:
    def __init__(self, config):
        """
        Initialize the fractal grid environment.

        Args:
            config: Configuration object/dict with parameters including:
                    - num_microgrids: Number of microgrids in the fractal grid.
                    - tie_lines: List of tie-line connections, e.g. [(mg1, mg2, status), ...].
                    - V_ref: Reference voltage.
                    - max_steps: Maximum simulation steps per episode.
                    - lambda_econ, alpha_sec, beta_volt: Reward weighting factors.
        """
        self.config = config
        self.num_microgrids = getattr(config, "num_microgrids", 1)
        self.tie_lines = getattr(config, "tie_lines", [])
        self.V_ref = getattr(config, "V_ref", 1.0)
        self.max_steps = getattr(config, "max_steps", 100)
        self.lambda_econ = getattr(config, "lambda_econ", 1.0)
        self.alpha_sec = getattr(config, "alpha_sec", 1.0)
        self.beta_volt = getattr(config, "beta_volt", 1.0)

        # Create microgrid instances
        self.microgrids = [MicroGrid(config, mg_id=i) for i in range(self.num_microgrids)]

        # Initialize PandaPower wrapper for AC power flow computations.
        self.pw_wrapper = PandaPowerWrapper(config)
        self.current_step = 0
        self.done = False

    def reset(self):
        self.current_step = 0
        self.done = False

        for mg in self.microgrids:
            mg.reset()
        # Update the power flow wrapper with current microgrids and tie-lines.
        self.pw_wrapper.reset_network(self.microgrids, self.tie_lines)
        return self._get_state()

    def step(self, tertiary_action):
        # Apply dispatch commands to each microgrid.
        dispatch = tertiary_action.get("dispatch", None)
        if dispatch:
            for i, mg in enumerate(self.microgrids):
                mg.apply_dispatch(dispatch[i])

        # Update tie-line statuses if provided.
        new_tie_lines = tertiary_action.get("tie_lines", None)
        if new_tie_lines is not None:
            self.tie_lines = new_tie_lines

        # Run AC power flow on each microgrid and adjust for tie-line connections.
        power_flow_results = self.pw_wrapper.run_power_flow(self.microgrids, self.tie_lines)

        # Update each microgrid's state using the power flow results.
        for mg in self.microgrids:
            mg.update_state_from_power_flow(power_flow_results.get(mg.mg_id, self.V_ref))

        # Compute the economic cost across microgrids.
        econ_cost = self._calculate_economic_cost()
        # Compute the voltage penalty (e.g., mean squared deviation from V_ref).
        voltage_penalty = self._calculate_voltage_penalty()

        # Compute the overall reward.
        reward = (-self.lambda_econ * econ_cost +
                  self.beta_volt * voltage_penalty)

        self.current_step += 1
        if self.current_step >= self.max_steps:
            self.done = True

        next_state = self._get_state()
        info = {
            "power_flow": power_flow_results,
            "econ_cost": econ_cost,
            "voltage_penalty": voltage_penalty
        }
        return next_state, reward, self.done, info

    def _get_state(self):
        mg_states = [mg.get_state() for mg in self.microgrids]

        state = {
            "microgrids": mg_states,
            "timestep": self.current_step
        }
        return state

    def _calculate_economic_cost(self):
        cost = sum(mg.get_generation_cost() for mg in self.microgrids)
        return cost

    def _calculate_voltage_penalty(self):
        penalty = sum(mg.get_voltage_deviation() for mg in self.microgrids)
        return penalty

# --- Testing the FractalGridEnv independently ---
if __name__ == "__main__":
    class Config:
        num_microgrids = 2
        tie_lines = [(0, 1, 1)]
        V_ref = 1.0
        max_steps = 10
        lambda_econ = 1.0
        alpha_sec = 0.5
        beta_volt = 1.0

        # Load parameters
        base_load = 50.0
        load_variability = 0.1
        load_cost_factor = 0.05

        # Solar parameters
        num_solar = 2
        solar_buses = [5, 10]
        solar_base_output = 0.005

        # Wind parameters
        num_wind = 1
        wind_buses = [15]
        wind_base_output = 0.007

        # BESS parameters
        bess_capacity = 100.0
        bess_max_charge = 20.0
        bess_max_discharge = 20.0
        bess_charge_efficiency = 0.95
        bess_discharge_efficiency = 0.95
        bess_initial_soc = 0.5

        seed = 42


    config = Config()
    env = FractalGridEnv(config)
    state = env.reset()
    print("Initial state:")
    print(state)
    # Define a dummy tertiary action
    tertiary_action = {
        "dispatch": [10.0, -5.0],  # Dispatch commands for each microgrid.
        "tie_lines": [(0, 1, 1)]
    }
    next_state, reward, done, info = env.step(tertiary_action)
    print("\nNext state:")
    print(next_state)
    print("\nReward:", reward)
    print("\nDone:", done)
    print("\nInfo:", info)