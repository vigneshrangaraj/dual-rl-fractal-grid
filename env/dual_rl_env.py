# env/dual_rl_env.py
"""
Dual RL Environment Wrapper

This module integrates the tertiary and secondary environments.
It performs the following:
  1. Executes a tertiary step based on a macro-action.
  2. Runs multiple secondary steps to simulate local voltage control.
  3. Aggregates rewards from both layers to form an overall reward.
"""

from env.tertiary.fractal_grid_env import FractalGridEnv
from env.secondary.secondary_env import SecondaryEnv  # Ensure this module exists

class DualRLEnv:
    def __init__(self, config):
        """
            config: A configuration object/dictionary containing simulation parameters.
                    Expected parameters include:
                      - num_secondary_steps: Number of secondary steps per tertiary step.
                      - alpha_sec: Weight for secondary rewards in overall reward calculation.
        """
        self.config = config
        self.tertiary_env = FractalGridEnv(config)
        self.secondary_env = SecondaryEnv(config)
        self.num_secondary_steps = getattr(config, "num_secondary_steps", 5)

    def reset(self):
        tertiary_state = self.tertiary_env.reset()
        secondary_state = self.secondary_env.reset()
        return {"tertiary": tertiary_state, "secondary": secondary_state}

    def step(self, tertiary_action, secondary_actions):
        """
            next_state (dict): Combined state from tertiary and secondary environments.
            rewards (dict): Contains separate rewards for tertiary and secondary layers and an overall reward.
            done (bool): Flag indicating if the episode is finished.
            info (dict): Additional information from the tertiary environment.
        """
        # Tertiary step: update grid topology, run power flow, and compute tertiary reward.
        next_ter_state, tertiary_reward, done, info = self.tertiary_env.step(tertiary_action)

        # Execute multiple secondary steps to simulate local voltage control.
        secondary_rewards = []
        secondary_states = []

        # Get the measured volt at each microgrid and apply that to all the inverters in the secondary env.
        for i in range(self.config.num_microgrids):
            measured_voltage = next_ter_state["microgrids"][i]["measured_voltage"]
            self.secondary_env.set_measured_voltage(i, measured_voltage)
        for _ in range(self.num_secondary_steps):
            next_sec_state, sec_reward, done_sec, sec_info = self.secondary_env.step(secondary_actions)
            secondary_rewards.append(sec_reward)
            secondary_states.append(next_sec_state)
            if done_sec:  # Optionally, break early if secondary environment signals done.
                break

        # Aggregate secondary reward (e.g., average over secondary steps).
        aggregated_sec_reward = (sum(secondary_rewards) / len(secondary_rewards)
                                 if secondary_rewards else 0.0)

        # Compute overall reward: combine tertiary and weighted secondary rewards.
        overall_reward = tertiary_reward + self.config.alpha_sec * aggregated_sec_reward

        # Combine states from both environments.
        next_state = {"tertiary": next_ter_state, "secondary": secondary_states}
        rewards = {"tertiary": tertiary_reward,
                   "secondary": aggregated_sec_reward,
                   "overall": overall_reward}

        return next_state, rewards, done, info

# --- Testing the DualRLEnv independently ---
if __name__ == "__main__":
    class DummyConfig:
        num_microgrids = 2
        tie_lines = [(0, 1, 1)]
        V_ref = 1.0
        alpha_sec = 0.5  # Weight for secondary reward
        num_secondary_steps = 3

    class DummySecondaryEnv:
        def __init__(self, config):
            self.config = config
            self.step_count = 0

        def reset(self):
            self.step_count = 0
            return {"dummy_secondary_state": True}

        def step(self):
            self.step_count += 1
            sec_reward = 0.1  # Dummy reward
            state = {"dummy_secondary_state": self.step_count}
            done = self.step_count >= 3
            info = {}
            return state, sec_reward, done, info

    dummy_config = DummyConfig()
    dual_env = DualRLEnv(dummy_config)
    from env.tertiary.fractal_grid_env import FractalGridEnv  # Ensure this exists or create a dummy version.
    dual_env.secondary_env = DummySecondaryEnv(dummy_config)

    state = dual_env.reset()
    print("Initial state:", state)
    dummy_action = {"dispatch": [0.0, 0.0], "tie_lines": [(0, 1, 1)]}
    next_state, rewards, done, info = dual_env.step(dummy_action)
    print("Next state:", next_state)
    print("Rewards:", rewards)
    print("Done:", done)
    print("Info:", info)