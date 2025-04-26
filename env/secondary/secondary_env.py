import numpy as np
from env.secondary.comm import CommunicationModule
from env.secondary.inverter import Inverter


class SecondaryEnv:
    def __init__(self, config):
        self.num_agents = config.num_secondary_agents
        self.num_microgrids = getattr(config, "num_microgrids", 1)
        self.V_ref = getattr(config, "V_ref", 1.0)
        self.voltage_gain = getattr(config, "voltage_gain", 0.05)
        self.action_penalty = getattr(config, "action_penalty", 0.1)
        self.consensus_penalty = getattr(config, "consensus_penalty", 0.05)
        self.noise_std = getattr(config, "secondary_noise_std", 0.002)
        self.max_steps = getattr(config, "secondary_max_steps", 200)
        self.adjacency_matrix = getattr(config, "adjacency_matrix", self.default_ring_topology(self.num_agents * self.num_microgrids))
        self.num_microgrids = getattr(config, "num_microgrids", 1)

        # loop through microgrids and then num_agents to properly identify the inverter and the microgrid
        # it belongs to
        self.inverters = []
        for i in range(self.num_microgrids):
            for j in range(self.num_agents):
                inverter = Inverter(config, j, i)
                self.inverters.append(inverter)

        self.reset()

    def reset(self):
        self.states = []
        for i in range(len(self.inverters)):
            inverter = self.inverters[i]
            state = {
                "voltage": np.random.uniform(0.95, 1.05),
                "reactive_power": 0.0,
                "i_d": inverter.i_d,
                "i_q": inverter.i_q,
                "delta": inverter.delta
            }
            self.states.append(state)

        self.time_step = 0
        return self.states

    def set_measured_voltage(self, microgrid_index, voltage):
        for i in range(self.num_agents):
            if self.inverters[i].mg_id == microgrid_index:
                self.states[i]["voltage"] = voltage
                break

    def step(self, actions):
        next_states = []
        rewards = []

        for i, inverter in enumerate(self.inverters):
            inverter = self.inverters[i]
            current_state = self.states[i]
            measured_voltage = current_state["voltage"]

            # --- Consensus voltage correction ---
            neighbor_voltages = CommunicationModule.get_neighbor_voltages(i, self.states, self.adjacency_matrix)
            consensus_error = sum([(measured_voltage - vj) for vj in neighbor_voltages])

            # --- Update voltage reference like feedback linearization ---
            V_ref = measured_voltage * consensus_error

            # --- Apply action (reactive power adjustment) ---
            action = actions[i]
            inverter_action_voltage = V_ref + self.voltage_gain * action

            # --- Update inverter state using inverter dynamics ---
            new_inverter_state = inverter.update(V_ref=inverter_action_voltage, measured_voltage=measured_voltage)
            new_voltage = new_inverter_state["V"]

            # --- Construct state dict ---
            new_state = {
                "voltage": new_voltage,
                "reactive_power": action,
                "i_d": new_inverter_state["i_d"],
                "i_q": new_inverter_state["i_q"],
                "delta": new_inverter_state["delta"]
            }
            next_states.append(new_state)

            # --- Compute voltage reward zone ---
            v_i = new_voltage
            if 0.95 <= v_i <= 1.05:
                reward = 0.05 - abs(1.0 - v_i)
            elif 0.8 <= v_i < 0.95 or 1.05 < v_i <= 1.25:
                reward = -abs(1.0 - v_i)
            else:
                reward = -10.0

            # Action penalty (optional)
            reward -= self.action_penalty * abs(action)

            # Consensus penalty (optional)
            consensus_penalty = sum([(v_i - vj) ** 2 for vj in neighbor_voltages])
            reward -= self.consensus_penalty * consensus_penalty

            rewards.append(reward)

        self.states = next_states
        self.time_step += 1
        done = self.time_step >= self.max_steps
        info = {}

        return next_states, rewards, done, info

    def default_ring_topology(self, n):
        """
        Ring topology for n agents.
        Each agent connected to two neighbors.
        """
        adj = np.zeros((n, n))
        for i in range(n):
            adj[i][(i - 1) % n] = 1
            adj[i][(i + 1) % n] = 1
        return adj



# --- Testing the updated SecondaryEnv with inverter dynamics ---
if __name__ == "__main__":
    class Config:
        num_secondary_agents = 3
        V_ref = 1.0
        secondary_noise_std = 0.002
        secondary_max_steps = 10
        action_penalty = 0.001
        comm_sigma = 1.0
        comm_threshold = 5.0
        droop_coefficient = 10.0
        inverter_time_constant = 0.1
        V_nom = 1.0


    config = Config()
    env = SecondaryEnv(config)
    state = env.reset()
    print("Initial secondary state:")
    print(state)

    # Run a few steps with random desired voltage references.
    for t in range(5):
        actions = np.random.uniform(0.98, 1.02, env.num_agents).tolist()
        next_state, rewards, done, info = env.step(actions)
        print(f"\nStep {t + 1}:")
        print("Actions (desired voltages):", actions)
        print("Next State:", next_state)
        print("Rewards:", rewards)
        print("Done:", done)
        if done:
            break