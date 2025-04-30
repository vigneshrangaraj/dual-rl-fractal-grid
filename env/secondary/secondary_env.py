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
        self.consensus_penalty = getattr(config, "consensus_penalty", 0.005)
        self.noise_std = getattr(config, "secondary_noise_std", 0.002)
        self.max_steps = getattr(config, "secondary_max_steps", 200)
        self.adjacency_matrix = getattr(config, "adjacency_matrix", self.default_ring_topology(self.num_agents * self.num_microgrids))
        self.num_microgrids = getattr(config, "num_microgrids", 1)
        
        # Add voltage limits
        self.V_min = getattr(config, "V_min", 0.5)  # Minimum allowed voltage
        self.V_max = getattr(config, "V_max", 2.0)  # Maximum allowed voltage
        self.max_voltage_change = getattr(config, "max_voltage_change", 0.05)  # Maximum voltage change per step

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

    def set_measured_voltage(self, microgrid_index, voltage, agent_index):
        for i in range(self.num_agents):
            if self.inverters[i].mg_id == microgrid_index:
                self.states[i]["voltage"] = voltage
                break

    def step(self, actions):
        next_states = []
        rewards = []

        for i, inverter in enumerate(self.inverters):
            inverter = self.inverters[i]
            measured_voltage = inverter.measured_voltage

            # --- Consensus voltage correction ---
            neighbor_voltages = CommunicationModule.get_neighbor_voltages(i, self.states, self.adjacency_matrix)
            consensus_error = sum([(measured_voltage - vj) for vj in neighbor_voltages])
            
            # Normalize consensus error to prevent large corrections
            consensus_error = np.clip(consensus_error, -0.1, 0.1)

            # --- Apply action (reactive power adjustment) ---
            action = actions[i]
            # Clip action to reasonable range
            action = np.clip(action, -1.0, 1.0)
            
            # --- Update voltage reference with proper constraints ---
            V_ref = self.V_ref + self.voltage_gain * action
            
            # Add consensus correction with proper scaling
            V_ref = V_ref + 0.1 * consensus_error

            # --- Update inverter state using inverter dynamics ---
            new_inverter_state = inverter.update(V_ref=V_ref, measured_voltage=measured_voltage)
            new_voltage = new_inverter_state["V"]
            
            # Ensure voltage change is bounded
            voltage_change = new_voltage - measured_voltage
            if abs(voltage_change) > self.max_voltage_change:
                new_voltage = measured_voltage + np.sign(voltage_change) * self.max_voltage_change

            # --- Construct state dict ---
            new_state = {
                "voltage": new_voltage,
                "reactive_power": action,
                "i_d": new_inverter_state["i_d"],
                "i_q": new_inverter_state["i_q"],
                "delta": new_inverter_state["delta"]
            }
            next_states.append(new_state)

            # --- Compute voltage reward with stronger penalties for voltage violations ---
            v_i = new_voltage
            if 0.98 <= v_i <= 1.02:
                reward = 10.0  # Strong positive reward for being very close to nominal
            elif 0.95 <= v_i < 0.98 or 1.02 < v_i <= 1.05:
                reward = 0.2 - abs(1.0 - v_i)
            elif 0.9 <= v_i < 0.95 or 1.05 < v_i <= 1.1:
                reward = -2.0 * abs(1.0 - v_i)
            else:
                reward = -10.0

            # Add action penalty to discourage large control actions
            reward -= self.action_penalty * (action ** 2)
            
            # Add consensus penalty to encourage voltage coordination
            reward -= self.consensus_penalty * (consensus_error ** 2)

            rewards.append(reward)

        self.states = next_states
        self.time_step += 1
        done = self.time_step >= self.max_steps or all(
            state["voltage"] < self.V_min or state["voltage"] > self.V_max for state in self.states
        )
        info = {}

        # log progress
        print(f"Secondary step {self.time_step}:")

        return self.states, rewards, done, info

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