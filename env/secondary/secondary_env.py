# env/secondary/secondary_env.py
import numpy as np
from env.secondary.der import DER
from env.secondary.inverter import Inverter
from env.secondary.comm import CommunicationModule


class SecondaryEnv:
    def __init__(self, config):
        """
        Initialize the secondary environment with inverter dynamics and per-DER communication.

        Args:
            config: Configuration object/dictionary with parameters including:
                - num_secondary_agents (int): Number of DER agents.
                - V_ref (float): Global reference voltage (pu).
                - secondary_noise_std (float): Noise standard deviation for inverter voltage update.
                - secondary_max_steps (int): Maximum steps per episode.
                - action_penalty (float): Penalty factor for control effort.
                - comm_sigma (float): Sigma for the Gaussian communication kernel.
                - comm_threshold (float, optional): Distance threshold for communication.
                - positions (np.array, optional): Array of shape (N, D) with DER positions.
        """
        self.config = config
        self.num_agents = getattr(config, "num_secondary_agents", 5)
        self.V_ref = getattr(config, "V_ref", 1.0)
        self.noise_std = getattr(config, "secondary_noise_std", 0.002)
        self.max_steps = getattr(config, "secondary_max_steps", 50)
        self.action_penalty = getattr(config, "action_penalty", 0.001)
        self.time_step = 0

        # Instantiate DER agents (for communication)

        self.num_microgrids = getattr(config, "num_microgrids", 1)

        # Loop through each microgrids. We are setting 5 agents per microgrid.
        for i in range(self.num_microgrids):
            for j in range(self.num_agents):
                self.der_agents = [DER(config, der_id=j, mg_id=i) for i in range(self.num_agents)]

        # Instantiate inverter models that simulate detailed inverter dynamics
        # Stamp microgrid id to each inverter.
        for i in range(self.num_microgrids):
            for j in range(self.num_agents):
                self.inverters = [Inverter(config, inverter_id=j, mg_id=i) for i in range(self.num_agents)]

        # Set DER positions; if not provided, generate random positions in a 10x10 area.
        if hasattr(config, "positions") and config.positions is not None:
            self.positions = config.positions
        else:
            self.positions = np.random.uniform(0, 10, (self.num_agents, 2))

        # Instantiate the communication module.
        self.comm_module = CommunicationModule(
            sigma=getattr(config, "comm_sigma", 1.0),
            threshold=getattr(config, "comm_threshold", None)
        )

    def set_measured_voltage(self, microgrid_id, voltage):
        """
        Set the measured voltage for a specific microgrid.

        Args:
            microgrid_id (int): Identifier for the microgrid.
            voltage (float): Measured voltage value.
        """
        # loop through each inverter in the microgrid and set the measured voltage.
        for i in range(self.num_agents):
            if (self.inverters[i].mg_id == microgrid_id):
                # Set the measured voltage for the inverter.
                self.inverters[i].V = voltage

    def reset(self):
        """
        Reset the secondary environment.

        Returns:
            List[dict]: The initial state for each DER/inverter agent.
        """
        self.time_step = 0
        # Reset inverter models.
        for inv in self.inverters:
            inv.reset()
        # Reset DER agents.
        for agent in self.der_agents:
            agent.current_q = 0.0

        self.states = [inv.get_state() for inv in self.inverters]
        return self.states

    def step(self, actions):
        """
        Execute one step in the secondary environment.

        Args:
            actions (List[float]): A list of desired voltage references (one per DER/inverter agent).

        Process:
          1. Each DER generates a message based on its current inverter state (e.g., voltage error).
          2. The communication module aggregates messages based on DER positions.
          3. Each inverter updates its state using the desired voltage reference (action) and its measured voltage.
          4. A reward is computed based on the voltage deviation from the global reference and a penalty on control effort.

        Returns:
            next_states (List[dict]): Updated state for each DER/inverter.
            rewards (List[float]): Reward for each agent.
            done (bool): True if maximum steps reached.
            info (dict): Additional info (empty in this example).
        """
        # 1. Collect messages from DER agents.
        messages = []
        for i in range(self.num_agents):
            current_voltage = self.inverters[i].get_state()["V"]
            msg = self.der_agents[i].get_message(current_voltage, self.V_ref)
            messages.append(msg)
        messages = np.array(messages)

        # 2. Aggregate communication signals.
        aggregated_comm = self.comm_module.aggregate_messages(messages, self.positions)

        next_states = []
        rewards = []
        # 3. Update inverter state for each agent.
        for i in range(self.num_agents):
            desired_voltage = actions[i]
            current_state = self.inverters[i].get_state()
            current_voltage = current_state["V"]
            new_state = self.inverters[i].update(V_ref=desired_voltage, measured_voltage=current_voltage)
            next_states.append(new_state)

            # 4. Compute reward: penalize squared voltage deviation from global V_ref and control effort.
            voltage_error = new_state["V"] - self.V_ref
            control_effort = abs(desired_voltage - current_voltage)
            reward = - (voltage_error ** 2) - self.action_penalty * control_effort
            rewards.append(reward)

        self.states = next_states
        self.time_step += 1
        done = self.time_step >= self.max_steps
        info = {
            "measured_volt": [current["V"] for current in next_states],
        }
        return next_states, rewards, done, info


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