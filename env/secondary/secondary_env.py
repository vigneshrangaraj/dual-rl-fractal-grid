import numpy as np
from env.secondary.comm import CommunicationModule
from env.secondary.inverter import Inverter
from env.tertiary.panda_power_wrapper import PandaPowerWrapper as pw


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

        self.V_min = getattr(config, "V_min", 0.5)
        self.V_max = getattr(config, "V_max", 2.0)
        self.max_voltage_change = getattr(config, "max_voltage_change", 0.05)

        self.config = config

        inv_buses = getattr(config, "combine_bus_inv_idx", None)

        self.inverters = []
        for i in range(self.num_microgrids):
            for j in range(self.num_agents):
                inverter = Inverter(config, inv_buses[j], j, i)
                self.inverters.append(inverter)

        self.rewards = []
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

    def step(self, actions, microgrids, tie_lines):
        next_states = []
        self.rewards = []
        new_actions = []
        new_inverter_states = []
        consensus_errors = []
        is_converged = False

        for i, inverter in enumerate(self.inverters):
            inverter = self.inverters[i]
            measured_voltage = inverter.measured_voltage

            neighbor_voltages = CommunicationModule.get_neighbor_voltages(i, self.states, self.adjacency_matrix)
            consensus_error = sum([(measured_voltage - vj) for vj in neighbor_voltages])
            consensus_error = np.clip(consensus_error, -0.1, 0.1)
            consensus_errors.append(consensus_error)

            action = actions[i]
            action = np.clip(action, getattr(self.config, "V_min"), getattr(self.config, "V_max"))
            action = action + 0.1 * consensus_error

            new_actions.append(action)

            for mg in microgrids:
                if mg.mg_id == inverter.mg_id:
                    mg.set_voltage_setpoint_for_der(action, i)
                    break

            new_inverter_state = inverter.update(V_ref=action, measured_voltage=measured_voltage)
            new_inverter_states.append(new_inverter_state)

        net, inv_voltages = pw.run_power_flow(microgrids, tie_lines)

        # set inverter measured voltage
        for i in range(len(self.config.combine_bus_inv_idx)):
            inverter = self.inverters[i]
            inverter.measured_voltage = inv_voltages[self.config.combine_bus_inv_idx[i] - 1]

        for i, inverter in enumerate(self.inverters):
            reward = 0
            new_voltage = inverter.measured_voltage

            if np.isnan(new_voltage):
                reward = -50
                new_voltage = 2
            else:
                is_converged = True

            action = new_actions[i]
            new_inverter_state = new_inverter_states[i]

            new_state = {
                "voltage": new_voltage,
                "reactive_power": action,
                "i_d": new_inverter_state["i_d"],
                "i_q": new_inverter_state["i_q"],
                "delta": new_inverter_state["delta"]
            }
            next_states.append(new_state)
            consensus_error = consensus_errors[i]

            v_i = new_voltage
            if 0.98 <= v_i <= 1.02:
                reward += 10.0
            elif 0.95 <= v_i < 0.98 or 1.02 < v_i <= 1.05:
                reward += 0.2 - abs(1.0 - v_i)
            elif 0.9 <= v_i < 0.95 or 1.05 < v_i <= 1.1:
                reward += -2.0 * abs(1.0 - v_i)
            else:
                reward += -10.0

            reward -= self.action_penalty * (action ** 2)
            reward -= self.consensus_penalty * (consensus_error ** 2)

            self.rewards.append(reward)

        self.states = next_states
        self.time_step += 1
        done = self.time_step >= self.max_steps or all(
            state["voltage"] > 0.98 and state["voltage"] < 1.02 for state in self.states
        )

        if done and self.time_step <= 2:
            # add + 20 to all the rewards
            print("---done at step " + str(self.time_step))
            self.rewards = [reward + 20 for reward in self.rewards]

        new_energy = pw.check_islanded_balance(net)

        info = {"net": net, "is_converged": is_converged, "new_energy": new_energy}

        print(f"==================Secondary step {self.time_step}: {np.mean(self.rewards)}")
        return self.states, self.rewards, done, info

    def default_ring_topology(self, n):
        adj = np.zeros((n, n))
        for i in range(n):
            adj[i][(i - 1) % n] = 1
            adj[i][(i + 1) % n] = 1
        return adj
