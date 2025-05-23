from typing import List, Dict, Any
import matplotlib.pyplot as plt
import numpy as np
import os

def flatten_tertiary_action(ter_action: Dict[str, Any]) -> List[float]:
    flat = []
    for mg in ter_action.get('microgrids', []):
        flat.append(float(mg.get('dispatch_power', 0.0)))
        flat.append(float(mg.get('battery_operation', 0.0)))
    return flat

def flatten_tertiary_state(ter_state: Dict[str, Any]) -> List[float]:
    flat = []
    for mg in ter_state.get('microgrids', []):
        flat.append(float(mg.get('bess_soc', 0.0)))
        flat.append(float(mg.get('load', 0.0)))
        flat.append(float(mg.get('grid_power', 0.0)))
        flat.append(float(mg.get('der_generation', 0.0)))
        flat.append(float(mg.get('measured_voltage', 0.0)))
    flat.append(float(ter_state.get('timestep', 0.0)))
    return flat

class ActionStatePlotter:
    def __init__(
        self,
        num_secondary_agents: int,
        num_tertiary_actions: int,
        num_tertiary_states: int,
        num_secondary_states: int
    ):
        self.episodes: List[int] = []
        self.tertiary_actions: List[List[float]] = [[] for _ in range(num_tertiary_actions)]
        self.secondary_actions: List[List[float]] = [[] for _ in range(num_secondary_agents)]
        self.tertiary_states: List[List[float]] = [[] for _ in range(num_tertiary_states)]
        self.secondary_states: List[List[float]] = [[] for _ in range(num_secondary_agents * num_secondary_states)]

        self.tertiary_action_labels = ["Dispatch Power", "Battery Operation"]
        self.tertiary_state_labels = ["BESS SOC", "Load", "Grid Power", "DER Generation", "Measured Voltage", "Timestep"]
        self.secondary_state_labels = ["Voltage", "Reactive Power", "i_d", "i_q", "Delta"]

        os.makedirs("plots", exist_ok=True)
        self.fig, self.axes = plt.subplots(5, 1, figsize=(14, 24))
        plt.ion()

        titles = [
            "Tertiary Actions",
            "Secondary Actions",
            "Tertiary States",
            "BESS SOC",
            "Secondary States"
        ]
        for ax, title in zip(self.axes, titles):
            ax.set_title(title)
            ax.set_xlabel("Episode")
            ax.set_ylabel("Value")
            ax.grid(True)

        plt.tight_layout()
        self.fig.subplots_adjust(right=0.75)

    def update(
        self,
        episode: int,
        tertiary_action: Dict[str, Any],
        secondary_actions: List[float],
        tertiary_state: Dict[str, Any],
        secondary_states: List[object]  # dicts or lists
    ) -> None:
        self.episodes.append(episode)

        flat_tertiary_action = flatten_tertiary_action(tertiary_action)
        flat_tertiary_state = flatten_tertiary_state(tertiary_state)

        for i in range(len(self.tertiary_actions)):
            try:
                self.tertiary_actions[i].append(float(flat_tertiary_action[i]))
            except IndexError:
                print(f"[Warning] Tertiary action index {i} not available in episode {episode}.")
                self.tertiary_actions[i].append(float("nan"))

        for i, a in enumerate(secondary_actions):
            self.secondary_actions[i].append(float(a))
        for i, s in enumerate(flat_tertiary_state):
            self.tertiary_states[i].append(float(s))

        num_secondary_agents = len(secondary_states)
        num_secondary_states = len(list(secondary_states[0].values()) if isinstance(secondary_states[0], dict) else secondary_states[0]) if num_secondary_agents > 0 else 0
        for agent_idx in range(num_secondary_agents):
            agent_data = secondary_states[agent_idx]
            state_values = list(agent_data.values()) if isinstance(agent_data, dict) else agent_data
            for state_idx, value in enumerate(state_values):
                flat_idx = agent_idx * num_secondary_states + state_idx
                self.secondary_states[flat_idx].append(float(value))

        self.axes[0].cla()
        self.axes[0].set_title("Tertiary Actions")
        for i, actions in enumerate(self.tertiary_actions):
            label = self.tertiary_action_labels[i] if i < len(self.tertiary_action_labels) else f"Action {i}"
            self.axes[0].plot(self.episodes, actions, label=label)
        self.axes[0].legend(loc='upper left', bbox_to_anchor=(1.01, 1), ncol=1, fontsize=8)

        self.axes[1].cla()
        self.axes[1].set_title("Secondary Actions")
        for i, actions in enumerate(self.secondary_actions):
            self.axes[1].plot(self.episodes, actions, label=f"Agent {i}")
        self.axes[1].legend(loc='upper left', bbox_to_anchor=(1.01, 1), ncol=1, fontsize=8)

        self.axes[2].cla()
        self.axes[2].set_title("Tertiary States")
        for i, states in enumerate(self.tertiary_states[1:]):  # exclude BESS SOC (index 0)
            label = self.tertiary_state_labels[i + 1] if (i + 1) < len(self.tertiary_state_labels) else f"State {i + 1}"
            self.axes[2].plot(self.episodes, states, label=label)
        self.axes[2].legend(loc='upper left', bbox_to_anchor=(1.01, 1), ncol=1, fontsize=8)

        self.axes[3].cla()
        self.axes[3].set_title("BESS SOC")
        if len(self.tertiary_states[0]) == len(self.episodes):
            self.axes[3].plot(self.episodes, self.tertiary_states[0], label="BESS SOC")
            self.axes[3].legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=8)

        self.axes[4].cla()
        self.axes[4].set_title("Secondary States")
        for i, states in enumerate(self.secondary_states):
            agent = i // num_secondary_states if num_secondary_states else 0
            state = i % num_secondary_states if num_secondary_states else 0
            label = f"Agent {agent} {self.secondary_state_labels[state] if state < len(self.secondary_state_labels) else f'State {state}'}"
            self.axes[4].plot(self.episodes, states, label=label)
        self.axes[4].legend(loc='upper left', bbox_to_anchor=(1.01, 1), ncol=1, fontsize=8)

        plt.tight_layout()
        self.fig.subplots_adjust(right=0.75)
        plt.draw()
        plt.pause(0.01)
