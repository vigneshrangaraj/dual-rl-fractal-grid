import os
import numpy as np

class ActionStatePlotter:
    def __init__(self):
        self.episodes = []
        self.tertiary_actions = []
        self.secondary_actions = []
        self.tertiary_states = []
        self.secondary_states = []

    def save_data(self):
        """
        Save all plot data as .npy files in the plots directory for later analysis.
        """
        os.makedirs("plots", exist_ok=True)
        np.save("plots/episodes.npy", np.array(self.episodes))
        np.save("plots/tertiary_actions.npy", np.array(self.tertiary_actions))
        np.save("plots/secondary_actions.npy", np.array(self.secondary_actions))
        np.save("plots/tertiary_states.npy", np.array(self.tertiary_states))
        np.save("plots/secondary_states.npy", np.array(self.secondary_states)) 