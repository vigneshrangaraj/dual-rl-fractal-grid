# env/secondary/comm.py
import numpy as np


class CommunicationModule:
    def __init__(self, sigma=1.0, threshold=None):
        self.sigma = sigma
        self.threshold = threshold

    def compute_distance_matrix(self, positions):
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        distance_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))
        return distance_matrix

    def compute_mask(self, positions):
        distance_matrix = self.compute_distance_matrix(positions)
        weights = np.exp(- (distance_matrix ** 2) / (2 * self.sigma ** 2))
        if self.threshold is not None:
            weights[distance_matrix > self.threshold] = 0.0
        return weights

    def aggregate_messages(self, messages, positions):
        weights = self.compute_mask(positions)  # Shape: (N, N)
        # Normalize weights for each DER (row-wise normalization)
        weight_sums = np.sum(weights, axis=1, keepdims=True)
        weight_sums[weight_sums == 0] = 1.0  # avoid division by zero
        normalized_weights = weights / weight_sums  # Shape: (N, N)

        # Extract the voltage error (first element) from each DER's message.
        voltage_errors = messages[:, 0]  # Shape: (N,)
        # Compute the weighted sum (aggregated communication signal) for each DER.
        aggregated_comm = normalized_weights @ voltage_errors  # Shape: (N,)
        return aggregated_comm


# --- Testing the updated Communication Module ---
if __name__ == "__main__":
    # Example with 3 DER agents in 2D space.
    positions = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 2.0]
    ])
    # Suppose each DER sends a message: [voltage_error, current_q]
    messages = np.array([
        [0.05, 1.0],
        [0.02, -0.5],
        [-0.01, 0.2]
    ])
    comm_module = CommunicationModule(sigma=1.0, threshold=2.0)
    agg_comm = comm_module.aggregate_messages(messages, positions)
    print("Aggregated communication signals (per DER):", agg_comm)