# env/secondary/comm.py
import numpy as np
from fns.der_4 import der_4


class CommunicationModule:
    def __init__(self, config, sigma=1.0, threshold=None ):
        self.sigma = sigma
        self.threshold = threshold
        self.fn = der_4()

        inv_buses = self.fn.combine_bus_inv_idx
        self.num_agents = self.fn.num_secondary_agents
        self.num_microgrids = getattr(config, "num_microgrids", 1)
        self.adjacency_matrix = self.default_ring_topology(self.num_agents * self.num_microgrids)

    def default_ring_topology(self, n):
        adj = np.zeros((n, n))
        for i in range(n):
            adj[i][(i - 1) % n] = 1
            adj[i][(i + 1) % n] = 1
        return adj

    def get_neighbor_voltages(self, agent_id, states, adjacency_matrix):
        neighbors = np.where(adjacency_matrix[agent_id] > 0)[0]
        return [states[n]["voltage"] for n in neighbors]

    def get_neighbors(self, agent_id):
        def get_neighbors(self, agent_id):
            """
            Returns the list of neighbor agent IDs for a given agent ID based on the adjacency matrix.

            Args:
                agent_id (int): The ID of the agent whose neighbors are being queried.

            Returns:
                List[int]: A list of neighboring agent IDs.
            """
        return list(np.where(self.adjacency_matrix[agent_id] > 0)[0])



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