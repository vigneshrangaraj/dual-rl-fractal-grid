import numpy as np
import torch

class Helper:
    def __init__(self):
        pass

    @staticmethod
    def flatten_tertiary_state(state_dict):
        """
        Expected s
          {
             "microgrids": [
                {
                  "bess_soc": float,          # Battery SOC as a fraction
                  "load": float,              # Total load (MW)
                  "grid_power": float,        # Net grid power (MW) -- computed via power balance
                  "der_generation": float,    # Total DER generation (MW)
                  "measured_voltage": float   # Voltage at the storage bus (pu)
                },
                ...  (for each microgrid)
             ],
             "timestep": scalar            # Current simulation step
          }

        Returns:
          A torch.FloatTensor representing the flattened state vector, of shape
          ([num_microgrids * 5] + 1,).
        """
        import numpy as np
        import torch

        features = []
        # Process each microgrid state.
        microgrids = state_dict.get("microgrids", [])
        for mg in microgrids:
            bess_soc = mg.get("bess_soc", 0.0)
            load = mg.get("load", 0.0)
            grid_power = mg.get("grid_power", 0.0)
            der_generation = mg.get("der_generation", 0.0)
            measured_voltage = mg.get("measured_voltage", 0.0)
            # Concatenate the five values in order.
            features.extend([bess_soc, load, grid_power, der_generation, measured_voltage])

        # Append the global timestep.
        timestep = state_dict.get("timestep", 0)
        features.append(timestep)

        flat_state = np.array(features, dtype=np.float32)
        return torch.tensor(flat_state)

    @staticmethod
    def unpack_tertiary_action(action_vector):
        """
          A dict with keys:
            "dispatch_power": float,
            "battery_operation": float,   # positive: charge; negative: discharge
            "energy_shared": float
        """
        # If action_vector is a torch tensor, move to CPU and convert to numpy.
        if torch.is_tensor(action_vector):
            action_vector = action_vector.detach().cpu().numpy()
        return {
            "dispatch_power": float(action_vector[0]),
            "battery_operation": float(action_vector[1]),
            "energy_shared": float(action_vector[2])
        }