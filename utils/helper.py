import numpy as np
import torch
from utils.config import Config as config

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
    def flatten_tertiary_action(action_dict):
        """
        Expected action_dict:
          {
            "microgrids": [
                {
                  "dispatch_power": float,
                  "battery_operation": float
                },
                ...  (for each microgrid)
            ],
            "tie_lines": [
                (from_mg, to_mg, value),  # value is either 0 or 1
                ...  (for each tie line)
            ]
          }

        Returns:
          A numpy array representing the flattened action vector.
        """
        action_vector = []
        microgrids = action_dict.get("microgrids", [])
        for mg in microgrids:
            dispatch_power = mg.get("dispatch_power", 0.0)
            battery_operation = mg.get("battery_operation", 0.0)
            action_vector.extend([dispatch_power, battery_operation])

        tie_lines = action_dict.get("tie_lines", [])
        for tie_line in tie_lines:
            value = tie_line[2]
            action_vector.append(value)

        # Convert to numpy array
        action_vector = np.array(action_vector, dtype=np.float32)
        return action_vector

    @staticmethod
    def unpack_tertiary_action(action_vector, switch_set):
        """
          A dict with keys:
            - "mictogrid": "dict" with keys:
                - "dispatch_power": float
                - "battery_operation": float
                - "tie_lines": tuple (from, to, value)
        """
        # If action_vector is a torch tensor, move to CPU and convert to numpy.
        if torch.is_tensor(action_vector):
            action_vector = action_vector.detach().cpu().numpy()

        num_microgrids = getattr(config, "num_microgrids", 1)
        # for each microgrid, we have two actions: dispatch_power and battery_operation
        # and the rest are tie lines
        microgrid_actions = []

        for i in range(num_microgrids):
            dispatch_power = action_vector[i * 2]
            battery_operation = action_vector[i * 2 + 1]
            microgrid_actions.append({
                "dispatch_power": dispatch_power,
                "battery_operation": battery_operation
            })

        # The rest of the action vector is for tie lines
        # switch set is a set of switches formatted like "S_0_to_1"
        # for the tielines we want something like (0, 1, 1) or (0, 1, 0)
        # where 1 means closed and 0 means open
        tie_lines = []
        k = 0
        for j in switch_set:
            # Extract the microgrid indices from the switch name
            indices = j.split("_")
            from_mg = int(indices[1])
            to_mg = int(indices[3])
            # The value is either 0 or 1
            value = action_vector[(num_microgrids * 2) + k]
            tie_lines.append((from_mg, to_mg, value))
            k += 1

        return {
            "microgrids": microgrid_actions,
            "tie_lines": tie_lines
        }






