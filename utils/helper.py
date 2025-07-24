import numpy as np
import torch
from utils.config import Config as config

class Helper:
    def __init__(self):
        pass

    @staticmethod
    def flatten_tertiary_state(state_dict):
        """
        Expected state_dict:
          {
             "microgrids": [
                {
                  "bess_soc": [float, ...],          # List of BESS SOCs as fractions
                  "load": float,              # Total load (MW)
                  "grid_power": float,        # Net grid power (MW)
                  "der_generation": float,    # Total DER generation (MW)
                  "measured_voltage": float   # Voltage at the storage bus (pu)
                },
                ...  (for each microgrid)
             ],
             "timestep": scalar           # Current simulation step
          }

        Returns:
          A torch.FloatTensor representing the flattened state vector.
        """
        import numpy as np
        import torch
        features = []
        num_bess_total = getattr(config, "num_bess_total", 1)
        # Process each microgrid state.
        microgrids = state_dict.get("microgrids", [])
        for mg in microgrids:
            # BESS SOCs as a list
            bess_socs = mg.get("bess_soc", [0.0] * num_bess_total)
            features.extend(list(bess_socs))
            load = mg.get("total_load", 0.0)
            grid_power = mg.get("grid_power", 0.0)
            der_generation = mg.get("der_generation", 0.0)
            measured_voltage = mg.get("measured_voltage", 0.0)
            features.extend([load, grid_power, der_generation, measured_voltage])
        # Append the global timestep.
        timestep = state_dict.get("timestep", 0)
        features.append(timestep)
        flat_state = np.array(features, dtype=np.float32)
        return torch.tensor(flat_state)

    @staticmethod
    def add_to_aggregated_state(aggregated_state, state_dict):
        """
        Adds the microgrid states from state_dict to aggregated_state.
        Handles bess_soc as a list and averages elementwise.
        """
        keys = ["bess_soc", "total_load", "grid_power", "der_generation", "measured_voltage"]
        microgrids = state_dict.get("microgrids", [])
        for mg_idx, mg in enumerate(microgrids):
            agg_microgrids = aggregated_state.get("microgrids", [])
            if mg_idx >= len(agg_microgrids):
                agg_microgrids.append({k: 0.0 for k in keys})
            agg_mg = agg_microgrids[mg_idx]
            for k in keys:
                if k == "bess_soc":
                    # Both should be lists
                    bess_soc_new = mg.get("bess_soc", [])
                    bess_soc_agg = agg_mg.get("bess_soc", [0.0] * len(bess_soc_new))
                    # Elementwise average
                    if len(bess_soc_new) != len(bess_soc_agg):
                        # If lengths mismatch, pad with zeros
                        max_len = max(len(bess_soc_new), len(bess_soc_agg))
                        bess_soc_new = list(bess_soc_new) + [0.0] * (max_len - len(bess_soc_new))
                        bess_soc_agg = list(bess_soc_agg) + [0.0] * (max_len - len(bess_soc_agg))
                    agg_mg["bess_soc"] = [ (b + n) / 2.0 for b, n in zip(bess_soc_agg, bess_soc_new) ]
                else:
                    agg_mg[k] += mg.get(k, 0.0)
            agg_microgrids[mg_idx] = agg_mg
        aggregated_state["timestep"] = state_dict.get("timestep", 0)

    @staticmethod
    def flatten_tertiary_action(action_dict):
        """
        Expected action_dict:
          {
            "microgrids": [
                {
                  "der_actions": [float, ...],  # Configurable number of DER actions
                  "battery_operation": [float, ...], # List of BESS actions
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
        num_der_total = getattr(config, "num_der_total", 4)
        num_bess_total = getattr(config, "num_bess_total", 1)
        for mg in microgrids:
            der_actions = mg.get("der_actions", [0.0] * num_der_total)
            battery_operations = mg.get("battery_operation", [0.0] * num_bess_total)
            action_vector.extend(der_actions)  # Add DER actions
            action_vector.extend(battery_operations)  # Add all BESS actions
        tie_lines = action_dict.get("tie_lines", [])
        for tie_line in tie_lines:
            value = tie_line[2]
            action_vector.append(value)
        action_vector = np.array(action_vector, dtype=np.float32)
        return action_vector

    @staticmethod
    def unpack_tertiary_action(action_vector, switch_set):
        """
          A dict with keys:
            - "microgrids": list of dicts with keys:
                - "der_actions": list of configurable floats (one for each DER)
                - "battery_operation": list of floats (one for each BESS)
            - "tie_lines": list of tuples (from, to, value)
        """
        if torch.is_tensor(action_vector):
            action_vector = action_vector.detach().cpu().numpy()
        num_microgrids = getattr(config, "num_microgrids", 1)
        num_der_total = getattr(config, "num_der_total", 4)
        num_bess_total = getattr(config, "num_bess_total", 1)
        microgrid_actions = []
        offset = 0
        for i in range(num_microgrids):
            der_actions = list(action_vector[offset:offset + num_der_total])
            offset += num_der_total
            battery_operations = list(action_vector[offset:offset + num_bess_total])
            offset += num_bess_total
            microgrid_actions.append({
                "der_actions": der_actions,
                "battery_operation": battery_operations
            })
        tie_lines = []
        for j, switch in enumerate(switch_set):
            indices = switch.split("_")
            from_mg = int(indices[1])
            to_mg = int(indices[3])
            value = action_vector[offset + j]
            tie_lines.append((from_mg, to_mg, value))
        return {
            "microgrids": microgrid_actions,
            "tie_lines": tie_lines
        }






