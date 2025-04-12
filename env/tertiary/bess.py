# env/tertiary/bess.py

class BESS:
    def __init__(self, config, bess_id=0):
        """
        Initialize the Battery Energy Storage System (BESS) module.

        Args:
            config: Configuration object/dictionary with parameters such as:
                - bess_capacity (float): Maximum energy capacity in kWh.
                - bess_max_charge (float): Maximum charging power in kW.
                - bess_max_discharge (float): Maximum discharging power in kW.
                - bess_charge_efficiency (float): Charging efficiency (e.g., 0.95).
                - bess_discharge_efficiency (float): Discharging efficiency (e.g., 0.95).
                - bess_initial_soc (float): Initial state-of-charge (fraction, e.g., 0.5).
            bess_id: Identifier for this BESS unit.
        """
        self.bess_id = bess_id
        self.capacity = config.bess_capacity  # kWh
        self.max_charge = config.bess_max_charge  # kW
        self.max_discharge = config.bess_max_discharge  # kW
        self.charge_efficiency = config.bess_charge_efficiency
        self.discharge_efficiency = config.bess_discharge_efficiency
        self.initial_soc = config.bess_initial_soc  # fraction (0-1)
        self.soc = self.initial_soc

    def reset(self):
        """
        Reset the BESS to its initial state.
        """
        self.soc = self.initial_soc

    def apply_action(self, power):
        """
        Apply a charging or discharging action.

        Args:
            power (float): Desired power in kW. Positive for charging, negative for discharging.
                           The value will be constrained by max_charge and max_discharge limits.

        Returns:
            actual_power (float): The constrained power that was actually applied.
        """
        # Constrain the action within the allowed limits.
        if power > 0:
            power = min(power, self.max_charge)
        else:
            power = max(power, -self.max_discharge)

        # Assume a time step of 1 hour for simplicity; adjust if using a different timestep.
        if power >= 0:
            # Charging: apply efficiency factor.
            delta_soc = (power * self.charge_efficiency) / self.capacity
        else:
            # Discharging: consider discharging efficiency.
            delta_soc = (power / self.discharge_efficiency) / self.capacity

        # Update state-of-charge while ensuring it stays within [0, 1].
        self.soc = max(0, min(1, self.soc + delta_soc))
        return power

    def get_state(self):
        """
        Get the current state of the BESS.

        Returns:
            state (dict): Dictionary with key parameters like state-of-charge.
        """
        return {
            'soc': self.soc,
            'capacity': self.capacity,
            'max_charge': self.max_charge,
            'max_discharge': self.max_discharge
        }

    def get_operation_cost(self, power):
        """
        Compute an operational cost associated with a given charge/discharge action.
        This can be used to penalize rapid cycling or high power usage that might degrade the battery.

        Args:
            power (float): The power applied in kW.

        Returns:
            cost (float): A scalar cost value.
        """
        # Placeholder cost function: cost proportional to the absolute power used.
        return abs(power) * 0.01  # Adjust the factor as needed


if __name__ == "__main__":
    # Simple test scenario if running this module directly.
    class Config:
        bess_capacity = 100.0  # kWh
        bess_max_charge = 20.0  # kW
        bess_max_discharge = 20.0  # kW
        bess_charge_efficiency = 0.95
        bess_discharge_efficiency = 0.95
        bess_initial_soc = 0.5


    config = Config()
    bess = BESS(config)
    print("Initial State:", bess.get_state())
    action_power = 10.0  # Attempt to charge with 10 kW
    applied_power = bess.apply_action(action_power)
    print("Applied Action (kW):", applied_power)
    print("Updated State:", bess.get_state())