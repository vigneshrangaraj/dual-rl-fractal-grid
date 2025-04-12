# env/secondary/der.py
import numpy as np

class DER:
    def __init__(self, config, der_id=0, mg_id=0):
        self.der_id = der_id
        self.mg_id = mg_id
        self.max_q = getattr(config, "der_max_reactive_power", 5.0)
        self.min_q = getattr(config, "der_min_reactive_power", -5.0)
        self.response_gain = getattr(config, "der_response_gain", 1.0)
        self.comm_gain = getattr(config, "der_comm_gain", 0.5)
        self.current_q = 0.0

    def update(self, voltage, V_ref, aggregated_comm=0.0):
        voltage_error = V_ref - voltage
        q_out = self.response_gain * voltage_error + self.comm_gain * aggregated_comm
        q_out = np.clip(q_out, self.min_q, self.max_q)
        self.current_q = q_out
        return self.current_q

    def get_message(self, voltage, V_ref):
        voltage_error = V_ref - voltage
        return np.array([voltage_error, self.current_q])

    def get_state(self):
        return {
            "der_id": self.der_id,
            "current_q": self.current_q,
            "max_q": self.max_q,
            "min_q": self.min_q,
            "response_gain": self.response_gain,
            "comm_gain": self.comm_gain
        }

# --- Testing the updated DER module ---
if __name__ == "__main__":
    class Config:
        der_max_reactive_power = 5.0
        der_min_reactive_power = -5.0
        der_response_gain = 2.0
        der_comm_gain = 1.0

    config = Config()
    der = DER(config, der_id=1)
    print("Initial DER state:", der.get_state())
    # Assume local voltage is 0.95 pu, V_ref is 1.0 pu, and aggregated communication signal is 0.2.
    q_out = der.update(voltage=0.95, V_ref=1.0, aggregated_comm=0.2)
    print("Updated reactive power output (kVAR):", q_out)
    message = der.get_message(voltage=0.95, V_ref=1.0)
    print("DER message (voltage error, current_q):", message)