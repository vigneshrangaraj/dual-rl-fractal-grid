# env/tertiary/panda_power_wrapper.py

import pandapower as pp


class PandaPowerWrapper:
    def __init__(self, config):
        self.config = config
        self.microgrids = []
        self.tie_lines = []

    def reset_network(self, microgrids, tie_lines):
        """
        Args:
            microgrids (list): List of microgrid objects.
            tie_lines (list): List of tie-line tuples in the form (mg_id1, mg_id2, status).
                              Status should be 1 (closed) or 0 (open).
        """
        self.microgrids = microgrids
        self.tie_lines = tie_lines

    def run_power_flow(self, microgrids, tie_lines):
        results = {}

        # Run power flow for each microgrid individually.
        for mg in microgrids:
            try:
                pp.runpp(mg.net)
            except Exception as e:
                print(f"Power flow did not converge for microgrid {mg.mg_id}: {e}")
                # Set a default voltage value if the power flow fails.
                results[mg.mg_id] = self.config.V_ref
                continue

            # Compute an aggregated voltage (mean vm_pu) for the microgrid.
            v_avg = mg.net.res_bus.vm_pu.mean()
            results[mg.mg_id] = v_avg

        # Adjust the voltages based on tie-line connections.
        # For each tie-line tuple: (mg_id1, mg_id2, status)
        for tie in tie_lines:
            mg1, mg2, status = tie
            if status:  # if tie-line is closed, average the voltages
                if mg1 in results and mg2 in results:
                    avg_voltage = (results[mg1] + results[mg2]) / 2
                    results[mg1] = avg_voltage
                    results[mg2] = avg_voltage
        return results


# For testing the wrapper standalone.
if __name__ == "__main__":
    import pandapower.networks as nw


    class DummyMicroGrid:
        def __init__(self, mg_id, V_ref):
            self.mg_id = mg_id
            # Create a simple test network (e.g., an IEEE 9-bus system for illustration)
            self.net = nw.simple_four_bus_system()


    class Config:
        V_ref = 1.0


    config = Config()
    mg1 = DummyMicroGrid(0, config.V_ref)
    mg2 = DummyMicroGrid(1, config.V_ref)

    # Assume tie-line between mg1 and mg2 is closed.
    tie_lines = [(0, 1, 1)]

    wrapper = PandaPowerWrapper(config)
    wrapper.reset_network([mg1, mg2], tie_lines)
    results = wrapper.run_power_flow([mg1, mg2], tie_lines)
    print("Power Flow Results:", results)