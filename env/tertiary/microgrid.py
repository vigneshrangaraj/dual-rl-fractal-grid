# env/tertiary/microgrid.py
import pandapower as pp
import numpy as np
from pandapower.plotting.plotly import simple_plotly
from pandapower.networks.create_examples import example_simple
import fns.der_4 as der4
from fns.der_4 import der_4


class MicroGrid:
    def __init__(self, config, mg_id=0):
        self.mg_id = mg_id
        self.config = config
        self.V_ref = getattr(config, "V_ref", 1.0)
        self.num_buses = getattr(config, "num_buses", 8)
        self.neighbors = []
        self.switches = {}
        self.pf_net = None

        self.storage_idx = None

        self.last_soc = 0.5

        # Create the IEEE 34 bus network (a simplified version)
        self.base_loads = []
        self.last_stored_battery_p_mw = 0.0

        self.wind_buses = None
        self.combine_bus_inv_idx = None
        self.num_buses = None
        self.num_secondary_agents = None
        self.solar_buses = None

        self.net = self._create_ieee34_network()

    def add_neighbor(self, neighbor, switch_name):
        self.neighbors.append(neighbor)
        self.switches[switch_name] = 0  # Initially, all switches are open (0)

    def _create_ieee34_network(self):
        der4_net = der4.der_4()
        self.wind_buses = der4_net.wind_buses
        self.combine_bus_inv_idx = der4_net.combine_bus_inv_idx
        self.num_buses = der4_net.num_buses
        self.num_secondary_agents = der4_net.num_secondary_agents
        self.solar_buses = der4_net.solar_buses
        self.storage_idx = der4_net.get_storage_idx()

        # populate base loads
        for i in range(self.num_buses):
            self.base_loads.append(der4_net.net.load.p_mw[i])

        return der4_net.get_network()

    def reset(self):
        self.net.res_bus.vm_pu = np.full(self.num_buses, self.V_ref)

    def get_voltage_at_bus(self, bus_idx):
        """
        Get the voltage at a specific bus.
        """
        try:
            return self.net.res_bus.vm_pu.loc[bus_idx]
        except Exception:
            return None

    def check_for_generation_overload(self):
        """
        Check if the generation exceeds current load.
        If so, return True, else return False. also return how much
        """
        # Check if the generation exceeds the load
        total_generation = self.net.gen.p_mw.sum() + self.net.storage.p_mw.sum()
        total_load = self.net.load.p_mw.sum()

        if total_generation > total_load:
            return True, total_generation - total_load
        elif total_generation < total_load:
            return False, total_load - total_generation
        else:
            return False, 0.0

    def _add_der(self):
        """
        Add DERs (Solar and Wind) based on configuration.
        Solar and wind generators are added as static generators (gen) at specified buses.
        """
        # Add solar DERs.
        solar_buses = self.solar_buses
        solar_base_output = getattr(self.config, "solar_base_output", 0.005)  # in MW
        for bus in solar_buses:
            pp.create_gen(self.net, bus=bus,
                           p_mw=solar_base_output,
                           q_mvar=0.0,
                           name=f"Solar_{bus}")

        # Add wind DERs.
        wind_buses = self.wind_buses
        wind_base_output = getattr(self.config, "wind_base_output", 0.007)  # in MW
        for bus in wind_buses:
            pp.create_gen(self.net, bus=bus,
                           p_mw=wind_base_output,
                           q_mvar=0.0,
                           name=f"Wind_{bus}")

    def get_grid_power(self):
        """
        Get the power exchanged with the grid after running power flow.
        Positive → grid supplies power (import), Negative → grid absorbs power (export).
        """
        if (self.pf_net is not None and "res_ext_grid" in self.pf_net and not self.pf_net.res_ext_grid.empty):
            return self.pf_net.res_ext_grid.p_mw.sum()
        else:
            return 0.0

    def get_generation_cost(self):
        total_cost = 0.0

        # PV & Wind via gen
        if self.net is not None and "gen" in self.net and not self.net.gen.empty:
            gen_data = self.net.gen

            for i, row in gen_data.iterrows():
                power = row["p_mw"]
                total_cost += 10 * power

        # BESS via storage
        if self.net is not None and "storage" in self.net and not self.net.storage.empty:
            for _, row in self.net.storage.iterrows():
                power = abs(row["p_mw"])  # both charging/discharging = degradation
                total_cost += 10 * power

        return total_cost

    def get_voltage_deviation(self):
        return self.measured_voltage - self.V_ref

    # def get_state(self):
    #     """
    #         dict: State dictionary, for example:
    #           {
    #               "bess_soc": SOC as fraction,
    #               "load": total load on the network (MW),
    #               "der_generation": total DER generation (MW),
    #               "measured_voltage": voltage at the storage bus,
    #               "timestep": current simulation step
    #           }
    #     """
    #     # Extract storage information from PandaPower.
    #     bess_soc = self.last_soc  # Both in MWh
    #     total_load = self.net.load.p_mw.sum()
    #
    #     # Sum generation from DERs (gen)
    #     der_generation = self.net.gen.p_mw.sum()
    #     try:
    #         voltages = self.pf_net.res_gen.vm_pu.sum()
    #         avg_voltage = float(voltages) / len(self.pf_net.res_gen.vm_pu)
    #         measured_voltage = avg_voltage
    #     except Exception:
    #         measured_voltage = 5
    #     if np.isnan(measured_voltage):
    #         measured_voltage = 5
    #
    #     # For simplicity, assume timestep is tracked externally.
    #     # Here we set it to 0 as a placeholder.
    #     timestep = 0
    #
    #     return {
    #         "bess_soc": bess_soc,
    #         "load": total_load,
    #         "grid_power": self.get_grid_power(),
    #         "der_generation": der_generation,
    #         "measured_voltage": measured_voltage
    #     }


# --- Testing the MicroGrid Module ---
if __name__ == "__main__":
    # Create a dummy configuration for testing.
    class Config:
        V_ref = 1.0
        num_buses = 10
        storage_bus = 9
        bess_capacity = 100.0  # kWh
        bess_max_charge = 20.0  # kW
        bess_max_discharge = 20.0  # kW
        bess_initial_e = 50.0  # kWh
        solar_buses = [4, 6]
        solar_base_output = 0.005  # MW
        wind_buses = [7]
        wind_base_output = 0.007  # MW


    config = Config()
    mg = MicroGrid(config, mg_id=0)
    print("Initial MicroGrid State:")
    state = mg.get_state()
    print(state)

    # Example: apply a dispatch command to charge the battery at 5 MW (so negative sign indicates charging)
    mg.apply_dispatch(100, 0)
    print("Dispatch applied: -5 MW")



    # Run power flow
    try:
        net = example_simple()
        simple_plotly(net)
        pp.runpp(net, max_iteration=100, tolerance=1e-3)
        # Update state based on power flow results
        mg.pf_net = mg.net
        updated_state = mg.get_state()
        print("Updated MicroGrid State:")
        print(updated_state)
    except Exception as e:
        print("Error running power flow:", e)