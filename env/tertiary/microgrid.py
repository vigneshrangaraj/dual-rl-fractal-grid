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
        self.net = self._create_ieee34_network()


    def add_neighbor(self, neighbor, switch_name):
        self.neighbors.append(neighbor)
        self.switches[switch_name] = 0  # Initially, all switches are open (0)

    def _create_ieee34_network(self):
        der4_net = der4.der_4()
        self.storage_idx = der4_net.get_storage_idx()

        return der4_net.get_network()

    def reset(self):
        self.net.res_bus.vm_pu = np.full(self.num_buses, self.V_ref)
        return self.get_state()

    def get_voltage_at_bus(self, bus_idx):
        """
        Get the voltage at a specific bus.
        """
        try:
            return self.net.res_bus.vm_pu.loc[bus_idx]
        except Exception:
            return None

    def set_voltage_setpoint_for_der(self, voltage_setpoint, inverter_idx):
        """
        Set the voltage setpoint for DERs.
        """
        # Set the voltage setpoint for the inverter
        self.net.gen.loc[inverter_idx, "vm_pu"] = voltage_setpoint

    def apply_load_p_mv_by_timestep(self, time_step):
        """
        Get the load in MW at each bus for a given time step.
        Add some randomness to the load to simulate real-world conditions.
        """
        # Load is assumed to be a function of time_step
        load_factor = 1.0 + 0.1 * np.sin(time_step / 10.0)

        for i in range(self.num_buses):
            # Set the load at each bus
            self.net.load.loc[i, "p_mw"] = load_factor * self.net.load.loc[i, "p_mw"]

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
        solar_buses = getattr(self.config, "solar_buses", [5, 10])
        solar_base_output = getattr(self.config, "solar_base_output", 0.005)  # in MW
        for bus in solar_buses:
            pp.create_gen(self.net, bus=bus,
                           p_mw=solar_base_output,
                           q_mvar=0.0,
                           name=f"Solar_{bus}")

        # Add wind DERs.
        wind_buses = getattr(self.config, "wind_buses", [15])
        wind_base_output = getattr(self.config, "wind_base_output", 0.007)  # in MW
        for bus in wind_buses:
            pp.create_gen(self.net, bus=bus,
                           p_mw=wind_base_output,
                           q_mvar=0.0,
                           name=f"Wind_{bus}")

    def get_available_der_output(self, time_step):
        """
        Simulates available solar and wind generation based on hour of the day (0-23).
        Returns (solar_availability_factor, wind_availability_factor).
        """

        # Solar Availability
        if 6 <= time_step <= 18:
            # Parabolic shape peaking at noon
            solar_peak = 1.0
            solar_availability = max(0.0, -0.01 * (time_step - 12) ** 2 + solar_peak)
        else:
            solar_availability = 0.0

        # Wind Availability
        np.random.seed(time_step)
        base_wind = 0.5  # Base wind availability
        fluctuation = np.random.uniform(-0.2, 0.2)  # +/-20% random fluctuation
        wind_availability = np.clip(base_wind + fluctuation, 0.0, 1.0)

        return solar_availability, wind_availability

    def apply_dispatch(self, dispatch_command, time_step):
        """
        Apply dispatch command to DERs.
        Now calls get_available_der_output() to get realistic availability based on time of day.
        """

        # Get available generation factors
        solar_availability, wind_availability = self.get_available_der_output(time_step)

        # Update Solar
        solar_buses = getattr(self.config, "solar_buses", [5, 10])
        solar_base_output = getattr(self.config, "solar_base_output", 0.005)

        for i, bus in enumerate(solar_buses):
            action = dispatch_command  # 0 or 1
            available_p_mw = solar_base_output * solar_availability
            self.net.gen.loc[self.net.gen['name'] == f"Solar_{bus}", 'p_mw'] = abs(action * available_p_mw)

        # Update Wind
        wind_buses = getattr(self.config, "wind_buses", [15])
        wind_base_output = getattr(self.config, "wind_base_output", 0.007)

        for i, bus in enumerate(wind_buses):
            action = dispatch_command  # 0 or 1
            available_p_mw = wind_base_output * wind_availability
            self.net.gen.loc[self.net.gen['name'] == f"Wind_{bus}", 'p_mw'] = abs(action * available_p_mw)

        # (optional) save availability factors if needed
        self.solar_availability = solar_availability
        self.wind_availability = wind_availability

    def apply_battery_operation(self, battery_operation):
        max_e_mwh = self.net.storage.loc[self.storage_idx, "max_e_mwh"]

        # Convert SOC to stored energy
        stored_energy_mwh = self.last_soc * max_e_mwh

        energy_change = battery_operation * self.net.storage.loc[self.storage_idx, "max_e_mwh"]

        # Clamp the new energy within limits
        new_energy = np.clip(stored_energy_mwh + energy_change, 0.0, max_e_mwh)

        # Update SOC
        new_soc = new_energy / max_e_mwh
        if not (0.0 < new_soc < 1.0):
            print("Battery SOC out of bounds:", new_soc)
            return

        self.last_soc = new_soc

        # Update network
        self.net.storage.loc[self.storage_idx, "p_mw"] = battery_operation * self.net.storage.loc[self.storage_idx, "max_e_mwh"]
        self.net.storage.loc[self.storage_idx, "initial_e_mwh"] = new_energy
        self.net.storage.loc[self.storage_idx, "soc_percent"] = new_soc * 100

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

    def get_state(self):
        """
            dict: State dictionary, for example:
              {
                  "bess_soc": SOC as fraction,
                  "load": total load on the network (MW),
                  "der_generation": total DER generation (MW),
                  "measured_voltage": voltage at the storage bus,
                  "timestep": current simulation step
              }
        """
        # Extract storage information from PandaPower.
        bess_soc = self.last_soc  # Both in MWh
        total_load = self.net.load.p_mw.sum()

        # Sum generation from DERs (gen)
        der_generation = self.net.gen.p_mw.sum()
        try:
            measured_voltage = self.pf_net.res_bus.vm_pu.loc[self.net.storage.bus[self.storage_idx]]
        except Exception:
            measured_voltage = 5
        if np.isnan(measured_voltage):
            measured_voltage = 5

        # For simplicity, assume timestep is tracked externally.
        # Here we set it to 0 as a placeholder.
        timestep = 0

        return {
            "bess_soc": bess_soc,
            "load": total_load,
            "grid_power": self.get_grid_power(),
            "der_generation": der_generation,
            "measured_voltage": measured_voltage
        }


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