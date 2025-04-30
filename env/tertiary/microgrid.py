# env/tertiary/microgrid.py
import pandapower as pp
import numpy as np


class MicroGrid:
    def __init__(self, config, mg_id=0):
        self.mg_id = mg_id
        self.config = config
        self.V_ref = getattr(config, "V_ref", 1.0)
        self.num_buses = getattr(config, "num_buses", 8)
        self.neighbors = []
        self.switches = {}

        # cost params from config
        self.bess_cost_per_mwh = getattr(config, "bess_cost_per_mwh", 100.0)
        self.pv_cost_per_mwh = getattr(config, "pv_cost_per_mwh", 25.0)
        self.wind_cost_per_mwh = getattr(config, "wind_cost_per_mwh", 30.0)

        # Create the IEEE 34 bus network (a simplified version)
        self.net = self._create_ieee34_network()
        self.filled_net = None

        # Add a storage element (BESS) at a designated bus (e.g., from config or default to bus 10)
        self.storage_bus = getattr(config, "storage_bus", 2)
        self.storage_idx = self._add_storage()

        self.der_max_capacity = getattr(config, "der_max_capacity", 0.0)

        # Add DERs (solar and wind) at specified buses
        self._add_der()

    def add_neighbor(self, neighbor, switch_name):
        self.neighbors.append(neighbor)
        self.switches[switch_name] = 0  # Initially, all switches are open (0)

    def _create_ieee34_network(self):
        """
        Create a simplified IEEE 34 bus network.
        Loads are added on all buses except the slack bus.
        Lines are created between successive buses.
        An external grid is connected to bus 0 as the slack bus.
        """
        net = pp.create_empty_network()
        bus_indices = []

        # Create buses; assume nominal voltage as in config (e.g., 0.4 kV)
        for i in range(self.num_buses):
            bus = pp.create_bus(net, vn_kv=0.4)
            bus_indices.append(bus)

        # Create an external grid at bus 0 (slack)
        pp.create_ext_grid(net, bus=bus_indices[0], vm_pu=self.V_ref)

        # Create lines and add loads at subsequent buses
        for i in range(self.num_buses - 1):
            pp.create_line(net, from_bus=bus_indices[i], to_bus=bus_indices[i + 1],
                           length_km=0.1, std_type="NAYY 4x150 SE")
            # Add a load at each bus except the slack bus.
            pp.create_load(net, bus=bus_indices[i + 1],
                           p_mw=0.02, q_mvar=0.005,
                           name=f"Load_{i + 1}")
        return net

    def reset(self):
        self.net.res_bus.vm_pu = np.full(self.num_buses, self.V_ref)
        return self.get_state()



    def _add_storage(self):
        """
        Add a storage element to the network to simulate BESS.
        In PandaPower, a storage element has attributes like max/min energy, initial energy, and controllability.
        A negative p_mw indicates charging and a positive value indicates discharging.

        Returns:
            storage_idx: The index of the storage element in the PandaPower network.
        """
        bess_capacity = getattr(self.config, "bess_capacity", 100.0)  # kWh
        bess_max_charge = getattr(self.config, "bess_max_charge", 20.0)  # kW
        bess_max_discharge = getattr(self.config, "bess_max_discharge", 20.0)  # kW
        bess_initial_e = getattr(self.config, "bess_initial_e", 50.0)  # kWh, initial energy

        # Create storage at specified bus. p_mw is set to 0 initially.
        storage_idx = pp.create_storage(
            self.net,
            bus=self.storage_bus,
            p_mw=0.0,
            max_e_mwh=bess_capacity / 1000.0,
            min_e_mwh=0.0,
            initial_e_mwh=bess_initial_e / 1000.0,
            q_mvar=0.0,
            controllable=True,
            name="BESS"
        )
        return storage_idx

    def _add_der(self):
        """
        Add DERs (Solar and Wind) based on configuration.
        Solar and wind generators are added as static generators (sgen) at specified buses.
        """
        # Add solar DERs.
        solar_buses = getattr(self.config, "solar_buses", [5, 10])
        solar_base_output = getattr(self.config, "solar_base_output", 0.005)  # in MW
        for bus in solar_buses:
            pp.create_sgen(self.net, bus=bus,
                           p_mw=solar_base_output,
                           q_mvar=0.0,
                           name=f"Solar_{bus}")

        # Add wind DERs.
        wind_buses = getattr(self.config, "wind_buses", [15])
        wind_base_output = getattr(self.config, "wind_base_output", 0.007)  # in MW
        for bus in wind_buses:
            pp.create_sgen(self.net, bus=bus,
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
            self.net.sgen.loc[self.net.sgen['name'] == f"Solar_{bus}", 'p_mw'] = abs(action * available_p_mw)

        # Update Wind
        wind_buses = getattr(self.config, "wind_buses", [15])
        wind_base_output = getattr(self.config, "wind_base_output", 0.007)

        for i, bus in enumerate(wind_buses):
            action = dispatch_command  # 0 or 1
            available_p_mw = wind_base_output * wind_availability
            self.net.sgen.loc[self.net.sgen['name'] == f"Wind_{bus}", 'p_mw'] = abs(action * available_p_mw)

        # (optional) save availability factors if needed
        self.solar_availability = solar_availability
        self.wind_availability = wind_availability

    def apply_battery_operation(self, battery_operation):
        """
        Apply battery operation command to the BESS.
        Positive value indicates discharging, negative indicates charging.
        """
        # Update the storage element in the network
        battery_capacity = getattr(self.config, "bess_capacity", 100.0)  # kWh
        self.net.storage.loc[self.storage_idx, "p_mw"] = battery_operation * (battery_capacity / 1000.0)  # Convert kWh to MW
        # Update the state of charge (SOC) based on the operation
        current_energy = self.net.storage.initial_e_mwh.iloc[self.storage_idx]
        new_energy = current_energy + battery_operation * (battery_capacity / 1000.0)  # Convert kWh to MWh
        self.net.storage.loc[self.storage_idx, "initial_e_mwh"] = max(0, min(new_energy, battery_capacity / 1000.0))
        # Update the storage element in the network
        self.net.storage.loc[self.storage_idx, "initial_e_mwh"] = self.net.storage.initial_e_mwh.iloc[self.storage_idx]
        # Update the storage element in the network
        self.net.storage.loc[self.storage_idx, "p_mw"] = battery_operation * (battery_capacity / 1000.0)  # Convert kWh to MW

    def update_state_from_power_flow(self, power_flow_results):
        # if not NaN
        if power_flow_results is not None and not np.isnan(power_flow_results):
            self.measured_voltage = power_flow_results
            self.net.res_bus.vm_pu.loc[self.storage_idx] = self.measured_voltage
        else:
            self.measured_voltage = self.V_ref

    def get_grid_power(self):
        total_power = self.net.res_bus.p_mw.sum()
        return total_power

    def get_generation_cost(self):
        total_cost = 0.0

        # PV & Wind via sgen
        if self.filled_net is not None and "res_sgen" in self.filled_net and not self.filled_net.res_sgen.empty:
            sgen_data = self.filled_net.res_sgen
            types = self.filled_net.sgen["name"] if "name" in self.filled_net.sgen else ["pv"] * len(sgen_data)

            for i, row in sgen_data.iterrows():
                gen_type = types.iloc[i]
                power = row["p_mw"]

                if gen_type.startswith("Solar"):
                    total_cost += self.pv_cost_per_mwh * power
                elif gen_type.startswith("Wind"):
                    total_cost += self.wind_cost_per_mwh * power

        # BESS via storage
        if self.filled_net is not None and "res_storage" in self.filled_net and not self.filled_net.res_storage.empty:
            for _, row in self.filled_net.res_storage.iterrows():
                power = abs(row["p_mw"])  # both charging/discharging = degradation
                total_cost += self.bess_cost_per_mwh * power

        if (self.filled_net is None):
            # This means that the power flow failed
            # and we have no generation cost.
            # We need to penalize this from a cost perspective
            total_cost = 100.0
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
        energy = self.net.storage.initial_e_mwh
        bess_capacity = getattr(self.config, "bess_capacity", 100.0)
        bess_soc = energy / (bess_capacity / 1000.0)  # Both in MWh
        total_load = self.net.load.p_mw.sum()

        # Make bess_soc is a tuple of (1,). Grab the first element if it's an array.
        bess_soc = bess_soc[0]

        # Sum generation from DERs (sgen)
        der_generation = self.net.sgen.p_mw.sum()

        # Measured voltage: update via power flow results if available.
        try:
            measured_voltage = self.net.res_bus.vm_pu.loc[self.net.storage.bus[self.storage_idx]]
        except Exception:
            measured_voltage = self.V_ref

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
        num_buses = 34
        storage_bus = 10
        bess_capacity = 100.0  # kWh
        bess_max_charge = 20.0  # kW
        bess_max_discharge = 20.0  # kW
        bess_initial_e = 50.0  # kWh
        solar_buses = [5, 10]
        solar_base_output = 0.005  # MW
        wind_buses = [15]
        wind_base_output = 0.007  # MW


    config = Config()
    mg = MicroGrid(config, mg_id=0)
    print("Initial MicroGrid State:")
    state = mg.get_state()
    print(state)

    # Example: apply a dispatch command to charge the battery at 5 MW (so negative sign indicates charging)
    mg.apply_dispatch(-5.0)
    print("Dispatch applied: -5 MW")

    # Run power flow
    try:
        pp.runpp(mg.net)
        # Update state based on power flow results
        mg.update_state_from_power_flow(None)
        updated_state = mg.get_state()
        print("Updated MicroGrid State:")
        print(updated_state)
    except Exception as e:
        print("Error running power flow:", e)