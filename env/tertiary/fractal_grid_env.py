# env/tertiary/fractal_grid_env.py

import numpy as np
import logging

from env.tertiary.microgrid import MicroGrid
from env.tertiary.panda_power_wrapper import PandaPowerWrapper
import pandapower as pp
import pandas as pd


class FractalGridEnv:
    def __init__(self, config):
        """
        Initialize the fractal grid environment.

        Args:
            config: Configuration object/dict with parameters including:
                    - num_microgrids: Number of microgrids in the fractal grid.
                    - tie_lines: List of tie-line connections, e.g. [(mg1, mg2, status), ...].
                    - V_ref: Reference voltage.
                    - max_steps: Maximum simulation steps per episode.
                    - lambda_econ, alpha_sec, beta_volt: Reward weighting factors.
        """
        self.config = config
        self.num_microgrids = getattr(config, "num_microgrids", 1)
        self.tie_lines = getattr(config, "tie_lines", [])
        self.V_ref = getattr(config, "V_ref", 1.0)
        self.max_steps = getattr(config, "max_steps", 100)
        self.lambda_econ = getattr(config, "lambda_econ", 1.0)
        self.alpha_sec = getattr(config, "alpha_sec", 1.0)
        self.beta_volt = getattr(config, "beta_volt", 1.0)
        self.load_disturbance_percent = getattr(config, "load_disturbance_percent", 0.1)

        self.index_map = {}
        self.tie_switch_index_map = {}

        self.peak_hours = config.peak_hours
        self.off_peak_hours = config.off_peak_hours

        self.net = None

        # Create microgrid instances
        self.microgrids, self.switches, self.switch_set = self.initialize_microgrids(config)

        # Initialize PandaPower wrapper for AC power flow computations.
        self.pw_wrapper = PandaPowerWrapper(config)
        self.current_step = 0
        self.done = False

    def initialize_microgrids(self, config):
        # Initialize and connect microgrids based on a Fractal Tree algorithm
        microgrids = []

        # Initialize the set to track unique switches
        switch_set = set()
        # Connect microgrids using a Fractal Tree structure
        if self.num_microgrids == 1:
            microgrids.append(MicroGrid(config, 0))
            self.net = microgrids[0].net
            self.index_map[0] = {
                'bus': {},
                'line': {},
                'trafo': {},
                'load': {},
                'gen': {},
                'ext_grid': {},
                'switch': {},
                'shunt': {},
                'storage': {},
                'solar_gen': {},
                'wind_gen': {}
            }
            # for 0th grid, set key and value to be the same and take it from the self.net
            self.index_map[0]['bus'] = {k: k for k in range(len(self.net.bus))}
            self.index_map[0]['line'] = {k: k for k in range(len(self.net.line))}
            self.index_map[0]['trafo'] = {k: k for k in range(len(self.net.trafo))}
            self.index_map[0]['load'] = {k: k for k in range(len(self.net.load))}
            self.index_map[0]['gen'] = {k: k for k in range(len(self.net.gen))}
            self.index_map[0]['ext_grid'] = {k: k for k in range(len(self.net.ext_grid))}
            self.index_map[0]['switch'] = {k: k for k in range(len(self.net.switch))}
            self.index_map[0]['shunt'] = {k: k for k in range(len(self.net.shunt))}
            self.index_map[0]['storage'] = {k: k for k in range(len(self.net.storage))}
            self.index_map[0]['solar_gen'] = {k: k for k in range(len(self.net.gen)) if
                                              self.net.gen.loc[k, 'name'].startswith("Solar_")}
            self.index_map[0]['wind_gen'] = {k: k for k in range(len(self.net.gen)) if
                                             self.net.gen.loc[k, 'name'].startswith("Wind_")}
            return microgrids, 0, switch_set  # Only one microgrid, no switches needed

        for i in range(self.num_microgrids):
            microgrid = MicroGrid(config, i)
            microgrids.append(microgrid)
            if i == 0:
                self.net = microgrid.net
                self.index_map[0] = {
                    'bus': {},
                    'line': {},
                    'trafo': {},
                    'load': {},
                    'gen': {},
                    'ext_grid': {},
                    'switch': {},
                    'shunt': {},
                    'storage': {},
                    'solar_gen': {},
                    'wind_gen': {}
                }
                # for 0th grid, set key and value to be the same and take it from the self.net
                self.index_map[0]['bus'] = { k: k for k in range(len(self.net.bus))}
                self.index_map[0]['line'] = { k: k for k in range(len(self.net.line))}
                self.index_map[0]['trafo'] = { k: k for k in range(len(self.net.trafo))}
                self.index_map[0]['load'] = { k: k for k in range(len(self.net.load))}
                self.index_map[0]['gen'] = { k: k for k in range(len(self.net.gen))}
                self.index_map[0]['ext_grid'] = { k: k for k in range(len(self.net.ext_grid))}
                self.index_map[0]['switch'] = { k: k for k in range(len(self.net.switch))}
                self.index_map[0]['shunt'] = { k: k for k in range(len(self.net.shunt))}
                self.index_map[0]['storage'] = { k: k for k in range(len(self.net.storage))}
                self.index_map[0]['solar_gen'] = { k: k for k in range(len(self.net.gen)) if self.net.gen['name'].startswith("Solar_")}
                self.index_map[0]['wind_gen'] = { k: k for k in range(len(self.net.gen)) if self.net.gen['name'].startswith("Wind_")}
            else:
                self._merge_networks(self.net, microgrid.net, i)
        for i in range(self.num_microgrids):
            left_child = 2 * i + 1  # Calculate index of left child
            right_child = 2 * i + 2  # Calculate index of right child

            microgrids[i].mg_id = i

            if left_child < self.num_microgrids:
                # Create a single bi-directional switch (consistent naming based on min/max index)
                switch_name = f"S_{min(i, left_child)}_to_{max(i, left_child)}"
                microgrids[i].add_neighbor(microgrids[left_child], switch_name)
                microgrids[left_child].add_neighbor(microgrids[i], switch_name)

                # Add the switch to the set (only if it hasn't been added before)
                switch_set.add(switch_name)
                self._create_tie_switch(i, left_child, self.V_ref)

            if right_child < self.num_microgrids:
                # Create a single bi-directional switch (consistent naming based on min/max index)
                switch_name = f"S_{min(i, right_child)}_to_{max(i, right_child)}"
                microgrids[i].add_neighbor(microgrids[right_child], switch_name)
                microgrids[right_child].add_neighbor(microgrids[i], switch_name)

                # Add the switch to the set (only if it hasn't been added before)
                switch_set.add(switch_name)

        total_switches = len(switch_set)
        return microgrids, total_switches, switch_set

    def _create_tie_switch(self, mg1_idx, mg2_idx, base_voltage):
        """
        Create a tie switch between two microgrids.

        Args:
            mg1_idx: Index of first microgrid
            mg2_idx: Index of second microgrid
            base_voltage: Base voltage for the connection (kV)
        """
        # Find appropriate buses to connect in each microgrid
        mg1 = self.microgrids[mg1_idx]
        mg2 = self.microgrids[mg2_idx]

        # Create buses for the tie connection
        bus1_idx = pp.create_bus(self.net, vn_kv=base_voltage, name=f"Tie Bus MG{mg1_idx}")
        bus2_idx = pp.create_bus(self.net, vn_kv=base_voltage, name=f"Tie Bus MG{mg2_idx}")

        # Connect these buses to their respective microgrids
        # In a real implementation, you'd need to connect to appropriate existing buses

        # Create a tie line between the two buses
        line_idx = pp.create_line(self.net, from_bus=bus1_idx, to_bus=bus2_idx,
                                  length_km=10, std_type="N2XS(FL)2Y 1x300 RM/35 64/110 kV",
                                  name=f"Tie Line MG{mg1_idx}-MG{mg2_idx}")
        self.tie_lines.append(line_idx)

        # Create a switch on the tie line (initially open)
        switch_idx = pp.create_switch(self.net, bus=bus1_idx, element=line_idx,
                                      et="l", type="LBS", closed=False,
                                      name=f"Tie Switch MG{mg1_idx}-MG{mg2_idx}")

        self.tie_switch_index_map [(mg1_idx, mg2_idx)] = switch_idx

        return switch_idx

    def operate_tie_switch(self, from_mg, to_mg, closed):
        """
        Open or close a tie switch.

        Args:
            switch_idx: Index of the switch to operate
            closed: Boolean indicating whether to close (True) or open (False) the switch
        """
        switch_idx = self.tie_switch_index_map[(from_mg, to_mg)]
        if switch_idx is not None:
            self.net.switch.at[switch_idx, 'closed'] = closed == 1

    def _merge_networks(self, main_net, new_net, mg_id):
        """
        Merge a new microgrid network into the main network by copying all elements
        from new_net to main_net with appropriate index mapping.

        Args:
            main_net: The main pandapower network to merge into
            new_net: The new pandapower network to merge from

        Returns:
            Dictionary of index mappings from new_net to main_net
        """
        # Dictionary to store the mapping between old and new indices
        index_map = self.index_map
        index_map[mg_id] = {
            'bus': {},
            'line': {},
            'trafo': {},
            'load': {},
            'gen': {},
            'ext_grid': {},
            'switch': {},
            'shunt': {},
            'storage': {}
        }

        # 1. Copy buses and store the index mapping
        for i, bus in new_net.bus.iterrows():
            # Create a copy of the bus parameters as a dictionary
            bus_data = bus.to_dict()
            # Remove the index since it will be auto-assigned in the main network
            if 'index' in bus_data:
                del bus_data['index']
            # Create new bus in main network
            new_idx = pp.create_bus(main_net, **bus_data)
            # Store the mapping
            index_map['bus'][i] = new_idx

        # 2. Copy lines
        for i, line in new_net.line.iterrows():
            line_data = line.to_dict()
            if 'index' in line_data:
                del line_data['index']

            # Update bus references using the mapping
            if 'from_bus' in line_data:
                line_data['from_bus'] = index_map['bus'][line_data['from_bus']]
            if 'to_bus' in line_data:
                line_data['to_bus'] = index_map['bus'][line_data['to_bus']]

            new_idx = pp.create_line(main_net, **line_data)
            index_map['line'][i] = new_idx

        # 3. Copy transformers
        if hasattr(new_net, 'trafo'):
            for i, trafo in new_net.trafo.iterrows():
                trafo_data = trafo.to_dict()
                if 'index' in trafo_data:
                    del trafo_data['index']

                # Update bus references
                if 'hv_bus' in trafo_data:
                    trafo_data['hv_bus'] = index_map['bus'][trafo_data['hv_bus']]
                if 'lv_bus' in trafo_data:
                    trafo_data['lv_bus'] = index_map['bus'][trafo_data['lv_bus']]

                new_idx = pp.create_transformer(main_net, **trafo_data)
                index_map['trafo'][i] = new_idx

        # 4. Copy loads
        if hasattr(new_net, 'load'):
            for i, load in new_net.load.iterrows():
                load_data = load.to_dict()
                if 'index' in load_data:
                    del load_data['index']

                # Update bus reference
                if 'bus' in load_data:
                    load_data['bus'] = index_map['bus'][load_data['bus']]

                new_idx = pp.create_load(main_net, **load_data)
                index_map['load'][i] = new_idx

        # 5. Copy generators
        if hasattr(new_net, 'gen'):
            for i, gen in new_net.gen.iterrows():
                gen_data = gen.to_dict()
                if 'index' in gen_data:
                    del gen_data['index']

                # Update bus reference
                if 'bus' in gen_data:
                    gen_data['bus'] = index_map['bus'][gen_data['bus']]

                new_idx = pp.create_gen(main_net, **gen_data)
                index_map['gen'][i] = new_idx
                if gen_data['name'].startswith("Solar_"):
                    index_map['solar_gen'][i] = new_idx
                elif gen_data['name'].startswith("Wind_"):
                    index_map['wind_gen'][i] = new_idx

        # 6. Copy external grids (with modified names to avoid conflicts)
        if hasattr(new_net, 'ext_grid'):
            for i, ext_grid in new_net.ext_grid.iterrows():
                ext_grid_data = ext_grid.to_dict()
                if 'index' in ext_grid_data:
                    del ext_grid_data['index']

                # Update bus reference
                if 'bus' in ext_grid_data:
                    ext_grid_data['bus'] = index_map['bus'][ext_grid_data['bus']]

                # Modify name to avoid conflicts
                if 'name' in ext_grid_data:
                    ext_grid_data['name'] = f"{ext_grid_data['name']}_merged"

                # For merged networks, only keep one external grid active
                # Set others as out of service
                ext_grid_data['in_service'] = False

                new_idx = pp.create_ext_grid(main_net, **ext_grid_data)
                index_map['ext_grid'][i] = new_idx

        # 7. Copy switches
        if hasattr(new_net, 'switch'):
            for i, switch in new_net.switch.iterrows():
                switch_data = switch.to_dict()
                if 'index' in switch_data:
                    del switch_data['index']

                # Update references based on element type
                if 'bus' in switch_data:
                    switch_data['bus'] = index_map['bus'][switch_data['bus']]

                if 'element' in switch_data and 'et' in switch_data:
                    et = switch_data['et']
                    if et == 'b':  # bus-bus switch
                        switch_data['element'] = index_map['bus'][switch_data['element']]
                    elif et == 'l':  # line switch
                        switch_data['element'] = index_map['line'][switch_data['element']]
                    elif et == 't':  # transformer switch
                        switch_data['element'] = index_map['trafo'][switch_data['element']]

                new_idx = pp.create_switch(main_net, **switch_data)
                index_map['switch'][i] = new_idx

        # 8. Copy shunts
        if hasattr(new_net, 'shunt'):
            for i, shunt in new_net.shunt.iterrows():
                shunt_data = shunt.to_dict()
                if 'index' in shunt_data:
                    del shunt_data['index']

                # Update bus reference
                if 'bus' in shunt_data:
                    shunt_data['bus'] = index_map['bus'][shunt_data['bus']]

                new_idx = pp.create_shunt(main_net, **shunt_data)
                index_map['shunt'][i] = new_idx

        # 9. Copy storage units
        if hasattr(new_net, 'storage'):
            for i, storage in new_net.storage.iterrows():
                storage_data = storage.to_dict()
                if 'index' in storage_data:
                    del storage_data['index']

                # Update bus reference
                if 'bus' in storage_data:
                    storage_data['bus'] = index_map['bus'][storage_data['bus']]

                new_idx = pp.create_storage(main_net, **storage_data)
                index_map['storage'][i] = new_idx

        return index_map

    def reset(self):
        self.current_step = 0
        self.done = False

        for mg in self.microgrids:
            mg.reset()
        # Update the power flow wrapper with current microgrids and tie-lines.
        self.pw_wrapper.reset_network(self.microgrids, self.tie_lines)
        return self._get_state()

    def check_power_deficit(self, net):

        # Total real power demand
        total_load = net.res_load.p_mw.sum()

        # Total real power generation from:
        gen_power = net.res_gen.p_mw[net.gen.p_mw > 0].sum()
        storage_discharge = net.res_storage.p_mw[net.res_storage.p_mw > 0].sum()

        total_supply = gen_power + storage_discharge

        # Net power shortfall (if positive)
        deficit = total_load - total_supply

        overloaded = deficit > 1e-4  # tolerance
        overloaded_amount = max(0, deficit)

        return overloaded, overloaded_amount

    def apply_load_p_mv_by_timestep(self, time_step, mg_id, episode_num=0):
        """
        Apply time-varying and episode-growing load with configurable disturbance percent.
        Modulates load based on time-of-day to encourage BESS charging and discharging behavior.
        """
        base_load = np.array(self.microgrids[mg_id].base_loads)
        buses = list(self.index_map[mg_id]['load'].values())

        oscillation = 1.0 + 0.1 * np.sin(time_step / 10.0)  # short-term
        drift = 1.0 + self.load_disturbance_percent  # long-term growth
        random_noise = np.random.normal(0, 0.02, size=len(base_load))

        # Temporal multiplier based on pricing window
        if time_step in self.peak_hours:
            temporal_multiplier = 1  # increase demand during peak to trigger BESS discharge
        elif time_step in self.off_peak_hours:
            temporal_multiplier = 0.4  # reduce demand to allow BESS charging
        else:
            temporal_multiplier = 0.6  # neutral during mid-day

        load_profile = drift * oscillation * base_load * (1 + random_noise) * temporal_multiplier
        self.net.load.loc[buses, "p_mw"] = load_profile


    def set_voltage_setpoint_for_der(self, voltage_setpoint, inverter_idx, mg_id):
        """
        Set the voltage setpoint for DERs.
        """

        der_gens = self.index_map[mg_id]['gen']
        self.net.gen.loc[der_gens[inverter_idx], "vm_pu"] = voltage_setpoint

    def step(self, tertiary_action, time_step):
        self.done = False
        # Apply dispatch commands to each microgrid.
        microgrid_actions = tertiary_action.get("microgrids", None)
        for i, mg in enumerate(self.microgrids):
            der_actions = microgrid_actions[i].get("der_actions", [0.0, 0.0, 0.0, 0.0])
            self.apply_load_p_mv_by_timestep(time_step, i)
            self.apply_dispatch(der_actions, time_step, i)
            battery_operation = microgrid_actions[i].get("battery_operation", None)
            self.apply_battery_operation(battery_operation, i)

        # Update tie-line statuses if provided.
        new_tie_lines = tertiary_action.get("tie_lines", None)
        if new_tie_lines is not None:
            self.tie_lines = new_tie_lines

        # Compute the economic cost across microgrids.
        econ_cost = self._calculate_economic_cost()

        # Compute the overall reward.
        reward = (-self.lambda_econ * econ_cost )

        self.current_step += 1
        if self.current_step >= self.max_steps:
            self.current_step = 0
            self.done = True

        next_state = self._get_state()
        info = {
            "econ_cost": econ_cost
        }

        # log progress
        print(f"Tertiary Step: {self.current_step}, Reward: {reward}, Done: {self.done}")

        return next_state, reward, self.done, info

    def simulate_bess_action(self, time_step):
        """
        Gradually choose battery action based on time of day and peak/off-peak hours.
        Creates smooth transitions between charging and discharging states.
        :param time_step: Current hour (0-23)
        :return: Battery action in range [-1, 1]
        """
        # Get peak and off-peak hours from config
        peak_hours = getattr(self.config, "peak_hours", [18, 19, 20, 21, 22])
        off_peak_hours = getattr(self.config, "off_peak_hours", [1, 2, 3, 4, 5, 6])
        
        # Initialize action
        bess_action = 0.0
        
        # Determine the optimal action based on time of day
        if time_step in peak_hours:
            # Peak hours: gradually increase discharging (negative action)
            # Find position within peak hours for gradual increase
            peak_duration = len(peak_hours)
            
            # Calculate position within peak hours (0 to peak_duration-1)
            peak_position = peak_hours.index(time_step)
            progress = (peak_position + 1) / peak_duration  # +1 to avoid 0 progress
            
            # Gradually increase discharging from -0.2 to -1.0
            bess_action = 0.2 + (progress * 0.8)
                
        elif time_step in off_peak_hours:
             return 0.0
        else:
            # Mid-day hours: gradual transition based on proximity to peak/off-peak
            # Find distance to nearest peak and off-peak periods
            distance_to_peak = min(abs(time_step - p) for p in peak_hours)
            distance_to_off_peak = min(abs(time_step - op) for op in off_peak_hours)
            
            if distance_to_peak < distance_to_off_peak:
                # Closer to peak hours, start mild discharging
                bess_action = -0.2
            else:
                # Closer to off-peak hours, start mild charging
                bess_action = -0.2
        
        # Ensure action is within bounds
        bess_action = max(-1.0, min(1.0, bess_action))
        
        return bess_action

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
        base_wind = 0.8  # Base wind availability
        fluctuation = np.random.uniform(-0.2, 0.2)  # +/-20% random fluctuation
        wind_availability = np.clip(base_wind + fluctuation, 0.0, 1.0)

        return solar_availability, wind_availability

    def apply_dispatch(self, der_actions, time_step, mg_id):
        """
        Apply individual dispatch commands to each DER.
        der_actions: list of configurable floats [solar1, solar2, ..., wind1, wind2, ...]
        """
        # Get available generation factors
        solar_availability, wind_availability = self.get_available_der_output(time_step)

        mg = self.microgrids[mg_id]
        
        # Get base outputs
        solar_base_output = getattr(self.config, "solar_base_output", 0.005)
        wind_base_output = getattr(self.config, "wind_base_output", 0.007)

        # Get DER bus indices
        cur_solar_buses = list(self.index_map[mg_id]['solar_gen'].values())
        cur_wind_buses = list(self.index_map[mg_id]['wind_gen'].values())

        # Get configurable DER counts
        num_der_solar = getattr(self.config, "num_der_solar", 2)
        num_der_wind = getattr(self.config, "num_der_wind", 2)

        # Apply individual DER actions
        # Solar DERs (first num_der_solar indices in der_actions)
        for i, solar_bus in enumerate(cur_solar_buses):
            if i < num_der_solar and i < len(der_actions):
                solar_action = der_actions[i]
                self.net.gen.loc[solar_bus, 'p_mw'] = abs(solar_action * solar_base_output * solar_availability)

        # Wind DERs (indices num_der_solar onwards in der_actions) 
        for i, wind_bus in enumerate(cur_wind_buses):
            wind_action_idx = num_der_solar + i
            if wind_action_idx < len(der_actions):
                wind_action = der_actions[wind_action_idx]
                self.net.gen.loc[wind_bus, 'p_mw'] = abs(wind_action * wind_base_output * wind_availability)


    def apply_battery_operation(self, battery_operations, mg_id):
        mg = self.microgrids[mg_id]
        if not isinstance(battery_operations, (list, tuple, np.ndarray)):
            battery_operations = [battery_operations]
        storage_idxs = self.index_map[mg_id]['storage']
        max_e_mwhs = [self.net.storage.loc[idx, "max_e_mwh"] for idx in storage_idxs]
        stored_energies = [mg.last_soc[i] * max_e_mwhs[i] for i in range(len(storage_idxs))]
        storage_bus_ids = mg.storage_bus_id if isinstance(mg.storage_bus_id, (list, tuple, np.ndarray)) else [mg.storage_bus_id]
        for i, (storage_idx, battery_operation) in enumerate(zip(storage_idxs, battery_operations)):
            max_e_mwh = max_e_mwhs[i]
            stored_energy_mwh = stored_energies[i]
            energy_change = -battery_operation * max_e_mwh
            new_energy = stored_energy_mwh + energy_change
            new_soc = new_energy / max_e_mwh
            if not (0.0 <= new_soc <= 1.0):
                print(f"Battery {i} SOC out of bounds:", new_soc)
                if new_soc < 0.0 and mg.last_soc[i] > 0.0:
                    print("Battery discharge reached limits, discharging remaining energy.")
                    new_soc = 0.0
                    self.net.storage.loc[storage_idx, "p_mw"] = -stored_energy_mwh
                    self.net.storage.loc[storage_idx, "soc_percent"] = 0.0
                    mg.last_soc[i] = 0.0
                elif new_soc > 1.0 and mg.last_soc[i] < 1.0:
                    print("Battery charge reached limits, charging remaining energy.")
                    new_soc = 1.0
                    mg.last_soc[i] = 1.0
                    self.net.storage.loc[storage_idx, "p_mw"] = max_e_mwh - stored_energy_mwh
                    self.net.storage.loc[storage_idx, "soc_percent"] = 100.0
                else:
                    print("Battery no more power to discharge or charge.")
                    self.net.storage.loc[storage_idx, "p_mw"] = 0.0
            else:
                self.net.storage.loc[storage_idx, "p_mw"] = -battery_operation * max_e_mwh
                self.net.storage.loc[storage_idx, "soc_percent"] = new_soc * 100
                mg.last_soc[i] = new_soc
            # Basic voltage support: absorb or inject reactive power based on voltage
            try:
                bus_storage_idx = storage_bus_ids[i] - 1  # Adjust for 0-based index
                bus_vm = self.net.res_bus.vm_pu[bus_storage_idx]
            except KeyError as e:
                bus_vm = self.net.bus.vn_kv[bus_storage_idx] / self.V_ref
            if bus_vm > 1.05:
                q_mvar = -0.5
            elif bus_vm < 0.95:
                q_mvar = 0.5
            else:
                q_mvar = 0.0
            self.net.storage.loc[storage_idx, "q_mvar"] = q_mvar
            print(f"Battery {i} operation applied, dispatching energy:", self.net.storage.loc[storage_idx, "p_mw"], "MW")

    def get_state_by_mg_id(self, mg_id):

        bess_soc = None
        total_load = None
        der_generation = None
        for mg in self.microgrids:
            if mg.mg_id == mg_id:
                load_indexes = list(self.index_map[mg_id]['load'].values())
                der_indexes = list(self.index_map[mg_id]['gen'].values())
                bess_soc = mg.this_soc
                total_load = self.net.load.loc[load_indexes, "p_mw"].sum()
                der_generation = self.net.gen.loc[der_indexes, "p_mw"].sum()
                break

        try:
            vals = list(self.index_map[mg_id]['gen'].values())
            gen_vm_pu = self.net.res_gen.vm_pu[vals].sum()
            avg_volt = float(gen_vm_pu) / len(vals)
            measured_voltage = avg_volt
        except KeyError as e:
            # print err
            print(f"KeyError: {e}")
            logging.error(f"Microgrid ID {mg_id} not found in index map.")
            measured_voltage = 5

        try:
            vals = list(self.index_map[mg_id]['ext_grid'].values())
            grid_power = self.net.res_ext_grid.p_mw[vals].sum()
        except KeyError as e:
            # print err
            print(f"KeyError: {e}")
            logging.error(f"External grid ID {mg_id} not found in index map.")
            grid_power = 100 # large value to let SAC know no convergence 

        state = {
            "bess_soc": bess_soc,
            "total_load": total_load,
            "der_generation": der_generation,
            "measured_voltage": measured_voltage,
            "grid_power": grid_power
        }
        return state

    def _get_state(self):
        mg_states = [self.get_state_by_mg_id(mg.mg_id) for mg in self.microgrids]

        # Get price forecast for current timestep
        price_forecast = self.config.get_price_forecast(self.current_step)
        current_prices = self.config.get_temporal_prices(self.current_step)
        
        # Combine current prices with forecast
        price_info = {
            "current_buy_price": current_prices["buy_price"],
            "current_sell_price": current_prices["sell_price"],
            "buy_price_forecast": price_forecast["buy_price_forecast"],
            "sell_price_forecast": price_forecast["sell_price_forecast"]
        }

        state = {
            "microgrids": mg_states,
            "timestep": self.current_step
        }
        return state

    def _calculate_economic_cost(self):
        total_cost = 0.0

        # PV & Wind via gen
        if self.net is not None and "gen" in self.net and not self.net.gen.empty:
            gen_data = self.net.gen

            for i, row in gen_data.iterrows():
                power = row["p_mw"]
                total_cost += 0.01 * power

        # BESS via storage
        if self.net is not None and "storage" in self.net and not self.net.storage.empty:
            for _, row in self.net.storage.iterrows():
                power = abs(row["p_mw"])  # both charging/discharging = degradation
                total_cost += 0.01 * power

        return total_cost

# --- Testing the FractalGridEnv independently ---
if __name__ == "__main__":
    class Config:
        num_microgrids = 2
        tie_lines = [(0, 1, 1)]
        V_ref = 1.0
        max_steps = 10
        lambda_econ = 1.0
        alpha_sec = 0.5
        beta_volt = 1.0

        # Load parameters
        base_load = 50.0
        load_variability = 0.1
        load_cost_factor = 0.05

        # Solar parameters
        num_solar = 2
        solar_buses = [5, 10]
        solar_base_output = 0.005

        # Wind parameters
        num_wind = 1
        wind_buses = [15]
        wind_base_output = 0.007

        # BESS parameters
        bess_capacity = 100.0
        bess_max_charge = 20.0
        bess_max_discharge = 20.0
        bess_charge_efficiency = 0.95
        bess_discharge_efficiency = 0.95
        bess_initial_soc = 0.5

        seed = 42


    config = Config()
    env = FractalGridEnv(config)
    state = env.reset()
    print("Initial state:")
    print(state)
    # Define a dummy tertiary action
    tertiary_action = {
        "dispatch": [10.0, -5.0],  # Dispatch commands for each microgrid.
        "tie_lines": [(0, 1, 1)]
    }
    next_state, reward, done, info = env.step(tertiary_action)
    print("\nNext state:")
    print(next_state)
    print("\nReward:", reward)
    print("\nDone:", done)
    print("\nInfo:", info)