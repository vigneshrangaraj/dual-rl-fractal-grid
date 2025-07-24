import pandapower as pp

class der_20():
    NUM_DER_TOTAL = 20
    NUM_BESS_TOTAL = 5
    def __init__(self):
        self.net = pp.create_empty_network()
        self.storage_idxs = []
        self.wind_buses = []
        self.solar_buses = []
        self.combine_bus_inv_idx = []
        self.num_buses = 18
        self.num_secondary_agents = 20
        self.bess_bus_ids = [4, 5, 6, 7, 8]
        self.build_network()

    def get_network(self):
        return self.net

    def build_network(self):
        net = self.net
        # 1. Create 24 buses
        buses = [pp.create_bus(net, name=f"Bus {i+1}", vn_kv=110 if i < 4 else 20, type="b") for i in range(24)]

        # 2. External grid and transformer
        pp.create_ext_grid(net, buses[0], vm_pu=1.02, va_degree=50)
        pp.create_transformer(net, buses[2], buses[4], name="110kV/20kV transformer", std_type="25 MVA 110/20 kV")

        # 3. Lines (ring and mesh, explicit)
        # Main ring
        pp.create_line(net, buses[4], buses[5], length_km=2.0, std_type="NA2XS2Y 1x240 RM/25 12/20 kV")
        pp.create_line(net, buses[5], buses[6], length_km=2.0, std_type="NA2XS2Y 1x240 RM/25 12/20 kV")
        pp.create_line(net, buses[6], buses[7], length_km=2.0, std_type="NA2XS2Y 1x240 RM/25 12/20 kV")
        pp.create_line(net, buses[7], buses[8], length_km=2.0, std_type="NA2XS2Y 1x240 RM/25 12/20 kV")
        pp.create_line(net, buses[8], buses[9], length_km=2.0, std_type="NA2XS2Y 1x240 RM/25 12/20 kV")
        pp.create_line(net, buses[9], buses[10], length_km=2.0, std_type="NA2XS2Y 1x240 RM/25 12/20 kV")
        pp.create_line(net, buses[10], buses[11], length_km=2.0, std_type="NA2XS2Y 1x240 RM/25 12/20 kV")
        pp.create_line(net, buses[11], buses[12], length_km=2.0, std_type="NA2XS2Y 1x240 RM/25 12/20 kV")
        pp.create_line(net, buses[12], buses[13], length_km=2.0, std_type="NA2XS2Y 1x240 RM/25 12/20 kV")
        pp.create_line(net, buses[13], buses[14], length_km=2.0, std_type="NA2XS2Y 1x240 RM/25 12/20 kV")
        pp.create_line(net, buses[14], buses[15], length_km=2.0, std_type="NA2XS2Y 1x240 RM/25 12/20 kV")
        pp.create_line(net, buses[15], buses[16], length_km=2.0, std_type="NA2XS2Y 1x240 RM/25 12/20 kV")
        pp.create_line(net, buses[16], buses[17], length_km=2.0, std_type="NA2XS2Y 1x240 RM/25 12/20 kV")
        pp.create_line(net, buses[17], buses[18], length_km=2.0, std_type="NA2XS2Y 1x240 RM/25 12/20 kV")
        pp.create_line(net, buses[18], buses[19], length_km=2.0, std_type="NA2XS2Y 1x240 RM/25 12/20 kV")
        pp.create_line(net, buses[19], buses[20], length_km=2.0, std_type="NA2XS2Y 1x240 RM/25 12/20 kV")
        pp.create_line(net, buses[20], buses[21], length_km=2.0, std_type="NA2XS2Y 1x240 RM/25 12/20 kV")
        pp.create_line(net, buses[21], buses[22], length_km=2.0, std_type="NA2XS2Y 1x240 RM/25 12/20 kV")
        pp.create_line(net, buses[22], buses[23], length_km=2.0, std_type="NA2XS2Y 1x240 RM/25 12/20 kV")
        pp.create_line(net, buses[23], buses[4], length_km=2.0, std_type="NA2XS2Y 1x240 RM/25 12/20 kV")
        # Mesh lines
        pp.create_line(net, buses[6], buses[12], length_km=3.0, std_type="NA2XS2Y 1x240 RM/25 12/20 kV")
        pp.create_line(net, buses[10], buses[18], length_km=3.0, std_type="NA2XS2Y 1x240 RM/25 12/20 kV")
        pp.create_line(net, buses[15], buses[21], length_km=3.0, std_type="NA2XS2Y 1x240 RM/25 12/20 kV")

        # 4. Loads (18 loads at buses 4-21)
        pp.create_load(net, buses[4], p_mw=6, q_mvar=3, scaling=1.0)
        pp.create_load(net, buses[5], p_mw=6, q_mvar=3, scaling=1.0)
        pp.create_load(net, buses[6], p_mw=6, q_mvar=3, scaling=1.0)
        pp.create_load(net, buses[7], p_mw=6, q_mvar=3, scaling=1.0)
        pp.create_load(net, buses[8], p_mw=6, q_mvar=3, scaling=1.0)
        pp.create_load(net, buses[9], p_mw=6, q_mvar=3, scaling=1.0)
        pp.create_load(net, buses[10], p_mw=6, q_mvar=3, scaling=1.0)
        pp.create_load(net, buses[11], p_mw=6, q_mvar=3, scaling=1.0)
        pp.create_load(net, buses[12], p_mw=6, q_mvar=3, scaling=1.0)
        pp.create_load(net, buses[13], p_mw=6, q_mvar=3, scaling=1.0)
        pp.create_load(net, buses[14], p_mw=6, q_mvar=3, scaling=1.0)
        pp.create_load(net, buses[15], p_mw=6, q_mvar=3, scaling=1.0)
        pp.create_load(net, buses[16], p_mw=6, q_mvar=3, scaling=1.0)
        pp.create_load(net, buses[17], p_mw=6, q_mvar=3, scaling=1.0)
        pp.create_load(net, buses[18], p_mw=6, q_mvar=3, scaling=1.0)
        pp.create_load(net, buses[19], p_mw=6, q_mvar=3, scaling=1.0)
        pp.create_load(net, buses[20], p_mw=6, q_mvar=3, scaling=1.0)
        pp.create_load(net, buses[21], p_mw=6, q_mvar=3, scaling=1.0)

        # 5. DERs (20 DERs, alternate solar/wind, spread out)
        der_bus_indices = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
        for i, bus_idx in enumerate(der_bus_indices):
            if i % 2 == 0:
                # Solar
                pp.create_gen(net, buses[bus_idx], p_mw=10, max_q_mvar=3, name=f"Solar_{i+1}", min_q_mvar=-3, vm_pu=1.03)
                self.solar_buses.append(buses[bus_idx])
            else:
                # Wind
                pp.create_gen(net, buses[bus_idx], p_mw=15, max_q_mvar=3, name=f"Wind_{i+1}", min_q_mvar=-3, vm_pu=1.03)
                self.wind_buses.append(buses[bus_idx])
            self.combine_bus_inv_idx.append(buses[bus_idx])

        # 6. BESS (5 BESS at buses 6, 10, 14, 18, 22)
        bess_bus_indices = [6, 10, 14, 18, 22]
        for i, bus_idx in enumerate(bess_bus_indices):
            storage_idx = pp.create_storage(
                net, buses[bus_idx], p_mw=20, max_e_mwh=50.0, min_e_mwh=15,
                soc_percent=50, name=f"storage_{i+1}", max_p_mw=50, min_p_mw=-50,
                initial_e_mwh=0.5, q_mvar=0.2
            )
            self.storage_idxs.append(storage_idx)

        # 7. Switches (example: connect some buses with circuit breakers)
        pp.create_switch(net, buses[5], buses[6], et="b", type="CB", closed=True)
        pp.create_switch(net, buses[10], buses[11], et="b", type="CB", closed=True)
        pp.create_switch(net, buses[15], buses[16], et="b", type="CB", closed=True)
        pp.create_switch(net, buses[20], buses[21], et="b", type="CB", closed=True)
        pp.create_switch(net, buses[23], buses[4], et="b", type="CB", closed=True)

        # 8. Shunt for voltage support
        pp.create_shunt(net, buses[2], q_mvar=-0.96, p_mw=0, name='Shunt')

    def get_storage_idxs(self):
        return self.storage_idxs 

    def plot_network(self, show_plot=True):
        import pandapower.plotting as plot
        plot.simple_plot(
            self.net,
            respect_switches=True,
            plot_loads=True,
            plot_sgens=True,
            plot_line_switches=True,
            show_plot=show_plot,
            bus_size=1.5,
            line_width=2.0,
            trafo_size=1.5,
            ext_grid_size=1.5,
            switch_size=2.0,
            load_size=1.2,
            sgen_size=1.2,
            bus_color='b',
            line_color='grey',
            trafo_color='k',
            ext_grid_color='y',
            switch_color='k'
        ) 