import pandapower as pp

class der_4():
    def __init__(self, bess_bus_id=6):
        """
        :param bess_bus_id: Integer index for MV bus (must be 4–8) where BESS should be placed
        """
        self.net = pp.create_empty_network()
        self.storage_idx = None
        self.wind_buses = [6, 7]
        self.combine_bus_inv_idx = [4, 5, 6, 7]
        self.num_buses = 5
        self.num_secondary_agents = 4
        self.solar_buses = [4, 5]
        self.bess_bus_id = bess_bus_id  # Store desired BESS location
        self.build_network()

    def get_network(self):
        return self.net

    def build_network(self):
        net = self.net

        # Create buses
        buses = [pp.create_bus(net, name=f"Bus {i+1}", vn_kv=110 if i < 4 else 20, type="b") for i in range(8)]
        bus1, bus2, bus3, bus4, bus5, bus6, bus7, bus8 = buses

        # External grid and transformer
        pp.create_ext_grid(net, bus1, vm_pu=1.02, va_degree=50)
        pp.create_transformer(net, bus3, bus4, name="110kV/20kV transformer", std_type="25 MVA 110/20 kV")

        # Lines
        pp.create_line(net, bus1, bus2, length_km=10, std_type="N2XS(FL)2Y 1x300 RM/35 64/110 kV")
        pp.create_line(net, bus5, bus6, length_km=2.0, std_type="NA2XS2Y 1x240 RM/25 12/20 kV")
        pp.create_line(net, bus6, bus7, length_km=3.5, std_type="48-AL1/8-ST1A 20.0")
        pp.create_line(net, bus7, bus5, length_km=2.5, std_type="NA2XS2Y 1x240 RM/25 12/20 kV")
        pp.create_line(net, bus7, bus8, length_km=0.5, std_type="NA2XS2Y 1x240 RM/25 12/20 kV")

        # Switches
        pp.create_switch(net, bus2, bus3, et="b", type="CB", closed=True)
        pp.create_switch(net, bus4, bus5, et="b", type="CB", closed=True)
        pp.create_switch(net, bus5, net.line.index[1], et="l", type="LBS", closed=True)
        pp.create_switch(net, bus6, net.line.index[1], et="l", type="LBS", closed=True)
        pp.create_switch(net, bus6, net.line.index[2], et="l", type="LBS", closed=True)
        pp.create_switch(net, bus7, net.line.index[2], et="l", type="LBS", closed=True)
        pp.create_switch(net, bus7, net.line.index[3], et="l", type="LBS", closed=True)
        pp.create_switch(net, bus5, net.line.index[3], et="l", type="LBS", closed=True)

        # Loads
        pp.create_load(net, bus7, p_mw=8, q_mvar=4, scaling=1.0)
        pp.create_load(net, bus6, p_mw=8, q_mvar=4, scaling=1.0)
        pp.create_load(net, bus5, p_mw=3, q_mvar=4, scaling=1.0)
        pp.create_load(net, bus4, p_mw=3, q_mvar=4, scaling=1.0)
        pp.create_load(net, bus3, p_mw=4, q_mvar=4, scaling=1.0)

        # Generators (DERs)
        pp.create_gen(net, bus5, p_mw=20, max_q_mvar=3, name="Solar_4", min_q_mvar=-3, vm_pu=1.03)
        pp.create_gen(net, bus6, p_mw=20, max_q_mvar=3, name="Solar_5", min_q_mvar=-3, vm_pu=1.03)
        pp.create_gen(net, bus7, p_mw=50, max_q_mvar=3, name="Wind_6", min_q_mvar=-3, vm_pu=1.03)
        pp.create_gen(net, bus8, p_mw=50, max_q_mvar=3, name="Wind_7", min_q_mvar=-3, vm_pu=1.03)

        # Dynamically place BESS at the requested MV bus (bus4 to bus8 → index 3 to 7)
        assert 4 <= self.bess_bus_id <= 8, "BESS bus must be in range 4 to 8"
        bess_bus = buses[self.bess_bus_id - 1]
        self.storage_idx = pp.create_storage(
            net, bus=bess_bus, p_mw=20, max_e_mwh=50.0, min_e_mwh=15,
            soc_percent=50, name="storage", max_p_mw=50, min_p_mw=-50,
            initial_e_mwh=0.5, q_mvar=0.2
        )

        # Shunt
        pp.create_shunt(net, bus3, q_mvar=-0.96, p_mw=0, name='Shunt')

    def get_storage_idx(self):
        return self.storage_idx

    def get_tie_switch(self, net):
        bus1 = pp.create_bus(self.net, vn_kv=110, name="Tie Bus 1")
        bus2 = pp.create_bus(net, vn_kv=110, name="Tie Bus 2")
        line = pp.create_line(net, bus1, bus2, length_km=10, std_type="N2XS(FL)2Y 1x300 RM/35 64/110 kV")
        switch = pp.create_switch(net, bus1, line, et="l", type="LBS", closed=True)
        return switch