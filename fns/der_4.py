import pandapower as pp

class der_4():

    def __init__(self):
        self.net = pp.create_empty_network()
        self.storage_idx = None
        self.wind_buses = [6, 7]  # Bus location for wind DER
        self.combine_bus_inv_idx = [4, 5, 6, 7]
        self.num_buses = 5
        self.num_secondary_agents = 4
        self.solar_buses = [4, 5]  # Bus locations where solar DERs are connected
        self.build_network()

    def get_network(self):
        return self.net

    def build_network(self):
        net = self.net
        bus1 = pp.create_bus(net, name="HV Busbar", vn_kv=110, type="b")
        bus2 = pp.create_bus(net, name="HV Busbar 2", vn_kv=110, type="b")
        bus3 = pp.create_bus(net, name="HV Transformer Bus", vn_kv=110, type="n")
        bus4 = pp.create_bus(net, name="MV Transformer Bus", vn_kv=20, type="n")
        bus5 = pp.create_bus(net, name="MV Main Bus", vn_kv=20, type="b")
        bus6 = pp.create_bus(net, name="MV Bus 1", vn_kv=20, type="b")
        bus7 = pp.create_bus(net, name="MV Bus 2", vn_kv=20, type="b")
        bus8 = pp.create_bus(net, name="MV Bus 3", vn_kv=20, type="b")

        pp.create_ext_grid(net, bus1, vm_pu=1.02, va_degree=50)  # Create an external grid connection

        trafo1 = pp.create_transformer(net, bus3, bus4, name="110kV/20kV transformer", std_type="25 MVA 110/20 kV")

        line1 = pp.create_line(net, bus1, bus2, length_km=10, std_type="N2XS(FL)2Y 1x300 RM/35 64/110 kV", name="Line 1")
        line2 = pp.create_line(net, bus5, bus6, length_km=2.0, std_type="NA2XS2Y 1x240 RM/25 12/20 kV", name="Line 2")
        line3 = pp.create_line(net, bus6, bus7, length_km=3.5, std_type="48-AL1/8-ST1A 20.0", name="Line 3")
        line4 = pp.create_line(net, bus7, bus5, length_km=2.5, std_type="NA2XS2Y 1x240 RM/25 12/20 kV", name="Line 4")
        line5 = pp.create_line(net, bus7, bus8, length_km=0.5, std_type="NA2XS2Y 1x240 RM/25 12/20 kV", name="Line 5")

        sw1 = pp.create_switch(net, bus2, bus3, et="b", type="CB", closed=True)
        sw2 = pp.create_switch(net, bus4, bus5, et="b", type="CB", closed=True)
        #
        #
        sw3 = pp.create_switch(net, bus5, line2, et="l", type="LBS", closed=True)
        sw4 = pp.create_switch(net, bus6, line2, et="l", type="LBS", closed=True)
        sw5 = pp.create_switch(net, bus6, line3, et="l", type="LBS", closed=True)
        sw6 = pp.create_switch(net, bus7, line3, et="l", type="LBS", closed=True)
        sw7 = pp.create_switch(net, bus7, line4, et="l", type="LBS", closed=True)
        sw8 = pp.create_switch(net, bus5, line4, et="l", type="LBS", closed=True)

        pp.create_load(net, bus7, p_mw=5, q_mvar=4, scaling=0.6, name="load")
        pp.create_load(net, bus6, p_mw=5, q_mvar=4, scaling=0.6, name="load")
        pp.create_load(net, bus5, p_mw=2, q_mvar=4, scaling=0.6, name="load")
        pp.create_load(net, bus4, p_mw=2, q_mvar=4, scaling=0.6, name="load")
        pp.create_load(net, bus3, p_mw=2, q_mvar=4, scaling=0.6, name="load")

        pp.create_gen(net, bus5,  p_mw=20, max_q_mvar=3, name="Solar_4", min_q_mvar=-3, vm_pu=1.03)
        pp.create_gen(net, bus6, p_mw=20, max_q_mvar=3, name='Solar_5', min_q_mvar=-3, vm_pu=1.03)
        pp.create_gen(net, bus7, p_mw=50, max_q_mvar=3, name='Wind_6', min_q_mvar=-3, vm_pu=1.03)
        pp.create_gen(net, bus8, p_mw=50, max_q_mvar=3, name='Wind_7', min_q_mvar=-3, vm_pu=1.03)

        self.storage_idx = pp.create_storage(net, bus=bus6, p_mw=20, max_e_mwh=50.0, min_e_mwh=15, soc_percent=50, name="storage",
                          max_p_mw=50, min_p_mw=-50, initial_e_mwh=0.5, q_mvar=0.2)

        pp.create_shunt(net, bus3, q_mvar=-0.96, p_mw=0, name='Shunt')

    def get_storage_idx(self):
        return self.storage_idx

    def get_tie_switch(self, net):
        '''
        Create a tie switch between any bus and connect the other net to it
        :param self:
        :return:
        '''
        # Create a new bus in the first network
        bus1 = pp.create_bus(self.net, vn_kv=110, name="Tie Bus 1")
        # Create a new bus in the second network
        bus2 = pp.create_bus(net, vn_kv=110, name="Tie Bus 2")
        # Create a new line between the two buses
        line = pp.create_line(net, bus1, bus2, length_km=10, std_type="N2XS(FL)2Y 1x300 RM/35 64/110 kV", name="Tie Line")
        # Create a new switch to connect the two networks
        switch = pp.create_switch(net, bus1, line, et="l", type="LBS", closed=True)
        return switch