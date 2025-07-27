import pandapower as pp
import pandapower.networks as pn
import random
import numpy as np

class der_24_ieee():
    NUM_DER_TOTAL = 10
    NUM_BESS_TOTAL = 6

    def __init__(self):
        self.net = pp.create_empty_network()
        self.storage_idxs = []
        self.wind_buses = []
        self.solar_buses = []
        self.combine_bus_inv_idx = []
        self.num_buses = 16
        self.num_secondary_agents = 20
        self.bess_bus_ids = [4, 8, 12, 14, 18]
        self.build_network()

    def get_network(self):
        return self.net

    def build_network(self):
        net = self.net

        buses = [pp.create_bus(net, name=f"Bus {i+1}", vn_kv=110 if i < 4 else 20, type="b") for i in range(24)]

        pp.create_ext_grid(net, buses[0], vm_pu=1.02, va_degree=50)

        pp.create_transformer(net, buses[2], buses[13], name="T1", std_type="25 MVA 110/20 kV")
        pp.create_line(net, buses[11], buses[23], length_km=2, std_type="NA2XS2Y 1x240 RM/25 12/20 kV")

        # Main lines
        pp.create_line(net, buses[0], buses[1], length_km=10, std_type="N2XS(FL)2Y 1x300 RM/35 64/110 kV")
        pp.create_line(net, buses[1], buses[2], length_km=8, std_type="N2XS(FL)2Y 1x300 RM/35 64/110 kV")
        pp.create_line(net, buses[2], buses[3], length_km=10, std_type="N2XS(FL)2Y 1x300 RM/35 64/110 kV")
        pp.create_transformer(net, buses[3], buses[4], std_type="25 MVA 110/20 kV", name="T6")
        pp.create_line(net, buses[4], buses[5], length_km=0.8, std_type="NA2XS2Y 1x240 RM/25 12/20 kV")
        pp.create_line(net, buses[5], buses[6], length_km=2, std_type="NA2XS2Y 1x240 RM/25 12/20 kV")
        pp.create_line(net, buses[6], buses[7], length_km=0.8, std_type="NA2XS2Y 1x240 RM/25 12/20 kV")
        pp.create_line(net, buses[7], buses[8], length_km=1, std_type="NA2XS2Y 1x240 RM/25 12/20 kV")
        pp.create_line(net, buses[8], buses[9], length_km=2, std_type="NA2XS2Y 1x240 RM/25 12/20 kV")
        pp.create_line(net, buses[9], buses[10], length_km=1, std_type="NA2XS2Y 1x240 RM/25 12/20 kV")
        pp.create_line(net, buses[10], buses[11], length_km=3, std_type="NA2XS2Y 1x240 RM/25 12/20 kV")
        pp.create_line(net, buses[11], buses[12], length_km=2, std_type="NA2XS2Y 1x240 RM/25 12/20 kV")

        # Cross connections
        pp.create_transformer(net, buses[2], buses[5], std_type="25 MVA 110/20 kV", name="T5")
        pp.create_line(net, buses[4], buses[7], length_km=3, std_type="NA2XS2Y 1x240 RM/25 12/20 kV")
        pp.create_line(net, buses[6], buses[9], length_km=2, std_type="NA2XS2Y 1x240 RM/25 12/20 kV")
        pp.create_line(net, buses[8], buses[11], length_km=2.5, std_type="NA2XS2Y 1x240 RM/25 12/20 kV")

        # Distribution lines
        pp.create_line(net, buses[13], buses[14], length_km=2, std_type="NA2XS2Y 1x240 RM/25 12/20 kV")
        pp.create_line(net, buses[14], buses[15], length_km=2, std_type="NA2XS2Y 1x240 RM/25 12/20 kV")
        pp.create_line(net, buses[15], buses[16], length_km=2, std_type="NA2XS2Y 1x240 RM/25 12/20 kV")
        pp.create_line(net, buses[16], buses[17], length_km=2, std_type="NA2XS2Y 1x240 RM/25 12/20 kV")
        pp.create_line(net, buses[18], buses[19], length_km=2, std_type="NA2XS2Y 1x240 RM/25 12/20 kV")
        pp.create_line(net, buses[19], buses[20], length_km=2, std_type="NA2XS2Y 1x240 RM/25 12/20 kV")
        pp.create_line(net, buses[20], buses[21], length_km=2, std_type="NA2XS2Y 1x240 RM/25 12/20 kV")
        pp.create_line(net, buses[21], buses[22], length_km=2, std_type="NA2XS2Y 1x240 RM/25 12/20 kV")
        pp.create_line(net, buses[22], buses[23], length_km=2, std_type="NA2XS2Y 1x240 RM/25 12/20 kV")

        # Loads
        load_buses = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 13, 15, 17, 19, 21]
        for i, bus_idx in enumerate(load_buses):
            if i % 2 == 0:
                pp.create_load(net, buses[bus_idx], p_mw=8, q_mvar=4, scaling=1.0, name=f"Load_{i+1}")
            else:
                pp.create_load(net, buses[bus_idx], p_mw=10, q_mvar=4, scaling=1.0, name=f"Load_{i+1}")
        # Solar and wind (standardize vm_pu=1.02)
        solar_gen_buses = [2, 4, 6, 8, 10, 12, 14, 16]
        wind_gen_buses = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
        for i, bus_idx in enumerate(solar_gen_buses):
            pp.create_gen(net, buses[bus_idx], p_mw=15, max_q_mvar=10, name=f"Solar_{i+1}", min_q_mvar=-4, vm_pu=1.03)
            self.solar_buses.append(buses[bus_idx])
            self.combine_bus_inv_idx.append(buses[bus_idx])
        for i, bus_idx in enumerate(wind_gen_buses):
            pp.create_gen(net, buses[bus_idx], p_mw=20, max_q_mvar=10, name=f"Wind_{i+1}", min_q_mvar=-4, vm_pu=1.03)
            self.wind_buses.append(buses[bus_idx])
            self.combine_bus_inv_idx.append(buses[bus_idx])

        self.combine_bus_inv_idx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 21, 23]

        # BESS
        bess_buses = [4, 8, 12, 16, 20]
        for i, bus_idx in enumerate(bess_buses):
            storage_idx = pp.create_storage(
                net, bus=buses[bus_idx], p_mw=20, max_e_mwh=20.0, min_e_mwh=5,
                soc_percent=50, name=f"storage_{i}", max_p_mw=20, min_p_mw=-20,
                initial_e_mwh=15, q_mvar=0.2
            )
            self.storage_idxs.append(storage_idx)

        # Line switches for key lines
        line_idx_45 = net.line[((net.line.from_bus == buses[4]) & (net.line.to_bus == buses[5])) | ((net.line.from_bus == buses[5]) & (net.line.to_bus == buses[4]))].index[0]
        pp.create_switch(net, buses[4], line_idx_45, et="l", type="LBS", closed=True)

        line_idx_89 = net.line[((net.line.from_bus == buses[8]) & (net.line.to_bus == buses[9])) | ((net.line.from_bus == buses[9]) & (net.line.to_bus == buses[8]))].index[0]
        pp.create_switch(net, buses[8], line_idx_89, et="l", type="LBS", closed=True)

        line_idx_1415 = net.line[((net.line.from_bus == buses[14]) & (net.line.to_bus == buses[15])) | ((net.line.from_bus == buses[15]) & (net.line.to_bus == buses[14]))].index[0]
        pp.create_switch(net, buses[14], line_idx_1415, et="l", type="LBS", closed=True)

        line_idx_2021 = net.line[((net.line.from_bus == buses[20]) & (net.line.to_bus == buses[21])) | ((net.line.from_bus == buses[21]) & (net.line.to_bus == buses[20]))].index[0]
        pp.create_switch(net, buses[20], line_idx_2021, et="l", type="LBS", closed=True)

        # Shunt
        pp.create_shunt(net, buses[2], q_mvar=-0.96, p_mw=0, name='Shunt')

        # Load the original network
        # net = pn.case89pegase()
        #
        # # 1. Remove all sgens
        # # net.sgen.drop(index=net.sgen.index, inplace=True)
        #
        # bess_buses = [4, 24, 32, 66, 80]
        # for i, bus_idx in enumerate(bess_buses):
        #     storage_idx = pp.create_storage(
        #         net, bus=bus_idx, p_mw=-50, max_e_mwh=50.0, min_e_mwh=15,
        #         soc_percent=50, name=f"storage_{i}", max_p_mw=50, min_p_mw=-50,
        #         initial_e_mwh=0.5, q_mvar=0.2
        #     )
        #     self.storage_idxs.append(storage_idx)
        #
        # self.combine_bus_inv_idx = [bus for bus in net.gen.bus]
        #
        # # Assume `net` is your pandapower network
        # gen_indices = list(net.gen.index)
        #
        # # Randomly select 4 indices for Solar
        # solar_indices = random.sample(gen_indices, 4)
        #
        # # The rest are Wind
        # wind_indices = [idx for idx in gen_indices if idx not in solar_indices]
        #
        # # Assign Solar names
        # for i, idx in enumerate(solar_indices, 1):
        #     net.gen.at[idx, "name"] = f"Solar_{i}"
        #
        # # Assign Wind names
        # for i, idx in enumerate(wind_indices, 1):
        #     net.gen.at[idx, "name"] = f"Wind_{i}"
        #
        # self.net = net



    def get_storage_idxs(self):
        return self.storage_idxs

    def run_network(self):
        try:
            pp.runpp(self.net, algorithm="nr", max_iteration=50, calculate_voltage_angles=True, tolerance_mva=1e-2, enforce_q_lims=True)
        except:
            print("Error running power flow")
        print("Power flow calculation was successful.")
        print("Results:")
        print(self.net.res_ext_grid)
        print(self.net.res_bus.vm_pu)

if __name__ == "__main__":
    # Load the original network
    net = pn.case89pegase()

    # 1. Remove all sgens
    #net.sgen.drop(index=net.sgen.index, inplace=True)

    bess_buses = [4, 24, 32, 66, 80]
    for i, bus_idx in enumerate(bess_buses):
        storage_idx = pp.create_storage(
            net, bus=bus_idx, p_mw=-100, max_e_mwh=100.0, min_e_mwh=15,
            soc_percent=50, name=f"storage_{i}", max_p_mw=100, min_p_mw=-100,
            initial_e_mwh=0.5, q_mvar=0.2
        )

    # Assume `net` is your pandapower network
    gen_indices = list(net.gen.index)

    # Randomly select 4 indices for Solar
    solar_indices = random.sample(gen_indices, 4)

    # The rest are Wind
    wind_indices = [idx for idx in gen_indices if idx not in solar_indices]

    # Assign Solar names
    for i, idx in enumerate(solar_indices, 1):
        net.gen.at[idx, "name"] = f"Solar_{i}"

    # Assign Wind names
    for i, idx in enumerate(wind_indices, 1):
        net.gen.at[idx, "name"] = f"Wind_{i}"

    try:
        pp.runpp(net)
        print("Power flow ran successfully.")
    except Exception as e:
        print("Power flow failed:", e)

    # Show updated generator info
    print(f"\nTotal generators now: {len(net.gen)}")
    print(net.gen[['bus', 'p_mw', 'name']])
    try:
        pp.runpp(net, algorithm="nr", max_iteration=50, calculate_voltage_angles=True, tolerance_mva=1e-2,
                 enforce_q_lims=True)
        print("Power flow calculation was successful.")
        print("Results:")
        print(net.res_ext_grid)
        print(net.res_bus.vm_pu)
    except:
        print("Error running power flow")
    print(net)