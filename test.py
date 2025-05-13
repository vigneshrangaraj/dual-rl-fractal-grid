import pandapower as pp #import pandapower

net = pp.create_empty_network() #create an empty network

bus1 = pp.create_bus(net, name="HV Busbar", vn_kv=110, type="b")
bus2 = pp.create_bus(net, name="HV Busbar 2", vn_kv=110, type="b")
bus3 = pp.create_bus(net, name="HV Transformer Bus", vn_kv=110, type="b")
bus4 = pp.create_bus(net, name="MV Transformer Bus", vn_kv=20, type="b")
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

sw1 = pp.create_switch(net, bus2, bus3, et="b", type="CB", closed=True)
sw2 = pp.create_switch(net, bus4, bus5, et="b", type="CB", closed=True)
#
#
sw3 = pp.create_switch(net, bus5, line2, et="l", type="LBS", closed=True)
sw4 = pp.create_switch(net, bus6, line2, et="l", type="LBS", closed=True)
sw5 = pp.create_switch(net, bus6, line3, et="l", type="LBS", closed=True)
sw6 = pp.create_switch(net, bus7, line3, et="l", type="LBS", closed=False)
sw7 = pp.create_switch(net, bus7, line4, et="l", type="LBS", closed=True)
sw8 = pp.create_switch(net, bus5, line4, et="l", type="LBS", closed=True)

pp.create_load(net, bus7, p_mw=20, q_mvar=4, scaling=0.6, name="load")
pp.create_load(net, bus6, p_mw=20, q_mvar=4, scaling=0.6, name="load")
pp.create_load(net, bus5, p_mw=20, q_mvar=4, scaling=0.6, name="load")
pp.create_load(net, bus4, p_mw=2, q_mvar=4, scaling=0.6, name="load")
pp.create_load(net, bus3, p_mw=2, q_mvar=4, scaling=0.6, name="load")

pp.create_gen(net, bus5,  p_mw=0.9, max_q_mvar=3, min_q_mvar=-3, vm_pu=1.09, name="generator")
pp.create_gen(net, bus6, p_mw=0.9, max_q_mvar=3, min_q_mvar=-3, vm_pu=1.05, name="generator")
pp.create_gen(net, bus7, p_mw=0.7, max_q_mvar=3, min_q_mvar=-3, vm_pu=1.08, name="generator")
pp.create_gen(net, bus8, p_mw=0.9, max_q_mvar=3, min_q_mvar=-3, vm_pu=1.09, name="generator")

storage_idx = pp.create_storage(net, bus=bus8, p_mw=0.5, max_e_mwh=1.0, min_e_mwh=0.2, soc_percent=50, name="storage",
                  max_p_mw=1.0, min_p_mw=-1.0, initial_e_mwh=0.5)

pp.create_shunt(net, bus3, q_mvar=-0.96, p_mw=0, name='Shunt')

try:
    pp.runpp(net, algorithm="nr") #run the power flow calculation using the Newton-Raphson method
    print("Power flow calculation was successful.")
    print("Results:")
    print(net.res_bus.vm_pu)
except Exception as e:
    print("Power flow calculation failed.")
    print("Error message:", str(e))