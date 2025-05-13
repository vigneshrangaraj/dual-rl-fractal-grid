# env/tertiary/panda_power_wrapper.py

import pandapower.diagnostic as diag

import pandapower as pp
import copy
import pandapower.topology as top


class PandaPowerWrapper:
    def __init__(self, config):
        self.config = config
        self.microgrids = []
        self.tie_lines = []
        self.filled_net = None

    def reset_network(self, microgrids, tie_lines):
        """
        Args:
            microgrids (list): List of microgrid objects.
            tie_lines (list): List of tie-line tuples in the form (mg_id1, mg_id2, status).
                              Status should be 1 (closed) or 0 (open).
        """
        self.microgrids = microgrids
        self.tie_lines = tie_lines

    @staticmethod
    def combine_microgrids_into_network(microgrids, tie_lines):
        combined_net = pp.create_empty_network()
        bus_mapping = {}

        for mg in microgrids:
            mg_copy = copy.deepcopy(mg.net)

            # Map and copy buses
            for idx in mg_copy.bus.index:
                bus_data = mg_copy.bus.loc[idx]
                new_bus = pp.create_bus(combined_net, vn_kv=bus_data.vn_kv, name=bus_data.name)
                bus_mapping[(mg.mg_id, idx)] = new_bus

            # Copy components: lines, loads, generators, storage
            for element in ["line", "load", "gen", "storage", "ext_grid"]:
                if hasattr(mg_copy, element):
                    df = getattr(mg_copy, element)
                    for _, row in df.iterrows():
                        row_dict = row.to_dict()

                        if element == "line":
                            from_bus = bus_mapping[(mg.mg_id, row['from_bus'])]
                            to_bus = bus_mapping[(mg.mg_id, row['to_bus'])]
                            params = {k: v for k, v in row_dict.items() if k not in ["from_bus", "to_bus"]}

                            pp.create_line(
                                combined_net, from_bus, to_bus,
                                length_km=row["length_km"],  # Keep the original length
                                std_type="NAYY 4x150 SE",
                                in_service=True
                            )
                        elif element == "ext_grid":
                            old_bus = row['bus']
                            new_bus = bus_mapping[(mg.mg_id, old_bus)]
                            params = {k: v for k, v in row_dict.items() if k != "bus"}
                            pp.create_ext_grid(net=combined_net, bus=new_bus, **params)
                        else:
                            old_bus = row['bus']
                            new_bus = bus_mapping[(mg.mg_id, old_bus)]
                            params = {k: v for k, v in row_dict.items() if k != "bus"}
                            create_fn = getattr(pp, f"create_{element}")
                            create_fn(net=combined_net, bus=new_bus, **params)

        # Add tie-lines between mapped buses of different microgrids
        for mg1_id, mg2_id, status in tie_lines:
            if status <= 0:
                # Tie is open - no physical connection!
                # Make sure both microgrids have their own ext_grid
                for mg_id in [mg1_id, mg2_id]:
                    mg = next(mg for mg in microgrids if mg.mg_id == mg_id)
                    slack_bus_orig = PandaPowerWrapper.find_ext_grid_bus(mg.net)
                    slack_bus_new = bus_mapping[(mg_id, slack_bus_orig)]
                    if not (combined_net.ext_grid.bus == slack_bus_new).any():
                        pp.create_ext_grid(combined_net, bus=slack_bus_new, vm_pu=1.0)
            else:
                # Tie is closed - physically connect them
                r_ohm_per_km = 0.01
                x_ohm_per_km = 0.01

                mg2 = next(mg for mg in microgrids if mg.mg_id == mg2_id)
                slack_bus_mg2 = PandaPowerWrapper.find_ext_grid_bus(mg2.net)
                slack_bus_mg2_new = bus_mapping[(mg2_id, slack_bus_mg2)]

                combined_net.ext_grid = combined_net.ext_grid[combined_net.ext_grid['bus'] != slack_bus_mg2_new]

                # Create the physical tie-line
                mg1_load_bus_orig = microgrids[mg1_id].net.load.bus.iloc[0]
                mg2_load_bus_orig = microgrids[mg2_id].net.load.bus.iloc[0]

                bus1 = bus_mapping[(mg1_id, mg1_load_bus_orig)]
                bus2 = bus_mapping[(mg2_id, mg2_load_bus_orig)]

                pp.create_line_from_parameters(
                    combined_net,
                    from_bus=bus1,
                    to_bus=bus2,
                    length_km=0.1,
                    r_ohm_per_km=r_ohm_per_km,
                    x_ohm_per_km=x_ohm_per_km,
                    c_nf_per_km=0,
                    max_i_ka=1.0,
                    name=f"tie_{mg1_id}_{mg2_id}"
                )

        print("switch", combined_net["switch"])


        return combined_net, bus_mapping

    @staticmethod
    def find_ext_grid_bus(mg_net):
        """Finds the bus index where the ext_grid is attached in a microgrid net."""
        ext_grid_df = mg_net.ext_grid
        if len(ext_grid_df) == 0:
            raise ValueError("Microgrid has no ext_grid defined.")
        return ext_grid_df.bus.values[0]

    @staticmethod
    def check_islanded_balance(filled_net):
        """
        In islanded mode, ext_grid should supply ≈ 0 power.
        If not, local DERs are under/over-provisioned.
        """
        if filled_net is not None and "res_ext_grid" in filled_net and not filled_net.res_ext_grid.empty:
            p_mw = filled_net.res_ext_grid.p_mw.sum()
            if abs(p_mw) > 1e-3:  # Tolerance of 1W
                print(f"[Warning] Ext grid is supplying {p_mw:.4f} MW — system is not balanced.")
            return p_mw
        return 0.0

    @staticmethod
    def run_power_flow(microgrids, tie_lines):
        if (len(microgrids) == 1):
            net = microgrids[0].net
        else:
            net = microgrids[0].net

        # print("=== EXT GRIDS ===")
        # print(net.ext_grid)
        #
        # print("=== BUSES ===")
        # print(net.bus)
        #
        # print("=== LINES ===")
        # print(net.line)
        #
        # print("=== line 34 ===")
        # print(net.line[net.line['name'] == 'tie_0_1'])
        #
        # print("=== UNSUPPLIED BUSES ===")
        # print(pp.topology.unsupplied_buses(net))
        #
        # print("=== LOADS ===")
        # print(net.load)
        #
        # print("=== genS ===")
        # print(net.gen)
        #
        # print("=== STORAGE ===")
        # print(net.storage)
        #
        # print("=== FULL NET DATA ===")
        # print(net)
        #
        # print("=== DIAGNOSTIC ===")
        # diagnostic_result = pp.diagnostic(net, report_style=None, silence_warnings=True)
        # for key, value in diagnostic_result.items():
        #     print(f"--- {key} ---")
        #     print(value)
        #
        # # find isolated buses
        # isolated_buses = top.unsupplied_buses(net)
        # print(isolated_buses)
        # #
        # # find unsupplied loads or generators
        # unsupplied_gens = top.unsupplied_buses(net)
        # print(f"Unsupplied buses: {unsupplied_gens}")

        try:
            pp.runpp(net, algorithm="nr", max_iteration=50, calculate_voltage_angles=True, tolerance_mva=1e-2, enforce_q_lims=True)
        except Exception as e:
            print(f"Unified power flow failed: {e}")
            inv_voltages = [2] * len(net.res_bus.vm_pu)
            print("==== print total mv at load and gen =====")
            print(net.load.p_mw)
            print(net.gen.p_mw)
            return net, inv_voltages

        print("=== POWER FLOW CONVERGED SUCCESSFULLY ===")
        print("=== Voltage Results ===")
        print(net.res_bus.vm_pu)
        inv_voltages = net.res_bus.vm_pu.values
        print("==== print total mv at load and gen =====")
        print(net.load.p_mw)
        print(net.gen.p_mw)

        for mg in microgrids:
            mg.filled_net = net
            avg_voltage = net.res_bus.vm_pu.mean()

        return net, inv_voltages


# For testing the wrapper standalone.
if __name__ == "__main__":
    import pandapower.networks as nw
    import pandapower as pp

    net = pp.create_empty_network()

    # display all standard lines
    print(pp.available_std_types(net))