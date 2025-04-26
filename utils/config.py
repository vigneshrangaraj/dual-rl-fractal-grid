# utils/config.py

class Config:
    """
    Configuration for the Dual-RL Project.
    This class contains parameters for:
      - Global settings (e.g., seed, device)
      - Tertiary environment (macro-level control)
      - Secondary environment (local voltage control)
      - DER modules and communication
      - Agent hyperparameters for IA3C and SAC
    """

    # Global parameters
    seed = 42
    device = "cuda"  # or "cpu", depending on your setup

    # -------------------------------
    # Tertiary Environment Settings
    # -------------------------------
    num_microgrids = 2
    num_episodes = 1000
    # Each tie-line is represented as (mg_id1, mg_id2, status); status 1 = closed, 0 = open
    tie_lines = [(0, 1, 1)]
    V_ref = 1.0
    max_steps = 100
    # Reward weighting factors
    lambda_econ = 1.0  # Weight for economic cost penalty
    alpha_sec = 0.5  # Weight for secondary performance feedback
    beta_volt = 1.0  # Weight for voltage deviation penalty

    # Load parameters (tertiary level, e.g., aggregated load)
    base_load = 50.0  # kW
    load_variability = 0.1  # ±10% variation
    load_cost_factor = 0.05  # Cost per kW of load

    # Solar parameters
    num_solar = 2
    solar_buses = [5, 10]  # Bus locations where solar DERs are connected
    solar_base_output = 1  # MW
    solar_variability = 0.1  # ±10% variability

    # Wind parameters
    num_wind = 1
    wind_buses = [15]  # Bus location for wind DER
    wind_base_output = 1  # MW
    wind_variability = 0.2  # ±20% variability

    der_max_capacity = 0.1  # Maximum capacity of DERs (e.g., solar/wind) in MW

    # BESS parameters
    bess_capacity = 10000.0  # kWh
    bess_max_charge = 20.0  # kW
    bess_max_discharge = 20.0  # kW
    bess_charge_efficiency = 0.95
    bess_discharge_efficiency = 0.95
    bess_initial_soc = 0.5  # Initial state-of-charge (fraction)

    # Cost parameters
    bess_cost_per_mwh = 100.0
    pv_cost_per_mwh = 25.0
    wind_cost_per_mwh = 30.0

    # -------------------------------
    # Secondary Environment Settings
    # -------------------------------
    num_secondary_agents = 3 # Number of DERs per microgrid
    voltage_gain = 0.05  # How reactive power changes affect voltage (pu per kVAR)
    secondary_noise_std = 0.002  # Std. deviation for voltage update noise
    secondary_max_steps = 50
    action_penalty = 0.001  # Penalty coefficient for large control actions

    # Communication settings for DER agents (secondary level)
    comm_sigma = 1.0  # Sigma for Gaussian kernel in communication module
    comm_threshold = 5.0  # Distance threshold for communication (units consistent with positions)
    # Positions for DER agents (if None, secondary_env will generate random positions)
    positions = None

    # -------------------------------
    # IA3C Agent (Secondary) Settings
    # -------------------------------
    # State typically includes local voltage and current reactive power output.
    state_dim = 4
    action_dim = 1
    hidden_dim = 128
    gamma = 0.99
    lr = 1e-3
    entropy_coef = 0.01
    batch_size = 64
    alpha = 0.2
    tau = 0.005
    value_loss_coef = 0.5

    # -------------------------------
    # SAC Agent (Tertiary) Settings
    # -------------------------------
    sac_gamma = 0.99
    sac_tau = 0.005
    sac_alpha = 0.2
    sac_lr = 3e-4
    sac_batch_size = 64

# To use the configuration in your modules, you can do:
# from utils.config import DualRLConfig
# config = DualRLConfig()