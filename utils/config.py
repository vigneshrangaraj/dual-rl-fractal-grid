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
    use_lstm = False
    use_centralized_sac = False  # Flag to enable centralized SAC for ablation study
    use_centralized_ddpg = False  # Flag to enable centralized DDPG for ablation study
    use_centralized_ppo = True  # Flag to enable centralized PPO for ablation study

    # -------------------------------
    # PPO Agent (Centralized) Settings
    # -------------------------------
    ppo_lr = 3e-4
    ppo_clip_ratio = 0.2
    ppo_value_loss_coef = 0.5
    ppo_entropy_coef = 0.01
    ppo_max_grad_norm = 0.5
    ppo_target_kl = 0.01
    ppo_update_epochs = 4
    ppo_batch_size = 64
    ppo_gamma = 0.99
    ppo_gae_lambda = 0.95
    ppo_buffer_size = 10000

    # -------------------------------
    # Tertiary Environment Settings
    # -------------------------------
    num_microgrids = 1
    num_episodes = 10000
    # Each tie-line is represented as (mg_id1, mg_id2, status); status 1 = closed, 0 = open
    tie_lines = [(0, 1, 1)]
    V_ref = 1.0
    max_steps = 23
    # Reward weighting factors
    lambda_econ = 1.0  # Weight for economic cost penalty
    alpha_sec = 0.5  # Weight for secondary performance feedback
    beta_volt = 1.0  # Weight for voltage deviation penalty
    beta_deficient = 1.0

    # Load parameters (tertiary level, e.g., aggregated load)
    base_load = 50.0  # kW
    load_disturbance_percent = 0.1  # ±10% variation
    load_cost_factor = 0.05  # Cost per kW of load

    # DER/BESS configuration
    num_der_total = 11  # Default, can be overridden
    num_bess_total = 5  # Default, can be overridden
    num_loads = 29

    num_der_solar = 4
    num_der_wind = 7
    
    # Solar parameters
    num_solar = 1
    solar_base_output = 250  # MW
    solar_variability = 0.1  # ±10% variability

    # Wind parameters
    num_wind = 1
    wind_base_output = 500  # MW
    wind_variability = 0.2  # ±20% variability

    beta_convergence = 0.1

    der_max_capacity = 0.1  # Maximum capacity of DERs (e.g., solar/wind) in MW

    # Temporal pricing parameters
    buy_ext_grid = 150  # Weight for external grid cost
    sell_ext_grid = 80  # Weight for external grid revenue
    
    # Time-varying price parameters (24-hour cycle)
    base_buy_price = 15.0  # Base buying price per MWh
    base_sell_price = 3.0  # Base selling price per MWh
    price_volatility = 0.3  # Price variation factor (30%)
    
    # Price forecast parameters
    price_forecast_horizon = 24  # Hours to forecast ahead
    price_forecast_noise = 0.1  # Noise in price forecast (10%)
    
    # Time-of-use pricing parameters
    peak_hours = [18, 19, 20, 21, 22]  # Evening peak hours
    off_peak_hours = [1, 2, 3, 4, 5, 6]  # Early morning off-peak hours
    peak_multiplier = 1.5  # Price multiplier during peak hours
    off_peak_multiplier = 0.7  # Price multiplier during off-peak hours

    # BESS parameters
    bess_capacity = 10000.0  # kWh
    bess_max_charge = 20.0  # MW
    bess_max_discharge = 20.0  # MW
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
    voltage_gain = 0.05  # How reactive power changes affect voltage (pu per kVAR)
    secondary_noise_std = 0.002  # Std. deviation for voltage update noise
    secondary_max_steps = 5
    action_penalty = 0.001  # Penalty coefficient for large control actions

    V_nom = 1  # Nominal voltage (pu)

    # Voltage control parameters
    V_min = 0.9  # Minimum allowed voltage
    V_max = 1.1  # Maximum allowed voltage
    max_voltage_change = 0.05  # Maximum voltage change per step
    voltage_proportional_gain = 0.1  # Proportional gain for voltage control
    voltage_integral_gain = 0.01  # Integral gain for voltage control

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
    action_dim = 10
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
    sac_alpha = 0.5  # Legacy parameter (kept for compatibility)
    sac_lr = 3e-4
    sac_batch_size = 256

    policy_lr = 3e-4
    q_lr = 3e-4
    memory_size = 100000
    replay_size = 100000  # Size of the replay buffer

    sac_action_scale = 0.3

    def get_der_bus_mapping(self):
        """
        Generate DER bus mapping based on configurable DER counts.
        Returns a dictionary with solar and wind bus mappings.
        """
        # Default bus mapping for der_4.py structure
        solar_buses = [4, 5]  # Default solar buses
        wind_buses = [6, 7]   # Default wind buses
        
        # Adjust based on configurable counts
        solar_buses = solar_buses[:self.num_der_solar]
        wind_buses = wind_buses[:self.num_der_wind]
        
        return {
            'solar_buses': solar_buses,
            'wind_buses': wind_buses,
            'total_der_count': self.num_der_total
        }

    def get_temporal_prices(self, time_step):
        """
        Generate time-varying electricity prices with arbitrage potential.
        """
        import numpy as np

        base_buy = self.base_buy_price
        base_sell = self.base_sell_price

        if time_step in self.peak_hours:
            # High demand: Buy price high, sell price high
            buy_price = base_buy * self.peak_multiplier
            sell_price = base_sell * self.peak_multiplier
        elif time_step in self.off_peak_hours:
            # Low demand: Buy cheap, but sell for a fair price
            buy_price = base_buy * self.off_peak_multiplier
            sell_price = base_sell * self.off_peak_multiplier * 0.2
        else:
            # Normal hours
            buy_price = base_buy
            sell_price = base_sell * 0.1  # small premium

        # Add volatility (same for reproducibility)
        np.random.seed(time_step)
        buy_noise = np.random.normal(0, self.price_volatility * buy_price * 0.1)
        sell_noise = np.random.normal(0, self.price_volatility * sell_price * 0.1)

        buy_price = max(10.0, buy_price + buy_noise)

        return {
            'buy_price': buy_price,
            'sell_price': sell_price
        }

    def get_price_forecast(self, current_time_step):
        """
        Generate price forecast for the next 24 hours using time-of-use pricing logic.

        Args:
            current_time_step: Current hour (0-23)

        Returns:
            dict: buy_price_forecast and sell_price_forecast arrays
        """
        import numpy as np

        buy_forecast = []
        sell_forecast = []

        for t in range(self.price_forecast_horizon):
            future_time = (current_time_step + t) % 24

            # Use temporal pricing logic
            base_buy = self.base_buy_price
            base_sell = self.base_sell_price

            if future_time in self.peak_hours:
                buy_price = base_buy * self.peak_multiplier
                sell_price = base_sell * self.peak_multiplier * 1.2
            elif future_time in self.off_peak_hours:
                buy_price = base_buy * self.off_peak_multiplier
                sell_price = base_sell * self.off_peak_multiplier * 0.9
            else:
                buy_price = base_buy
                sell_price = base_sell * 0.1

            # Add noise to forecast
            np.random.seed(current_time_step + t)
            buy_noise = np.random.normal(0, self.price_forecast_noise * buy_price)
            sell_noise = np.random.normal(0, self.price_forecast_noise * sell_price)

            final_buy = max(10.0, buy_price + buy_noise)
            final_sell = max(5.0, sell_price + sell_noise)

            buy_forecast.append(final_buy)
            sell_forecast.append(final_sell)

        return {
            'buy_price_forecast': buy_forecast,
            'sell_price_forecast': sell_forecast
        }

# To use the configuration in your modules, you can do:
# from utils.config import DualRLConfig
# config = DualRLConfig()