# from scipy.io import loadmat
import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta
from copy import copy
import matplotlib.pyplot as plt
import csv
import seaborn as sns
import json
from pathlib import Path
import pvlib

from legacy.utils.helper_functions_vf import co2dens2ppm, weather2ppmrh, vaporDens2rh, denormalise_array, load_disturbances, rh2vaporDens, co2ppm2dens
from legacy.models.ModelFunctions_final_changed import DefineParameters, F, PVmoduleParameters, PVsysParameters, BattParameters, WindParameters, OtherParameters, EvalParameters
from legacy.models.Pv_power_model_class import PvPowerCalculator
from legacy.models.wind_energy_model_class import WindEnergyModel
from legacy.utils.other_helper_functions import extract_scalar, DefineConfigParameters, update_parameters, f, g, unscale_inputs, power_load, pro_act_D_i, pro_act_N_i, calculate_pv_output, plot_simulation_results
from legacy.economics.economic_class_nl import EnergyLettuceEconomics

# %%

class IntegratedSimulation:
    def __init__(self, DLI, design_para, new_eval_p, pv_power_traj, wind_power_traj, electricity_prices,period_codes):
        self.DLI = DLI
        self.design_para = design_para
        self.new_eval_p = new_eval_p
        self.pv_power_traj = pv_power_traj * design_para["num_pv"]
        self.wind_power_traj = wind_power_traj * design_para["num_wind"]
        self.electricity_prices = electricity_prices
        self.epw_path = None
        self.output_dir = Path("outputs")
        self.save_outputs = True
        self.external_d_values = None
        self.verbose = True

        # Initialize configuration and parameters
        self.config = DefineConfigParameters(design_para)
        self.p = DefineParameters()
        
        # print("FLOOR area:", self.p["A_floor"], "m2")
        # print ("CULTIVATION area:", self.p["A_cul"] ,"m2")
        # print("eta_par:", self.p["eta_par"])
        # print("CO2cap" "A_height", self.p["CO2cap"], self.p["A_height"])
        # print("resp_c", self.p["resp_c"])
        # print("eta_led", self.p["eta_led"])
        
        self.p, self.module_parameters, self.system_parameters, self.batt_param, self.other_p, self.eval_param = update_parameters(
            self.p, self.design_para, self.config, self.new_eval_p
        )
        self.period_codes = period_codes
    
        # self.econ = EnergyLettuceEconomics(self.eval_param, self.p, self.other_p, self.DLI, electricity_prices)
        # self.P_light_fixed = self.econ.calculate_ppfd(design_para["num_light"])
        # self.P_light_fixed = P_light_fixed

        # Time and simulation setup
        self.h = 10 * 60
        self.n_days = 365
        self.c = 86400
        self.t = np.arange(0, self.n_days * self.c + self.h, self.h)
        self.N = len(self.t) - 1
        self.energy_conversion_factor = self.h / 3600 * 1e-3  # the unit is kWh

        # Calculates P_light_fixed from num_light using the inverse of:
        # num_light = ceil(14812 + 741 * ((P_light_fixed - 200) / 10))
        # Remove the ceiling effect by adjusting down by 1 before solving
        adjusted_light = self.design_para["num_light"]
        self.P_light_fixed = (10 * (adjusted_light-14812)) / 741 + 200   # was adjusted_light - 14812
        
        # update the operational constraint of grid electricity export according to light numbers inputs
        def estimate_max_load(light_number):
            return 1.18110215e+02 * light_number + -1.70585814e+04
        
        # self.batt_param["P_grid_max_export"] = 0.8*estimate_max_load(adjusted_light)   # commented this using the default value in the model function
        
        
        # print("check P_grid_max_export:", self.batt_param["P_grid_max_export"])


    def _initialize_states(self):
        # Set initial state vectors
        x0 = np.array([0.496e-3, 1.45e-3, 22, rh2vaporDens(22, 70)])  #0.0035 0.496e-3
        u0 = np.array([0, 0, 1.5e-5, 0])
        y0 = g(x0)
        return x0, u0, y0
    
    def _setup_environmental_inputs(self):
        if self.external_d_values is not None:
            return self.external_d_values

        # === Read EPW weather file ===
        # epw_path = r"C:\ProjectCode\Design_Study\Data\ESP_AN_Sevilla.AP.083910_TMYx.epw"
        if self.epw_path is None:
            raise ValueError("epw_path must be set before running the simulation.")
        data,_ = pvlib.iotools.read_epw(self.epw_path)
        # print(self.epw_path)
        selected_data = data[['temp_air', 'ghi', 'wind_speed']].copy()
        # === Interpolate to 10-minute resolution (no extrapolation) ===
        # EPW data is hourly, indexed by datetime → resample to 10-minute frequency
        selected_data_10min = (
            selected_data
            .resample("10min")
            .interpolate(method="linear", limit_direction="both")
        )
        tem_data = selected_data_10min['temp_air'].values
        solar_data = selected_data_10min['ghi'].values  # use GHI as solar proxy
        # The interpolation covers exactly from the first to last EPW timestamp.
        # So we keep this as-is.
        # nan_column = np.full((len(solar_data), 1), np.nan)
        return np.column_stack((solar_data, tem_data))

    # === Return array: [solar, nan, temperature] ===

    def _compute_light_period(self):
        # photo_time = self.DLI / self.P_light_fixed * 1e6 / 3600
        # light_start_time = 12 - photo_time / 2
        # light_end_time = 12 + photo_time / 2
        light_start_time = 5 # 5
        light_end_time = 21 #21
        # 19-24 or 0-11
        
        return light_start_time * 3600, light_end_time * 3600

    # def _generate_light_schedule(self, light_start, light_end):
    #     def control_light(current_time):
    #         current_hour = current_time % self.c
    #         return 1 if light_start <= current_hour < light_end else 0
    #     return np.array([control_light(t) for t in self.t])
    
    def _generate_light_schedule(self, light_start, light_end):

        def control_light(current_time):
            # seconds within day
            current_sec = current_time % self.c

            # Case 1: same-day interval (e.g., 05:00 → 21:00)
            if light_start < light_end:
                return 1 if light_start <= current_sec < light_end else 0
            
            # Case 2: overnight interval (e.g., 19:00 → next day 11:00)
            else:
                return 1 if (current_sec >= light_start or current_sec < light_end) else 0

        return np.array([control_light(t) for t in self.t])

    def run(self):
        x0, u0, y0 = self._initialize_states()
        d_values = self._setup_environmental_inputs()
        light_start, light_end = self._compute_light_period()
        light_schedule = self._generate_light_schedule(light_start, light_end)

        N = self.N
        x_values = np.zeros((N, 4))
        y_values = np.zeros((N, 4))
        u_values = np.zeros((N, 4))
        nor_u = np.zeros((N, 4))
        p_load = np.zeros(N)

        x_values[0] = x0.copy()
        y_values[0] = y0.copy()
        u_values[0] = u0.copy()

        SoC_history = np.zeros(N)
        SoC_history[0] = self.batt_param["SoC_initial"]
        P_battery = P_grid_ex = P_grid_in = P_excess = P_deficit = 0

        P_battery_history = np.zeros(N)
        P_grid_ex_history = np.zeros(N)
        P_grid_in_history = np.zeros(N)
        P_excess_history = np.zeros(N)
        P_deficit_history = np.zeros(N)
        E_excess_history = np.zeros(N)
        E_deficit_history = np.zeros(N)
        E_grid_ex_history = np.zeros(N)
        E_grid_in_history = np.zeros(N)
        E_load_history = np.zeros(N)
        P_direct_renewable_to_load = np.zeros(N)
        P_renewable_to_battery = np.zeros(N)
        P_battery_to_load = np.zeros(N)
        self.electricity_prices_updated = self.electricity_prices.copy()  # copy the original electricity prices to this one as initial value and updated later during simulation

        nu = 4 
        # default values (mine)
        temp_range =  ((18, 20), (22, 25)) #((19,21),(23,25))
        humidity_range =(70, 80) #(68, 72)  
        carbon_range = (800, 850) # (1000, 1050) # 
        
        # temp_range = ((09.5,20.5),(22,23)) #((18, 20), (22, 25))
        # humidity_range = (74, 75) # (70, 80)
        # carbon_range = (780, 800) # (800, 850)

        u_prev_c, u_prev_t, u_prev_h = [0], [0], [0]
        growing_days = 0
        m = 0
        harvest_count = 0
        cycle_id = 0
        m_last_harvest = 0
        harvest_dw_arr = []
        target_dw =0.229 # 0.229

        
        # light_norm = (self.P_light_fixed - 200) / 90
        # u_cool_max = 500 + 2.1 * light_norm  # W/m²
        # u_cool_max *=0.92
        # # u_cool_max = 460
        
        # u_dehum_max = 9.95e-6 * self.P_light_fixed**0.27 - 2.662e-5  # kg/m²/s
        # u_dehum_max *=0.93
        # # u_dehum_max = 1.95e-5
        # # u_dehum_max = 1.38e-5 +4.63e-8 * P_light_fixed  # kg/m²/s
        # min_action = np.array([0, -u_cool_max*0.98, 0, 0])
        # max_action = np.array([
        #     5e-6,         # CO₂ injection (fixed or use your own)
        #     u_cool_max,
        #     u_dehum_max,
        #     self.P_light_fixed
        # ])        
        # print("min_action, max_action:", min_action, max_action)
        # --- New PPFD-dependent control limits ---
        # --- PPFD-dependent control limits (updated, quadratic fit) ---

        # --- PPFD-dependent control limits (adjusted quadratic fit) ---

        P = self.P_light_fixed  # PPFD (µmol m⁻² s⁻¹)

        # Cooling upper limit [W/m²]
        u_cool_max = -0.0029 * P**2 + 1.63 * P + 247.0


        # Dehumidification (unchanged)
        u_dehum_max = 1.627e-8 * P**1.293 - 1.770e-6  # kg/m²/s

        # Control bounds
        min_action = np.array([
            0,
            -0.98 * u_cool_max,
            0,
            0
        ])

        max_action = np.array([
            5e-6,          # CO₂ injection
            u_cool_max,
            u_dehum_max,
            P
        ])

        if self.verbose:
            print("min_action, max_action:", min_action, max_action)


        
        # min_action = np.array([0, -625, 0, 0])
        # max_action = np.array([5e-6, 625, 1.8e-5, self.P_light_fixed])


        # N = 3000
        try:
            while m < N - 1:
                current_weight = y_values[m][0]

                if m == 0 or current_weight >= target_dw:
                    x_values[m] = x0.copy()
                    u_prev_c[-1], u_prev_t[-1], u_prev_h[-1] = 0, 0, 0
                    if m > 0:
                        harvest_count += 1
                        cycle_id += 1
                        harvest_dw_arr.append(current_weight)
                        # print(f"harvest dw array: ", harvest_dw_arr)
                        # print(f"Harvest {harvest_count} at {current_weight:.4f} kg dw after {m} timesteps")
                        # print(f"Starting new cycle {cycle_id} at timestep {m}")

                y_values[m] = g(x_values[m])

                if light_schedule[m] == 1:
                    proactive_control = pro_act_D_i(y_values[m], nu, carbon_range, temp_range, humidity_range, u_prev_c, u_prev_t, u_prev_h)
                else:
                    proactive_control = pro_act_N_i(y_values[m], nu, carbon_range, temp_range, humidity_range, u_prev_c, u_prev_t, u_prev_h)

                u_prev_c.append(proactive_control[0])
                u_prev_h.append(proactive_control[1])
                u_prev_t.append(proactive_control[2])

                u_values[m] = unscale_inputs(proactive_control, min_action, max_action)
                nor_u[m] = proactive_control
                p_load[m] = power_load(u_values[m], self.p)
                
                P_pv = self.pv_power_traj[m]
                P_wind = self.wind_power_traj[m]
                P_renewable = P_pv + P_wind
                P_load = p_load[m]
                SoC = SoC_history[m]
                current_net_power = P_renewable - P_load

                
                no_storage = self.batt_param.get("battery_capacity", 0) <= 0 or \
                            self.batt_param.get("P_battery_max", 0) <= 0

                P_direct_renewable_to_load[m] = min(P_renewable, P_load)

                # without storage
                if no_storage:
                    # No battery → SoC fixed at 0, no (dis)charge flows
                    # force initial SoC to zero
                    SoC_history[:] = 0
                    SoC = 0
                    P_battery = 0
                    P_renewable_to_battery[m] = 0
                    P_battery_to_load[m] = 0

                    net = P_renewable - P_load

                    if net >= 0:
                        # Surplus → export up to limit, then curtail
                        P_grid_in  = 0
                        P_grid_ex  = min(net, self.batt_param.get("P_grid_max_export", float("inf")))
                        P_excess   = max(0, net - P_grid_ex)        # curtailed
                        P_deficit  = 0
                        # If you price exports differently, update price here if desired
                        self.electricity_prices_updated[m] = self.electricity_prices_updated[m] * 1.0 + 0.00  # or 0.5, etc.
                    else:
                        # Deficit → import up to limit, then leave unmet
                        need       = -net
                        P_grid_in  = min(need, self.batt_param.get("P_grid_max_import", float("inf")))
                        self.electricity_prices_updated[m] = (self.electricity_prices_updated[m] +0.03735)*1.21  # normal price for import
                        P_deficit  = max(0, need - P_grid_in)
                        P_grid_ex  = 0
                        P_excess   = 0                

                else:  # hybrid system, grid and renewable energy and storage (or any of them in the system)
                    # Direct renewable supply to load
                    P_direct_renewable_to_load[m] = min(P_renewable, P_load)
                    # charge battery with surplus renewable energy
                    if current_net_power >= 0:  
                        P_battery = min(
                            current_net_power,
                            (self.batt_param["SoC_max"] - SoC) * self.batt_param["battery_capacity"] / (self.batt_param["eta_c"] * self.h / 3600),
                            self.batt_param["P_battery_max"]
                        )   
                        P_renewable_to_battery[m] = P_battery
                        SoC += P_battery * self.batt_param["eta_c"] * self.h / 3600 / (self.batt_param["battery_capacity"]) # calculate SOC and update it for the next state

                        P_grid_ex = min(0, -current_net_power + P_battery/ 0.95)   # sell electricity to the grid   # add inverter efficiency with /0.95 when charging 

                        self.electricity_prices_updated[m] = self.electricity_prices_updated[m] * 1.0 + 0.00
                        if P_grid_ex < 0:
                            # print(f"Export at step {m}, P_grid_ex = {P_grid_ex}, price = {self.electricity_prices_updated[m]}")
                            P_grid_in = 0
                        
                        if P_grid_ex < -self.batt_param["P_grid_max_export"]:
                            P_excess = P_grid_ex + self.batt_param["P_grid_max_export"]  # (-)???
                            P_grid_ex = -self.batt_param["P_grid_max_export"]
                            
                            # self.electricity_prices_updated[m] = self.electricity_prices_updated[m] * 1.1 + 0.02 # can be 0.5
                            # print("this happens")
                        else:
                            P_excess = 0

                        P_grid_in = 0
                        P_deficit = 0

                    else:                          
                        P_battery = -min(
                            -current_net_power,
                            (SoC - self.batt_param["SoC_min"]) * self.batt_param["battery_capacity"] * self.batt_param["eta_d"] / (self.h / 3600),
                            self.batt_param["P_battery_max"]
                        )
                        P_battery_to_load[m] = -P_battery  # Make positive
                        SoC += P_battery * self.h / 3600 / (self.batt_param["battery_capacity"] * self.batt_param["eta_d"])

                        P_grid_in = max(0, -current_net_power + P_battery *0.95)   # buy electricity from the grid   # add inverter efficiency by *0.95 when discharging
                        self.electricity_prices_updated[m] = (self.electricity_prices_updated[m] +0.03735)*1.21  # normal price for import


                            
                        #below 7 lines are the same as before
                        if P_grid_in > self.batt_param["P_grid_max_import"]:
                            P_deficit = P_grid_in - self.batt_param["P_grid_max_import"]
                            P_grid_in = self.batt_param["P_grid_max_import"]
                        else:
                            P_deficit = 0

                        P_grid_ex = 0
                        P_excess = 0                        
                            
                            
                SoC_history[m + 1] = SoC
                electricity_price_history = self.electricity_prices_updated
                P_battery_history[m] = -P_battery  # change the signal here, considering the flow in as +, flow out as -, discharge should be flow in to the system, from system perspectives.
                P_grid_ex_history[m] = P_grid_ex
                P_grid_in_history[m] = P_grid_in
                P_excess_history[m] = P_excess
                P_deficit_history[m] = P_deficit
                E_excess_history[m] = P_excess * self.energy_conversion_factor
                E_deficit_history[m] = P_deficit * self.energy_conversion_factor
                E_grid_ex_history[m] = P_grid_ex * self.energy_conversion_factor
                E_grid_in_history[m] = P_grid_in * self.energy_conversion_factor
                E_load_history[m] = p_load[m] * self.energy_conversion_factor # signal of laod also changes since it is flow out,no not change this, as economic terms......
                # print("P_excess_history", np.sum(P_excess_history))
                # Update next state
                x_values[m + 1] = f(x_values[m], u_values[m], d_values[m], self.h, self.p)
                y_values[m + 1] = g(x_values[m + 1])

                # Track maturity
                if y_values[m][0] < target_dw and y_values[m + 1][0] >= target_dw:
                    growing_days = (m - (m_last_harvest if cycle_id > 0 else 0)) / (6 * 24)
                    m_last_harvest = m + 1
                    # print(f"Crop reached maturity at timestep {m+1}. Days to grow: {growing_days:.2f}")
                # print("growing days", growing_days)
                m += 1

        except Exception as e:
            if self.verbose:
                print(f"Error during simulation at timestep {m}: {e}")
            import traceback
            if self.verbose:
                print(traceback.format_exc())

        harvest_dw_arr.append(y_values[m][0])

        # Store histories
        self.electricity_price_history = electricity_price_history.reshape(-1, 1)
        self.growing_days = growing_days
        self.p_load = p_load.reshape(-1, 1)
        self.P_battery_history = P_battery_history.reshape(-1, 1)
        self.P_grid_ex_history = P_grid_ex_history.reshape(-1, 1)
        self.P_grid_in_history = P_grid_in_history.reshape(-1, 1)
        self.P_excess_history = P_excess_history.reshape(-1, 1)
        self.P_deficit_history = P_deficit_history.reshape(-1, 1)
        self.y_values = y_values
        self.u_values = u_values
        self.nor_u = nor_u
        self.d_values = d_values 
        self.x_values = x_values
        self.pv_power_traj = self.pv_power_traj.reshape(-1, 1)  
        self.wind_power_traj = self.wind_power_traj.reshape(-1, 1) 
        self.E_load_history = E_load_history.reshape(-1, 1)
        self.E_grid_in_history = E_grid_in_history.reshape(-1, 1)
        self.E_grid_ex_history = E_grid_ex_history.reshape(-1, 1)
        self.E_excess_history = E_excess_history.reshape(-1, 1)
        self.E_deficit_history = E_deficit_history.reshape(-1, 1)
        self.SOC_history = SoC_history.reshape(-1, 1)


        self.E_direct_renewable_to_load = P_direct_renewable_to_load.reshape(-1, 1)*self.energy_conversion_factor
        self.E_renewable_to_battery = P_renewable_to_battery.reshape(-1, 1) * self.energy_conversion_factor
        self.E_batt_to_load = P_battery_to_load.reshape(-1, 1) *self.energy_conversion_factor
        self.harvest_dw_arr = harvest_dw_arr
        
        self.E_re_generation = (self.pv_power_traj + self.wind_power_traj).reshape(-1, 1) * self.energy_conversion_factor
        self.E_pv_generation = self.pv_power_traj.reshape(-1,1) * self.energy_conversion_factor
        self.E_wind_generation = self.wind_power_traj.reshape(-1,1) * self.energy_conversion_factor

        if self.save_outputs:
            prices_df = pd.DataFrame({
                "Original Prices": self.electricity_prices,
                "Updated Prices": self.electricity_price_history.flatten()
            })
            self.output_dir.mkdir(parents=True, exist_ok=True)
            prices_df.to_excel(self.output_dir / "electricity_prices_comparison.xlsx", index=False)
        

        return self._final_metrics(self.harvest_dw_arr)


    # define a function that calculate the average value of states
    def _average_states(self, x_values, y_values, u_values):
        x_avg = np.mean(x_values, axis=0)
        y_avg = np.mean(y_values, axis=0)
        u_avg = np.mean(u_values, axis=0)
        return x_avg, y_avg, u_avg

    def _final_metrics(self, harvest_dw_arr):
        # self.econ = EnergyLettuceEconomics(self.eval_param, self.p, self.other_p, self.DLI, E_grid_in, E_grid_ex, self.electricity_prices_updated)
        self.econ = EnergyLettuceEconomics(electricity_prices=self.electricity_price_history,
                                eval_param=self.eval_param,
                                DLI=self.DLI,
                                E_grid_ex_history=self.E_grid_ex_history,
                                p=self.p,
                                E_grid_in_history=self.E_grid_in_history,
                                E_load_history=self.E_load_history,
                                P_grid_in_history=self.P_grid_in_history,   # <--- add this
                                other_p=self.other_p,
                                E_re_generation=self.E_re_generation,
                                E_pv_generation=self.E_pv_generation,
                                E_wind_generation=self.E_wind_generation,
                                E_batt_to_load= self.E_batt_to_load,
                                E_excess_history = self.E_excess_history,
                                E_deficit_history= self.E_deficit_history,
                                period_codes=self.period_codes   # <-- PASS THROUGH
                                )
        co2_emission_grid = extract_scalar(self.econ.objective_emission())   #checked
        C_lifetime, period_cost = self.econ.investment_cost_with_light(self.design_para, self.growing_days)
        # total_revenue_period, total_cost_period, electricity_cost = self.econ.other_op_revenue_cost(???)
        re_fraction = extract_scalar(self.econ.cal_refrac())   #checked, now it the net import from the grid is the nonrenewable part
        self_consum_rate = self.econ.self_consumption_rate()   #checked
        self_suf_ratio = self.econ.self_suf_ratio()   #checked
        curtail_ratio = self.econ.curtail_ratio()
        
        l_revenue = self.econ.lettuce_revenue(harvest_dw_arr)                          # checked
        annual_net_cash_flow, net_grid_e_cost, NPV = self.econ.calculate_annual_net_cash_flow_with_fixed_cost_grid(
            self.design_para, harvest_dw_arr
        )

        area_pv = extract_scalar(self.econ.calculate_area_pv(self.design_para))  
        area_wind = self.econ.calculate_area_wind(self.design_para)
        total_area = area_pv + area_wind
        Total_e_load = np.sum(self.E_load_history)  #unit kWh
        e_load_per_unit_freshweight = Total_e_load / (np.sum(harvest_dw_arr) * self.p["dw_fw"] * self.p["A_cul"])  # kWh/kg
        total_e_pv = np.sum(self.pv_power_traj) * self.energy_conversion_factor
        total_e_wind = np.sum(self.wind_power_traj) * self.energy_conversion_factor
        e_pv_unit_area = total_e_pv / area_pv
        e_wt_unit_area = total_e_wind / area_wind if area_wind != 0 else 0
        net_cost = self.econ.grid_operational_cost()

        total_co2_absorbed = np.sum(self.u_values[:,0]) * 10 * 60 * self.p["A_width"] * self.p["A_length"]
        net_co2_generation = co2_emission_grid - total_co2_absorbed  # unit wrong, need to update, ton - kg now
        
        lcoe = self.econ.calculate_lcoe_re(self.design_para)
        lcoe_pv, lcoe_wind, lcoe_batt, lcoe_curtailed = self.econ.lcoe_comp(self.design_para)
        lcoe_grid = (self.econ.grid_operational_cost() / np.sum(self.E_grid_in_history)
             if np.sum(self.E_grid_in_history) != 0 else 0)  # (net buy)/total buy


        results = {
            # "DLI": round(float(self.DLI), 4), 
            # "light number": int(self.design_para["num_light"]),                  
            # "PPFD": round(float(self.P_light_fixed), 4),
            "Growing_days per period": (float(self.growing_days)),            # use the growing days of the last full cycle
            "Energy use per unit Freshweight (kWh/kg)": round(float(e_load_per_unit_freshweight), 4),       # checked
            "Total Energy Demand (kWh)": round(float(Total_e_load), 4),    # checked
            "CR Curtailment ratio": round(float(curtail_ratio), 4),
            "REF Renewable Fraction": round(float(re_fraction), 4),            # checked and updated to net import from grid
            "SCR Self-consumption ratio": round(float(self_consum_rate), 4),      # checked
            "SSR Self-sufficiency ratio": round(float(self_suf_ratio), 4),      # checked, 
            "LCOE": round(float(lcoe), 4),                                # checked, this is the 
            "Grid CO2 Emission (ton/yr)": round(float(co2_emission_grid), 4),   # checked 
            # "CO2 Absorbed (kg/yr)": round(float(total_co2_absorbed), 4),           # estimated value, need to be updated
            # "Net CO2 Generation (kg/yr)": round(float(net_co2_generation), 4),     # checked,this is sum of the cost of each timestep
            "Grid Net Cost (€/yr)": round(float(net_cost), 4),                #checked
            # "Project Lifetime cost (€)": round(float(C_lifetime), 4),          # this is a simple average of the system lifetime cost (with light)
            # "Growing Period Cost": round(float(period_cost), 4),           # this is a simple average of the system lifetime cost
            "Lettuce Revenue (€/yr)": round(float(l_revenue), 4),             # checked
            "Electricity Cost with connectivity cost (€)": round(float(net_grid_e_cost), 4), # checked, this is sum of the cost of each timestep
            "Total annual net cash flow (€/yr)": round(float(annual_net_cash_flow), 4), # checked
            # "Payback Time (yr)" : round(float(payback_time), 4),   # checked 
            # "NPV (€)": round(float(NPV), 4),                                # checked
            # "Total PV Area (m²)": round(float(area_pv), 4),
            # "PV Area per Panel (m²)": round(float(area_pv / self.design_para['num_pv']), 4),
            "Total PV Energy (kWh)": round(float(total_e_pv), 4), # checked
            "Total Wind Energy (kWh)": round(float(total_e_wind), 4), # checked
            "Total battery to load Energy (kWh)": round(float(np.sum(self.E_batt_to_load)), 4), # checked
            "lcoe PV (€/kWh)": round(float(lcoe_pv), 4), # checked
            "lcoe Wind (€/kWh)": round(float(lcoe_wind), 4), # checked
            "lcoe Battery (€/kWh)": round(float(lcoe_batt), 4), # checked
            "lcoe Curtailed (€/kWh)": round(float(lcoe_curtailed), 4), # checked
            "lcoe Grid (€/kWh)": round(float(lcoe_grid), 4), # checked
            # "Number of PV Panels": self.design_para["num_pv"],
            # "Number of Batteries": self.design_para["num_batt"],
            # "Number of Wind Turbines": self.design_para["num_wind"],
            # "Number of Lights": self.design_para["num_light"],
            "area_pv": round(float(area_pv),4),
            "area_wt": round(float(area_wind),4),
            # "e_pv_unit_area": round(float(e_pv_unit_area),4),
            # "e_wt_unit_area": round(float(e_wt_unit_area),4)
        }



        if self.verbose:
            print("\nSimulation Summary:")
            summary_df = pd.DataFrame.from_dict(results, orient='index', columns=['Value'])
            print(summary_df.to_string(float_format="%.4f"))

        # print("max value of E_load_history (converted to P)", np.max(self.E_load_history)/self.energy_conversion_factor)
        
        return {
            "result_summary": results,
            "obj_annual_net_cash_flow": annual_net_cash_flow,
            "co2_emission_grid": co2_emission_grid,
            "re_fraction": re_fraction,
            "self_consum_rate": self_consum_rate,
            "self_suf_ratio": self_suf_ratio,
            "lcoe": lcoe,
            "curtailment_ratio": curtail_ratio,
            "total_area": total_area
        }
        
    def export_results(self, summary_dict, filepath="simulation_results.csv"):
        """Exports simulation results to a CSV file."""
        df = pd.DataFrame(summary_dict).transpose().T  # Transpose to get one row
        df.to_csv(filepath, mode='a', header=False, index=False)
        # print(f"Results exported to: {filepath}")
        
    # store the updated electricity prices in npy file as an array
    def update_electricity_prices(self):
        """save the updated electricity prices, self.electricity_prices_updated in npy file as an array."""
        if not self.save_outputs:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        np.save(self.output_dir / "electricity_prices_updated.npy", self.electricity_prices_updated)

    # Code for plotting, previously used. now I change to save data, decouple plot from simulation run
    def plot_results(self):
        """Calls external plotting utility using simulation data."""
        plot_simulation_results(
            # 3000,
            self.N,
            self.h,
            self.pv_power_traj,             
            self.wind_power_traj,
            -self.p_load.reshape(-1, 1),
            self.P_battery_history,
            self.P_grid_ex_history,
            self.P_grid_in_history,
            self.P_excess_history,
            self.P_deficit_history,
            self.y_values,
            self.u_values,
            self.nor_u,
            self.d_values,
            self.x_values
        )       
    
    def save_plot_data(self, out_dir="plot_export", basename="simulation"):
        """
        Save all time-series needed for plotting to CSV, plus a small JSON meta file.

        Files written:
          <out_dir>/<basename>_timeseries.csv
          <out_dir>/<basename>_meta.json
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        m = int(self.N)          # number of samples
        h = float(self.h)        # step [s]

        # Construct time in days (same as you plot)
        time_days = np.arange(0, m * h, h) / 3600.0 / 24.0

        # Helper to safely get a 2D numpy view with at least one column
        def col(a, idx=0):
            a = np.asarray(a)
            if a.ndim == 1:
                return a[:m]
            return a[:m, idx]

        def mat(a, k=None):
            """Return a 2D matrix of shape (m, k_guess) clipped to m rows."""
            a = np.asarray(a)
            a = a[:m]
            if a.ndim == 1:
                a = a.reshape(-1, 1)
            if k is not None and a.shape[1] != k:
                # Trim or pad columns to expected width if user supplied k
                k_eff = min(k, a.shape[1])
                a = a[:, :k_eff]
            return a

        # Build a dict of columns -> arrays (length m)
        df_dict = {
            "time_days": time_days,
            # powers in W (raw); we'll scale to MW in the plotting notebook
            "pv_power_W":        col(self.pv_power_traj, 0),
            "wind_power_W":      col(self.wind_power_traj, 0),
            # you used -self.p_load.reshape(-1,1) in the plot; we store the plotted sign here:
            "load_power_W":      -np.asarray(self.p_load).reshape(-1)[:m],
            "battery_power_W":   col(self.P_battery_history, 0),
            "grid_export_W":     col(self.P_grid_ex_history, 0),
            "grid_import_W":     col(self.P_grid_in_history, 0),
            "curtailment_W":     col(self.P_excess_history, 0),
            "deficit_W":         col(self.P_deficit_history, 0),
        }

        # Measurements y_check (m x 4 expected)
        if hasattr(self, "y_values") and self.y_values is not None:
            Y = mat(self.y_values)
            y_cols = ["y_dry_mass",
                      "y_co2_ppm",
                      "y_temp_C",
                      "y_rel_humidity_pct"]
            for i in range(min(Y.shape[1], len(y_cols))):
                df_dict[y_cols[i]] = Y[:, i]

        # Controls u_values (m x 4 expected, *denormalized*)
        if hasattr(self, "u_values") and self.u_values is not None:
            U = mat(self.u_values)
            u_cols = ["u_co2_injection_kg_m2_s",
                      "u_Q_cool_W_m2",
                      "u_phi_cond_kg_m2_s",
                      "u_light_ppfd_umol_m2_s"]
            for i in range(min(U.shape[1], len(u_cols))):
                df_dict[u_cols[i]] = U[:, i]

        # Normalized controls (optional)
        if hasattr(self, "nor_u") and self.nor_u is not None:
            UN = mat(self.nor_u)
            for i in range(UN.shape[1]):
                df_dict[f"u_norm_{i+1}"] = UN[:, i]

        # Disturbances (optional)
        if hasattr(self, "d_values") and self.d_values is not None:
            D = mat(self.d_values)
            for i in range(D.shape[1]):
                df_dict[f"d_{i+1}"] = D[:, i]

        # States (optional)
        if hasattr(self, "x_values") and self.x_values is not None:
            X = mat(self.x_values)
            for i in range(X.shape[1]):
                df_dict[f"x_{i+1}"] = X[:, i]

        # Write CSV
        df = pd.DataFrame(df_dict)
        csv_path = out_dir / f"{basename}_timeseries.csv"
        df.to_csv(csv_path, index=False)

        # Write meta JSON
        meta = {
            "h_seconds": h,
            "m": m,
            "notes": "Powers are in Watts; scale by 1e-6 for MW in plots. time_days is precomputed."
        }
        meta_path = out_dir / f"{basename}_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        print(f"Saved: {csv_path}")
        print(f"Saved: {meta_path}") 
        
# %%
# DLI = 16.99 # mol.m-2.d-1

if __name__ == "__main__":  
    import os
    print("CWD =", os.getcwd())
    
    PERIOD_CODES_2023 = np.load(
    r"C:\ProjectCode\Design_Study\draft_code\time_period_codes_2023_spain.npy"
    ).astype(np.int8)
    
    design_para = {}
    design_para["num_batt"] =0 #200 #109# 500
    design_para["num_pv"] =0 #4000 #5500#5000
    design_para["num_wind"] = 0#1 3#8#8
    design_para["num_light"] = 14812 #+741*9#21482 # 6670
    
    # ppfd= 200
    # design_para["num_light"] = math.ceil(14812 + 741*((ppfd-200)/10)) # 14812
    
    DLI = 16.99
    
    new_eval_p = {}
    new_eval_p["pv_unit_cost"] = 810*0.95*0.34  #810$/kw=692euro,2021 # 350# 140*2.5  Change reference: 692 euro/kw, 1300 euro/kw, 350 euro/kwh for pv, wt and batt.
    new_eval_p["wind_unit_cost"] = 1590*0.95*300 #1590$/kw=1340,2021 # 464000 # 20000 *10 * 1.3
    new_eval_p["batt_unit_cost"] = 353*4.8 #353 euro/kwh # 2164 # 1300 *2.5
    new_eval_p["led_unit_cost"] = 30
    new_eval_p["lettuce_price"] = 1.08 
    new_eval_p["co2_emission_factor"] = 0.329 # 0.329 #0.329 #0.329 # 0.174 # 0.329 # netherlands 2021: 0.329 tonsco2/MWh
    # Spain 2021: 0.174 tonsco2/MWh. change for location every time
    
    # P_light_fixed = 290 # only used here to calculate number of lights. (range from 200 to 290) (umol.m-2.s-1)
    
    pv_power_traj = np.load('NL_TMY_year_pv_power_traj.npy')
    wind_power_traj = np.load('NL_TMY_year_wind_power_traj.npy')
    # electricity_prices = np.load('NL_year_electricity_prices.npy')   
    electricity_prices = np.load('NL_syn_electricity_prices.npy')
    weather_path = r"C:\ProjectCode\Design_Study\Data\NLD_GE_Hupsel.062830_TMYx.epw"
  
    # pv_power_traj = np.load('SP_TMY_year_pv_power_traj.npy')
    # wind_power_traj = np.load('SP_TMY_year_wind_power_traj.npy')
    # electricity_prices = np.load('SP_syn_electricity_prices.npy')   
    # weather_path = r"C:\ProjectCode\Design_Study\Data\ESP_AN_Sevilla.AP.083910_TMYx.epw"

    print("check max pv power, max wind power, max electricity prices",np.max(pv_power_traj), np.max(wind_power_traj), np.max(electricity_prices))
    # result = evaluate_model(DLI, P_light_fixed, design_para, 
    #                         pv_power_traj, wind_power_traj, electricity_prices) 

    sim = IntegratedSimulation(DLI, design_para, new_eval_p, pv_power_traj, wind_power_traj, electricity_prices, PERIOD_CODES_2023)
    # Specify the EPW file you want to use before running any environment setup
    sim.epw_path = weather_path # r"C:\ProjectCode\Design_Study\Data\ESP_AN_Sevilla.AP.083910_TMYx.epw"
    # sim.epw_path = r"C:\ProjectCode\Design_Study\Data\NLD_GE_Hupsel.062830_TMYx.epw"
    # Then call the method normally
    results = sim.run()
    sim.update_electricity_prices()
    # print(results)
    
    # result1, result2, result3 = sim.run()
    re_fraction = results["re_fraction"]
    annual_net_cash_flow = results["obj_annual_net_cash_flow"]
    test_one = results["result_summary"]["LCOE"]
    # print("RE Fraction:", re_fraction)
    # print("Annual Net Cash Flow:", annual_net_cash_flow)
    # Show plots
    sim.plot_results()
    
    sim.save_plot_data(out_dir="plot_export", basename="simulation_results")

    # pot the array of SOC_history
    plt.figure(figsize=(10, 5))
    plt.plot(sim.SOC_history, label='SoC History', color='blue')
    plt.title('Battery State of Charge (SoC) History')
    plt.xlabel('Time Step')
    plt.ylabel('State of Charge (SoC)')
    # plt.show()

# %%
