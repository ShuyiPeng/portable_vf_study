import numpy as np
import math
import time
from functools import wraps

# Timing decorator to measure method execution time
def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        # Keep optimization output clean; enable ad-hoc timing with a local print if needed.
        return result
    return wrapper

class EnergyLettuceEconomics:
    def __init__(self, eval_param,P_grid_in_history, p,other_p, DLI, electricity_prices, E_grid_in_history, E_grid_ex_history, 
                 E_load_history, E_re_generation,
                 E_pv_generation, E_wind_generation,E_batt_to_load, 
                 E_excess_history, E_deficit_history, period_codes, light_life_hours=36000):
        # print("DEBUG econ period_codes input:", type(period_codes),
        #     None if period_codes is None else period_codes.shape)
        
        self.eval_param = eval_param
        self.light_life_hours = light_life_hours
        self.period_codes = np.asarray(period_codes, dtype=np.int16).ravel()
        self.p = p
        self.P_grid_in_history = P_grid_in_history  # <-- add this,if run the NL code, change it here or change in that py file.
        self.other_p = other_p
        self.DLI = DLI
        self.electricity_prices = electricity_prices # the array has already been converted to EUR/kWh
        self.E_grid_in_history = E_grid_in_history
        self.E_grid_ex_history = E_grid_ex_history
        self.E_load_history = E_load_history
        self.E_re_generation = E_re_generation
        self.E_pv_generation = E_pv_generation 
        self.E_wind_generation = E_wind_generation
        self.E_batt_to_load = E_batt_to_load
        self.E_curtail = E_excess_history
        self.E_deficit = E_deficit_history
        

    @timer
    def grid_power_toll_spain(self, vat=1.21):
        if self.P_grid_in_history is None:
            raise ValueError("P_grid_in_history is required for power toll (€/kW·year).")

        N = len(self.E_grid_in_history)
        codes = self.period_codes[:N]

        POWER_TOLL = np.array([23.946498, 12.687713, 4.747747, 3.339695, 0.070979, 0.062703], dtype=float)

        # Convert P_grid_in_history from W to kW
        P_in = np.clip(self.P_grid_in_history / 1000.0, 0.0, None)
        n = min(len(P_in), len(codes))
        P_in = P_in[:n]
        codes = codes[:n]

        # Vectorized calculation using bincount (much faster than loop)
        avg_kw = np.zeros(6, dtype=float)
        for i in range(6):
            mask = (codes == (i + 1))
            count = mask.sum()
            avg_kw[i] = P_in[mask].sum() / count if count > 0 else 0.0

        cost = float(np.sum(avg_kw * POWER_TOLL)) * vat
        return cost, avg_kw

    
    @timer
    def grid_energy_toll_spain(self, vat=1.21):
        N = len(self.E_grid_in_history)
        codes = self.period_codes[:N]

        ENERGY_TOLL = np.array([0.026785, 0.012281, 0.005133, 0.002780, 0.000120, 0.000029], dtype=float)

        Ein = np.clip(self.E_grid_in_history, 0.0, None).ravel()
        Eex = np.clip(self.E_grid_ex_history, None, 0.0).ravel()

        n = min(len(Ein), len(Eex), len(codes))
        Ein = Ein[:n]
        Eex = Eex[:n]
        codes = codes[:n].ravel()

        # Vectorized toll lookup: much faster than fancy indexing
        c = ENERGY_TOLL[codes - 1]
        abs_throughput = np.abs(Ein) + np.abs(Eex)

        base = float(np.dot(abs_throughput, c))
        vat_addon = float((vat - 1.0) * np.dot(Ein, c))
        total = base + vat_addon
        return total, base, vat_addon
    
    @timer
    def grid_tolls_spain(self, vat=1.21):
        power_cost, avg_kw = self.grid_power_toll_spain(vat=vat)
        energy_cost, energy_base, energy_vat = self.grid_energy_toll_spain(vat=vat)
        return power_cost + energy_cost
        # return {
        #     "power_cost_eur": power_cost,
        #     "avg_import_kw_by_period": avg_kw,
        #     "energy_cost_eur": energy_cost,
        #     "energy_base_eur": energy_base,
        #     "energy_vat_addon_eur": energy_vat,
        #     "total_tolls_eur": power_cost + energy_cost,
        # }
    
    # only for debug
    # def sanity_check_grid_signs(self):
    #     Ein = np.asarray(self.E_grid_in_history).ravel()
    #     Eex = np.asarray(self.E_grid_ex_history).ravel()
    #     if np.any(Ein < -1e-9):
    #         raise ValueError("E_grid_in_history has negative values (should be >=0).")
    #     if np.any(Eex > 1e-9):
    #         raise ValueError("E_grid_ex_history has positive values (expected <=0). You may need to flip sign.")

    def objective_emission(self) -> float:
        """
        Calculate total CO₂ emissions from grid electricity usage.

        Uses only grid imports (not net of export), assuming:
        - E_grid_in_history is in kWh
        - co2_emission_factor is in tons CO₂ per MWh
        """
        co2_factor = self.eval_param["co2_emission_factor"]
        grid_import = np.sum(self.E_grid_in_history) * 1e-3     #kwh->mwh
        return grid_import * co2_factor 

    # def grid_operational_cost(self): #, electricity_prices=None):
    #     """
    #     Calculate net grid operational cost using dynamic electricity prices.
        
    #     Args:
    #         E_grid_in_history (np.array): Array of imported energy (kWh) at each timestep
    #         E_grid_ex_history (np.array): Array of exported energy (kWh) at each timestep
    #         electricity_prices (np.array): Array of electricity prices (EUR/kWh) at each timestep
    #     Returns:
    #         float: Net grid operational cost (EUR)
    #     """
    #     # if electricity_prices is None:
    #     #     electricity_prices = self.electricity_prices
    #     E_grid_in_history = self.E_grid_in_history.flatten()
    #     E_grid_ex_history = self.E_grid_ex_history.flatten()
    #     electricity_prices = self.electricity_prices.flatten()
    #     # print("check the shape of the arrays:", E_grid_in_history.shape, E_grid_ex_history.shape, electricity_prices.shape)
    #     cost_per_timestep = (E_grid_in_history * electricity_prices) - (np.abs(E_grid_ex_history) * electricity_prices)
    #     return np.sum(cost_per_timestep)
    @timer
    def grid_operational_cost(self):
        """
        Calculate net grid operational cost using dynamic electricity prices.
        Returns:
            float: Net grid operational cost (EUR)
        """
        E_in = self.E_grid_in_history.ravel()
        E_ex = self.E_grid_ex_history.ravel()
        prices = self.electricity_prices.ravel()
        return float(np.dot(E_in - np.abs(E_ex), prices))
    


    @timer
    def lettuce_revenue(self, harvest_dw_arr):
        total_dw = np.sum(harvest_dw_arr)
        total_fw = total_dw * self.p["dw_fw"]
        area = self.p["A_cul"]
        L_revenue = total_fw * area * self.eval_param["lettuce_price"]
        return L_revenue

    def calculate_ppfd(self, num_lights):
        '''
        Define a function to calculate PPFD(P_light_fixed) from number of lights and PPF of each light.
        The equation is P_light_fixed = (num_lights * PPF_unit_light) / Area_cultivation

        '''
        PPF_unit_light = 168
        A_cultivation = self.p["A_cul"]
        P_light_fixed = (num_lights * PPF_unit_light) / A_cultivation
        return P_light_fixed
    

    def light_investment(self, design_para):
        light_init = design_para["num_light"] * self.eval_param["led_unit_cost"]
        
        return light_init


    def investment_cost(self, design_para, growing_days):
        C_init = (design_para["num_pv"] * self.eval_param["pv_unit_cost"] +
                  design_para["num_batt"] * self.eval_param["batt_unit_cost"] +
                  design_para["num_wind"] * self.eval_param["wind_unit_cost"])
        C_OM = 0.02 * C_init
        C_batt = design_para["num_batt"] * self.eval_param["batt_unit_cost"]
        discount_rate = self.eval_param["discount_rate"]
        R_batt = np.sum(C_batt / (1 + discount_rate) ** year
                     for year in range(5, self.eval_param["project_time"], 5))
        C_lifetime = C_init + np.sum(C_OM / (1 + discount_rate) ** year
                                  for year in range(1, self.eval_param["project_time"])) + R_batt
        total_days = self.eval_param["project_time"] * 365
        period_cost = C_lifetime / total_days * growing_days
        
        return C_lifetime, period_cost

    def investment_cost_with_light(self, design_para, growing_days):
        C_init = (design_para["num_pv"] * self.eval_param["pv_unit_cost"] +
                  design_para["num_batt"] * self.eval_param["batt_unit_cost"] +
                  design_para["num_wind"] * self.eval_param["wind_unit_cost"] +
                  design_para["num_light"] * self.eval_param["led_unit_cost"])
        C_OM = 0.02 * C_init         # Annual O&M cost (2% of initial investment cost)
        C_batt = design_para["num_batt"] * self.eval_param["batt_unit_cost"]
        C_light = design_para["num_light"] * self.eval_param["led_unit_cost"]
        discount_rate = self.eval_param["discount_rate"]

        R_batt = np.sum(C_batt / (1 + discount_rate) ** year
                     for year in range(5, self.eval_param["project_time"], 5))

        P_light_fixed = self.calculate_ppfd(design_para["num_light"]) # was "not +200"
        photo_time = self.DLI / P_light_fixed * 1e6 / 3600
        
        light_life_years_rounded = math.floor(self.light_life_hours / photo_time / 365)
        replacement_years = list(range(light_life_years_rounded, self.eval_param["project_time"] + 1, light_life_years_rounded))
        R_light = np.sum(C_light / (1 + discount_rate) ** year for year in replacement_years)

        C_lifetime = C_init + np.sum(C_OM / (1 + discount_rate) ** year
                                  for year in range(1, self.eval_param["project_time"])) + R_batt + R_light
        total_days = self.eval_param["project_time"] * 365
        period_cost = C_lifetime / total_days  * growing_days                                                               # this should be changed, should calculate by sum of the simulation results/cycle number
        return C_lifetime, period_cost

    # define a new function to calculate the margin profit
    
    # def calculate_annual_net_cash_flow(self, design_para, harvest_dw_arr):
    #     r = self.eval_param["discount_rate"]
    #     project_time = self.eval_param["project_time"]
    #     P_light_fixed = self.calculate_ppfd(design_para["num_light"])  # here it returns 0 when num_light = 0 after change the design parameter, so need to +200 to get real ppfd
    #     photo_time = self.DLI / P_light_fixed * 1e6 / 3600
    #     light_life_years_rounded = math.floor(36000 / photo_time / 365)
    #     replacement_years_light = list(range(light_life_years_rounded, project_time + 1, light_life_years_rounded))
    #     replacement_years_batt = list(range(5, project_time + 1, 5))
    #     print("replacement years light:",replacement_years_light)

    #     def crf(rate, years):
    #         return (rate * (1 + rate)**years) / ((1 + rate)**years - 1)

    #     crf_project = crf(r, project_time)
    #     crf_light = crf(r, light_life_years_rounded)
    #     crf_batt = crf(r, 5)
    #     crf_pv = crf(r, 30)
    #     crf_wind = crf(r, 30)

    #     C_pv = design_para["num_pv"] * self.eval_param["pv_unit_cost"]
    #     C_batt = design_para["num_batt"] * self.eval_param["batt_unit_cost"]
    #     C_wind = design_para["num_wind"] * self.eval_param["wind_unit_cost"]
    #     C_light = design_para["num_light"] * self.eval_param["led_unit_cost"]
    #     CAPEX = C_pv + C_batt + C_wind + C_light

    #     CAPEX_pv_wind = C_pv + C_wind
    #     annualized_CAPEX = C_pv * crf_pv + C_batt * crf_batt + C_wind * crf_wind + C_light * crf_light

    #     R_batt = np.sum(C_batt / (1 + r)**year for year in replacement_years_batt)
    #     R_light = np.sum(C_light / (1 + r)**year for year in replacement_years_light)
    #     annualized_replacement = (R_batt + R_light) * crf_project
    #     annual_om = 0.02 * CAPEX

    #     revenue = self.lettuce_revenue(harvest_dw_arr)
    #     net_grid_e_cost = self.grid_operational_cost()

  
    #     # this is the equation used before 5/12/2025
    #     annual_net_cash_flow_cal = -annualized_CAPEX + revenue - annual_om - net_grid_e_cost - annualized_replacement # annual net cash flow     
        
    #     op_cash_flow_cal = revenue - annual_om - net_grid_e_cost # annual operational cash flow   
        
        
    #       # # print the below items
    #     # print("CAPEX:", CAPEX)
    #     # print("Annualized CAPEX:", annualized_CAPEX)
    #     # print("revenue", revenue)   #per year
    #     # print("net_grid_e_cost", net_grid_e_cost) #per year
    #     # print("CAPEX_pv_wind", CAPEX_pv_wind*1.02)
    #     # print("annual_om", annual_om)
    #     # print("annualized_replacement", annualized_replacement)
    #     # # print("net_cash_flow_cal:", net_cash_flow_cal)   
    #     net_cash_flow = annual_net_cash_flow_cal  # or np.mean(...) if that makes more sense
    #     payback_time = CAPEX / net_cash_flow if net_cash_flow > 0 else float('inf')
    #     # Add a calcualtion of NPV:
    #     # NPV calculation assuming constant annual net cash flow
    #     NPV = -CAPEX + np.sum(annual_net_cash_flow_cal / (1 + r) ** t for t in range(1, project_time + 1))
        
    #     # --- Salvage Value Calculations ---
    #     salvage_value = 0

    #     # Battery: check if last replacement < project end
    #     last_batt_year = max(replacement_years_batt) if replacement_years_batt else 0
    #     years_used_batt = project_time - last_batt_year
    #     if years_used_batt < 5:
    #         salvage_batt = (1 - years_used_batt / 5) * C_batt
    #         salvage_value += salvage_batt / (1 + r)**project_time

    #     # Light: same logic, dynamic life
    #     last_light_year = max(replacement_years_light) if replacement_years_light else 0
    #     light_life = light_life_years_rounded
    #     years_used_light = project_time - last_light_year
    #     if years_used_light < light_life:
    #         salvage_light = (1 - years_used_light / light_life) * C_light
    #         salvage_value += salvage_light / (1 + r)**project_time

    #     # Final NPV with salvage
    #     NPV += salvage_value


        
    #     return net_cash_flow, payback_time, NPV
    
    @timer
    def calculate_annual_net_cash_flow(self, design_para, harvest_dw_arr):
        r = self.eval_param["discount_rate"]
        project_time = self.eval_param["project_time"]

        def crf(rate, years):
            return (rate * (1 + rate) ** years) / ((1 + rate) ** years - 1)

        # CAPEX per component
        C_pv = design_para["num_pv"] * self.eval_param["pv_unit_cost"]
        C_batt = design_para["num_batt"] * self.eval_param["batt_unit_cost"]
        C_wind = design_para["num_wind"] * self.eval_param["wind_unit_cost"]
        C_light = design_para["num_light"] * self.eval_param["led_unit_cost"]
        CAPEX = C_pv + C_batt + C_wind + C_light

        # CRF by component
        crf_pv = crf(r, 30)
        crf_wind = crf(r, 30)
        crf_batt = crf(r, 5)
        crf_light = crf(r, 6)  # fixed at 6 years

        # ✅ Annualized CAPEX via CRF only
        annualized_CAPEX = (
            C_pv * crf_pv +
            C_batt * crf_batt +
            C_wind * crf_wind +
            C_light * crf_light
        )

        # ✅ No explicit replacement costs (batt & light already included in CRF)
        annualized_replacement = 0

        # O&M cost
        annual_om = 0.02 * CAPEX

        # Revenue & grid cost
        revenue = self.lettuce_revenue(harvest_dw_arr)
        net_grid_e_cost = self.grid_operational_cost()

        # Annual net cash flow
        net_cash_flow = -annualized_CAPEX + revenue - annual_om - net_grid_e_cost - annualized_replacement
        payback_time = CAPEX / net_cash_flow if net_cash_flow > 0 else float("inf")

        # NPV (constant annual net cash flow assumption)
        NPV = -CAPEX + np.sum(
            net_cash_flow / (1 + r) ** t
            for t in range(1, project_time + 1)
        )

        # ✅ No salvage needed for batt & light (already covered by CRF)
        return net_cash_flow, net_grid_e_cost, NPV

    @timer
    def calculate_annual_net_cash_flow_with_fixed_cost_grid(self, design_para, harvest_dw_arr):
        r = self.eval_param["discount_rate"]
        project_time = self.eval_param["project_time"]

        def crf(rate, years):
            return (rate * (1 + rate) ** years) / ((1 + rate) ** years - 1)

        # CAPEX per component
        C_pv = design_para["num_pv"] * self.eval_param["pv_unit_cost"]
        C_batt = design_para["num_batt"] * self.eval_param["batt_unit_cost"]
        C_wind = design_para["num_wind"] * self.eval_param["wind_unit_cost"]
        C_light = design_para["num_light"] * self.eval_param["led_unit_cost"]
        CAPEX = C_pv + C_batt + C_wind + C_light

        # CRF by component
        crf_pv = crf(r, 30)
        crf_wind = crf(r, 30)
        crf_batt = crf(r, 5)
        crf_light = crf(r, 6)  # fixed at 6 years

        # ✅ Annualized CAPEX via CRF only
        annualized_CAPEX = (
            C_pv * crf_pv +
            C_batt * crf_batt +
            C_wind * crf_wind +
            C_light * crf_light
        )

        # ✅ No explicit replacement costs (batt & light already included in CRF)
        annualized_replacement = 0

        # O&M cost
        annual_om = 0.02 * CAPEX

        # Revenue & grid cost
        revenue = self.lettuce_revenue(harvest_dw_arr)
        net_grid_e_cost = self.grid_operational_cost()

        ####### add one term for fixed grid cost
        # input: design_para["num_light"], output: corresponding fixed grid cost
        num_light = design_para["num_light"]
        fixed_cost_table = {
            14812: 63729.0,
            15553: 67245.0,
            16294: 69354.6,
            17035: 72870.6,
            17776: 76386.6,
            18517: 79902.6,
            19258: 83418.6,
            19999: 86231.4,
            20740: 89747.4,
            21481: 91857.0,
        }
        # fixed_cost_table = {
        #     14812: 6666.0,
        #     15553: 6666.0,
        #     16294: 6666.0,
        #     17035: 6666.0,  
        #     17776: 6666.0,
        #     18517: 6666.0,
        #     19258: 6666.0,
        #     19999: 6666.0,
        #     20740: 6666.0,
        #     21481: 6666.0,
        # }
        if num_light not in fixed_cost_table:
            raise ValueError(
                f"Invalid num_light={num_light}. "
                f"Expected one of {list(fixed_cost_table.keys())}."
            )
        fixed_grid_cost = fixed_cost_table[num_light]
        # grid_tolls_spain = self.grid_tolls_spain(vat=1.21)

        # Annual net cash flow
        net_cash_flow = -annualized_CAPEX + revenue - annual_om - net_grid_e_cost 
        - annualized_replacement - fixed_grid_cost # - grid_tolls_spain
        payback_time = CAPEX / net_cash_flow if net_cash_flow > 0 else float("inf")

        # NPV (constant annual net cash flow assumption)
        NPV = -CAPEX + np.sum(
            net_cash_flow / (1 + r) ** t
            for t in range(1, project_time + 1)
        )

        # ✅ No salvage needed for batt & light (already covered by CRF)
        return net_cash_flow, net_grid_e_cost, NPV
    

        
        
    def calculate_lcoe_re(self, design_para):
        """
        Calculate the Levelized Cost of Electricity (LCOE) for the renewable energy system (PV + Wind [+ Battery optional])
        Units: LCOE in $/kWh
        """
        r = self.eval_param["discount_rate"]
        project_time = self.eval_param["project_time"]

        # --- Capital costs ---
        C_pv = design_para["num_pv"] * self.eval_param["pv_unit_cost"]
        C_wind = design_para["num_wind"] * self.eval_param["wind_unit_cost"]
        C_batt = design_para["num_batt"] * self.eval_param["batt_unit_cost"]  # optional

        # --- Capital Recovery Factors ---
        def crf(rate, years):
            return (rate * (1 + rate)**years) / ((1 + rate)**years - 1)

        crf_project = crf(r, project_time)
        crf_pv = crf(r, 30)
        crf_wind = crf(r, 30)
        crf_batt = crf(r, 5)

        # --- Annualized capital cost ---
        annual_cost_pv = C_pv * crf_pv
        annual_cost_wind = C_wind * crf_wind
        annual_cost_batt = C_batt * crf_batt

        # --- Replacement costs for battery ---
        replacement_years_batt = list(range(5, project_time , 5))
        R_batt = sum(C_batt / (1 + r) ** year for year in replacement_years_batt)
        annualized_replacement_batt = R_batt * crf_project

        # --- O&M cost (assume 2% of RE CapEx) ---
        annual_om_re = 0.02 * (C_pv + C_wind + C_batt)

        # --- Total annual cost of RE system ---
        total_annual_cost_re = (
            annual_cost_pv + annual_cost_wind + annual_cost_batt +
            annualized_replacement_batt + annual_om_re
        )

        # --- Annual RE electricity generation (in kWh) ---
        annual_generation_kwh = np.sum(self.E_re_generation)


        # --- LCOE Calculation --- 
        if annual_generation_kwh > 0:
            lcoe = total_annual_cost_re / annual_generation_kwh  # in $/kWh
        else:
            lcoe = 1  # Assign fixed fallback LCOE for infeasible/no-generation case


        return lcoe

    def lcoe_comp(self, design_para):
        """
        Calculate the Levelized Cost of Electricity (LCOE) for the each renewable energy source (PV + Wind [+ Battery optional])
        Units: euro/kWh
        """
        r = self.eval_param["discount_rate"]
        project_time = self.eval_param["project_time"]

        # --- Capital costs ---
        C_pv = design_para["num_pv"] * self.eval_param["pv_unit_cost"]
        C_wind = design_para["num_wind"] * self.eval_param["wind_unit_cost"]
        C_batt = design_para["num_batt"] * self.eval_param["batt_unit_cost"]  # optional

        # --- Capital Recovery Factors ---
        def crf(rate, years):
            return (rate * (1 + rate)**years) / ((1 + rate)**years - 1)

        crf_project = crf(r, project_time)
        crf_pv = crf(r, 30)
        crf_wind = crf(r, 30)
        crf_batt = crf(r, 5)

        # --- Replacement costs for battery ---
        replacement_years_batt = list(range(5, project_time, 5))
        R_batt = sum(C_batt / (1 + r) ** year for year in replacement_years_batt)
        annualized_replacement_batt = R_batt * crf_project

        # --- Annualized capital cost ---
        annual_cost_pv = C_pv * crf_pv + 0.02* C_pv  # 2% O&M cost included
        annual_cost_wind = C_wind * crf_wind + 0.02* C_wind  # 2% O&M cost included
        annual_cost_batt = C_batt * crf_batt + 0.02* C_batt + annualized_replacement_batt  # 2% O&M and replacement cost included


        # --- Annual RE electricity generation by component (in kWh) ---
        annual_generation_kwh_curtailed = np.sum(self.E_re_generation) - np.sum(np.abs(self.E_curtail))  # total used renewable energy
        annual_generation_pv_kwh = np.sum(self.E_pv_generation)
        annual_generation_wind_kwh = np.sum(self.E_wind_generation)
        annual_batt_to_load_kwh = np.sum(self.E_batt_to_load)

        # # --- LCOE Calculation ---
        # if annual_generation_kwh > 0:
        #     lcoe = total_annual_cost_re / annual_generation_kwh  # in $/kWh
        # else:
        #     lcoe = float('inf')  # to avoid division by zero
        # --- LCOE Calculation --- 
        if annual_generation_pv_kwh > 0:
            lcoe_pv = annual_cost_pv / annual_generation_pv_kwh
        else:
            lcoe_pv = float('inf')  # or 0 if you want a non-infinite default

        if annual_generation_wind_kwh > 0:
            lcoe_wind = annual_cost_wind / annual_generation_wind_kwh
        else:
            lcoe_wind = float('inf')

        if annual_batt_to_load_kwh > 0:
            lcoe_batt = annual_cost_batt / annual_batt_to_load_kwh
        else:
            lcoe_batt = float('inf')
            
        if annual_generation_kwh_curtailed > 0:
            lcoe_curtailed = (annual_cost_pv + annual_cost_wind) / annual_generation_kwh_curtailed
        else:
            lcoe_curtailed = float('inf')

        return lcoe_pv, lcoe_wind, lcoe_batt, lcoe_curtailed

        return lcoe_pv, lcoe_wind, lcoe_batt

    def other_op_revenue_cost(self, h, m, y_values, u_values):
        revenue_lettuce = (y_values[m, 0] * self.p["A_cul"] * self.p["dw_fw"]) * self.p["productPrice2"] + self.p["productPrice1"]
        revenue_electricity = np.sum(np.abs(self.E_grid_ex_history)) * self.eval_param["price_sell_electricity"]
        # cost_electricity = sum(E_grid_in_history) * self.eval_param["price_buy_electricity"]
        net_grid_e_cost = self.grid_operational_cost()

        cost_CO2 = sum(u_values[:, 0] * h) * self.p["A_floor"] * self.p["co2Cost"]
        total_revenue = revenue_lettuce + revenue_electricity
        total_cost = cost_CO2 + net_grid_e_cost
        return total_revenue, total_cost, net_grid_e_cost

    # def re_frac(self, E_grid_in_history, E_grid_ex_history, E_load):
    #     return 1 - (sum(E_grid_in_history)-sum(E_grid_ex_history)) / sum(E_load)   # previous calculation, also results in a 50% max ref, sometimes nagetive...
    
    # def re_frac(self, direct_renewable_to_load, battery_to_load):
    #     renewable_used = np.sum(direct_renewable_to_load) + np.sum(battery_to_load)
    #     total_load = np.sum(self.E_load_history)
    #     print("renewable:", np.max(direct_renewable_to_load), direct_renewable_to_load)
    #     # print("battery used:", np.max(battery_to_load), battery_to_load)

    #     # print("total load:", np.max(E_load_history), E_load_history)
    #     return renewable_used / total_load if total_load > 0 else 0
    
    @timer
    def cal_refrac(self):
        """
        Calculate the fraction of renewable energy used in the grid.
        Args:
            grid_in (np.array): Array of electricity imported from grid at each timestep
            grid_ex (np.array): Array of electricity exported to grid at each timestep
            E_load (np.array): Array of total load at each timestep
        Returns:
            renewable_frac (float): Fraction of renewable energy used in the grid
        """
        return 1 - (np.sum(self.E_grid_in_history) - np.sum(np.abs(self.E_grid_ex_history))) / np.sum(self.E_load_history)

    # add a function to calculate the self-consumption rate of electricity
    # def self_suf_ratio(self):
    #     """
    #     Calculate the self_sufficient_rate of electricity.
    #     Args:   
    #         E_load_history (np.array): Array of total load at each timestep
    #     Returns:
    #         self_consum_rate (float): Self-consumption rate of electricity,
    #         which is (E_re_generation - E_grid_ex_history) / E_load_history
    #     """
    #     self_suf_ratio = (np.sum(self.E_re_generation) - np.sum(np.abs(self.E_grid_ex_history))) / np.sum(self.E_load_history) 
    #     return self_suf_ratio
    
    @timer
    def self_suf_ratio(self):
        """
        Calculate the self-sufficiency ratio of renewable electricity.
        Definition:
            = (RE generation used to meet demand) / (Total electricity demand)
            = (RE generation - grid export - curtailment) / total load
        """
        numerator = np.sum(self.E_re_generation) - np.sum(np.abs(self.E_grid_ex_history)) - np.sum(self.E_curtail)
        denominator = np.sum(self.E_load_history)

        if denominator == 0:
            return 0.0  # or np.nan

        return numerator / denominator
    
    # def self_consumption_rate(self):
    #     """
    #     Calculate the self-consumption ratio of renewable electricity.

    #     Definition:
    #         Self-consumption ratio = (RE generation used on-site) / (Total RE generation)
    #         = (RE generation - grid export) / RE generation

    #     Returns:
    #         self_consum_rate (float): Self-consumption ratio (0 to 1).
    #         Returns 0.0 if there is no RE generation to avoid division by zero.
    #     """
    #     total_re_gen = np.sum(self.E_re_generation)
    #     total_export = np.sum(np.abs(self.E_grid_ex_history))

    #     if total_re_gen == 0:
    #         return 0.0  # or np.nan depending on how you want to treat no-generation cases

    #     self_consum_rate = (total_re_gen - total_export) / total_re_gen
    #     return self_consum_rate

    # def self_consumption_rate(self):
    #     """
    #     Calculate the self-consumption rate of renewable electricity.
    #     Definition:
    #         = (RE generation used on-site) / (Total RE generation)
    #         = (RE generation - grid export - curtailment) / RE generation
    #     """
    #     total_re_gen = np.sum(self.E_re_generation)
    #     total_export = np.sum(np.abs(self.E_grid_ex_history))
    #     total_curtail = np.sum(self.E_curtail)

    #     if total_re_gen == 0:
    #         return 0.0

    #     self_consum_rate = (total_re_gen - total_export - total_curtail) / total_re_gen
    #     return self_consum_rate
    
    @timer
    def self_consumption_rate(self):
        """
        Calculate the self-consumption rate of renewable electricity:
            = (RE generation - export - curtailment) / RE generation

        Assumptions:
            - E_re_generation >= 0  (PV + wind)
            - E_grid_ex_history: positive = import, negative = export
            - E_curtail >= 0
        """

        total_re_gen = np.sum(self.E_re_generation)

        # --- Export: only count negative values ---
        #   If export = negative, take its absolute value
        export_only = self.E_grid_ex_history[self.E_grid_ex_history < 0]
        total_export = -np.sum(export_only)   # make positive

        # --- Curtailment: clip negatives, ensure non-negative ---
        total_curtail = np.sum(np.clip(self.E_curtail, 0, None))

        if total_re_gen <= 0:
            return 0.0

        # --- Self-consumption calculation ---
        scr = (total_re_gen - total_export - total_curtail) / total_re_gen

        # Numerical safety: SCR cannot exceed 1 or drop below 0
        scr = max(0.0, min(1.0, scr))

        return scr



    # add a function to calculate the self-sufficiency rate of electricity
    # def self_suff_rate(self):
    #     """
    #     Calculate the self-sufficiency rate of electricity.
    #     Args:   
    #         E_load_history (np.array): Array of total load at each timestep
    #     Returns:
    #         self_suff_rate (float): Self-sufficiency rate of electricity
    #     """
    #     return np.sum(self.E_load_history) / np.sum(self.E_grid_in_history + np.abs(self.E_grid_ex_history))
    
    @timer
    def curtail_ratio(self):
        """
        Calculate the curtailment ratio of renewable energy.
        Args:
            E_curtail (np.array): Array of curtailed renewable energy at each timestep
        Returns:
            curtail_ratio (float): Ratio of curtailed energy to total renewable generation
        """
        total_gen = np.sum(self.E_re_generation)
        total_curtail = np.sum(np.abs(self.E_curtail))
        return total_curtail / total_gen if total_gen > 0 else 0.0
    
    def deficit_ratio(self):
        """
        Calculate the deficit ratio of renewable energy.
        Args:
            E_deficit (np.array): Array of deficit renewable energy at each timestep
        Returns:
            deficit_ratio (float): Ratio of deficit energy to total load
        """
        total_load = np.sum(self.E_load_history)
        total_deficit = np.sum(np.abs(self.E_deficit))
        return total_deficit / total_load if total_load > 0 else 0.0

    # calculate area required by pv
    def calculate_area_pv(self, design_para):
        """
        Calculates the land area required for a PV array given the tilt angle and sun elevation angle.
        """
        tilt_rad = np.radians(self.other_p["surface_tilt"])
        sun_elevation_rad = np.radians(self.other_p["sun_elevation"])
        # compute row spacing to avoid shading
        row_spacing = self.eval_param["pv_width"] * np.sin(tilt_rad) / np.tan(sun_elevation_rad)
        # Total area required
        area_pv = design_para["num_pv"] * self.eval_param["pv_length"] * (self.eval_param["pv_width"]* np.cos(tilt_rad) + row_spacing)

        # print("area_pv (m^2):", area_pv)
        return area_pv
  
    def calculate_area_wind(self, design_para):
        """
        Calculates the land area required for wind turbines based on the number of turbines and spacing requirements.
        """
        rotor_diameter = 32  # typical rotor diameter in meters
        spacing_factor = 3.2  # 3-5 rotor diameters between turbines
        area_per_turbine = (spacing_factor * rotor_diameter) ** 2
        area_wind = design_para["num_wind"] * area_per_turbine

        # print("area_wind (m^2):", total_area_wind)
        return area_wind
      
# import numpy as np
# from models.ModelFunctions_final_changed import DefineParameters, OtherParameters

# if __name__ == "__main__":
#     # Example configuration dictionaries
#     eval_param = {
#         "co2_emission_factor": 400,
#         "pv_unit_cost": 1000,
#         "batt_unit_cost": 500,
#         "wind_unit_cost": 1500,
#         "led_unit_cost": 200,
#         "discount_rate": 0.05,
#         "project_time": 30,
#         "lettuce_price": 1.5,
#         "price_sell_electricity": 0.05,
#         "price_buy_electricity": 0.15
#     }

#     p = DefineParameters()
#     other_p = OtherParameters()
#     DLI = 16.99  # Daily Light Integral
#     electricity_prices = np.random.rand(90)

#     # Create class instance
#     econ = EnergyLettuceEconomics(eval_param, p, other_p, DLI, electricity_prices)

#     # Example input data
#     design_para = {"num_pv": 10, "num_batt": 5, "num_wind": 3, "num_light": 20}
#     E_in = np.random.rand(90)
#     E_ex = np.random.rand(90)
#     y_vals = np.random.rand(10)
#     growing_days = 90

#     # Call some methods
#     emission = econ.objective_emission(E_in, E_ex)
#     net_cost = econ.grid_operational_cost(E_in, E_ex)
#     revenue = econ.lettuce_revenue(y_vals)
#     inv_cost, period_cost = econ.investment_cost(design_para, growing_days)
#     light_init_cost, replacements = econ.light_investment(design_para)

#     print("CO2 Emission (kg):", emission)
#     print("Grid Operational Cost (€):", net_cost)
#     print("Lettuce Revenue (€):", revenue)
#     print("Initial Investment Cost (€):", inv_cost)
#     print("Period-Specific Cost (€):", period_cost)
#     print("Initial Light Investment (€):", light_init_cost)
#     print("Light Replacements Required:", replacements)
    
