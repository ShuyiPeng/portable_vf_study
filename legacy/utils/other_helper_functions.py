# %%
from scipy.io import loadmat
import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta
from copy import copy
import matplotlib.pyplot as plt
import csv
import seaborn as sns

import legacy.utils.helper_functions_vf as hf
from legacy.models.ModelFunctions_final_changed import DefineParameters, F, PVmoduleParameters, PVsysParameters, BattParameters, WindParameters, OtherParameters, EvalParameters
from legacy.models.Pv_power_model_class import PvPowerCalculator
from legacy.models.wind_energy_model_class import WindEnergyModel

'''distubance: "0,Solar Radiation",     "1,CO2",                 "2,Temperature",              "3,RH"
state variables: "0,dry mass",       "1,CO2",                 "2,Temperature",             "3,AH"  (the state is AH instead of RH, the unit of AH is kg/m^-3)
control input:   "0,supply rate of CO2",         "1,Q_cool_s",           "2,phi_cond_h"    "3,P_l"
aggregate all parameters related to design in both model and simulation

Evaluation functions
 Calculate the objectives using a function, including emission model (CO2 emission of grid electricity, carbon emission of pv and battery themselves, CO2 obsorbed by plants), economic model (LCOE, operation cost).
    # category 1: technical: renewble energy fraction
    # category 2: economic: total investment, electricity cost and revenue, yield revenue  
    # category 3: environmental: land area, co2 emission (grid), co2 emission of facilities (pv and batt), co2 injection (negative)
'''

def extract_scalar(value):
    if isinstance(value, (list, tuple)) and len(value) == 1:  # If it's a single-element list/tuple
        return value[0]
    elif isinstance(value, np.ndarray) and value.size == 1:  # If it's a single-element NumPy array
        return value.item()
    return value  # Return as is if it's already a scalar


def co2dens2ppm(temp, dens):
    """
    CO2DENS2PPM Convert CO2 density [kg m^{-3}] to molar concetration [ppm] 

    Usage: 
    ppm = co2dens2ppm(temp, dens)
    Inputs:
    temp        given temperatures [°C] (numeric vector)
    dens        CO2  in air [kg m^{-3}] (numeric vector)
    Inputs should have identical dimensions
    Outputs:
    ppm         Molar concentration of CO2 in air [ppm] (numerical vector)

    calculation based on ideal gas law pV=nRT, pressure is assumed to be 1 atm

    David Katzin, Wageningen University
    david.katzdensityin@wur.nl
    """

    # molar gas constant [J mol^{-1} K^{-1}]
    R = 8.3144598
    # conversion from Celsius to Kelvin [K]
    C2K = 273.15
    # molar mass of CO2 [kg mol^-{1}]
    M_CO2 = 44.01e-3
    # pressure (assumed to be 1 atm) [Pa]
    P = 101325

    return 1e6*R*(temp+C2K)*dens/(P*M_CO2)

def vaporDens2rh(temp, vaporDens):
    """
    vaporDens2rh Convert vapor density [kg{H2O} m^{-3}] to relative humidity [%]

    Usage:
    rh = vaporDens2rh(temp, vaporDens)
    Inputs:
    temp        given temperatures [°C] (numeric vector)
    vaporDens   absolute humidity [kg{H20} m^{-3}] (numeric vector)
    Inputs should have identical dimensions
    Outputs:
    rh          relative humidity [%] between 0 and 100 (numeric vector)

    Calculation based on 
    http://www.conservationphysics.org/atmcalc/atmoclc2.pdf

    David Katzin, Wageningen University
    david.katzin@wur.nl
    """
    # constants
    # molar gas constant [J mol^{-1} K^{-1}]
    R = 8.3144598 
    # conversion from Celsius to Kelvin [K]
    C2K = 273.15  
    # molar mass of water [kg mol^-{1}]
    Mw = 18.01528e-3  
    
    # parameters used in the conversion
    # default value is [610.78 238.3 17.2694 -6140.4 273 28.916]
    p = [610.78, 238.3, 17.2694, -6140.4, 273, 28.916]
    
    # Saturation vapor pressure of air in given temperature [Pa]
    satP = p[0]*np.exp(p[2]*temp/(temp+p[1])) 
    # convert to relative humidity using the ideal gas law pV=nRT => n=pV/RT 
    # so n=p/RT is the number of moles in a m^3, and Mw*n=Mw*p/(R*T) is the 
    # number of kg in a m^3, where Mw is the molar mass of water.
    relhumid = 100*R*(temp+C2K)/(Mw*satP)*vaporDens
    # if np.isinf(relhumid).any():
    #     print(temp, vaporDens)
    return np.clip(relhumid, a_min=0, a_max=100)

def DefineConfigParameters(design_para):
    config = {}
    # Structural parameters
    config["A_length"] = 50
    config["A_length"] = 50
    # config["A_height"] = 4.5
    # config["A_factor"] = 5  # Investigate common farm design practices
    config["U_value"] = 0.2  # Thermal transmittance (Weidner et al., 2021), not necessary
    # update the dependent parameters
    # config["A_floor"] = config["width"] * config["length"]  
    # config["A_surface"] = config["A_floor"] + 2 * (config["width"] + config["length"]) 
    # config["A_cul"] = config["A_factor"] * config["A_floor"]
    # config["CO2cap"] = (config["A_floor"] * config["A_height"]) / (config["A_floor"] * config["A_factor"])
    # config["H2Ocap"] = (config["A_floor"] * config["A_height"]) / (config["A_floor"] * config["A_factor"])
    # Energy system parameters
    config["p_max"] = 340 # * design_para["num_pv"]  # PV Max power output in Watts (each unit is 340 Wp)
    config["battery_capacity"] = 4800 * design_para["num_batt"]  # Battery capacity in Wh (each unit is 4.8 kWh)
    config["P_battery_max"] = 4800 * design_para["num_batt"]  # Max charging power to the battery in Watts
    config["P_wind_rated"] = 0.3e6 * design_para["num_wind"]  # Rated power of the wind turbine in Watts
    return config


# %%
# update some of the parameters in dictionary p from DefineParameters function
def update_parameters(p, design_para, config, new_eval_p):
    module_parameters = PVmoduleParameters()
    system_parameters = PVsysParameters()
    batt_param = BattParameters()
    wind_param = WindParameters()
    other_p = OtherParameters()
    eval_param = EvalParameters()
    config = DefineConfigParameters(design_para)

    p.update(design_para)
    p.update(config)

    module_parameters.update(config)
    system_parameters.update(config)
    batt_param.update(config)
    
    eval_param.update(new_eval_p)
    
    return p, module_parameters, system_parameters, batt_param, other_p, eval_param
# %%
def f(state, action, d, h, p): # fourth Runge-Kutta
    k1 = F(state, action, d, p)
    k2 = F(state + h/2 *k1, action, d, p)
    k3 = F(state + h/2 *k2, action, d, p)
    k4 = F(state + h *k3, action, d, p)
    state += h/6*(k1+ 2*k2 + 2*k3 + k4)
    return state

# %%
def g(state): #convert unit
    y = np.array([state[0], #// original: 1e3*state[0] // unit is kg
            co2dens2ppm(state[2],state[1]), #why 1e-3? // original: 1e-3*co2dens2ppm(state[2],state[1])
            state[2],
            vaporDens2rh(state[2], state[3])], dtype=np.float32)
    return y    

def unscale_inputs(u, min_action, max_action):
     return (u + 1)*(max_action - min_action)/2 + min_action
# %%
def power_load(u, p):
    """
    Calculate the total power load for the system based on control inputs and parameters.

    Parameters:
    - u: Control inputs (array or list)
    - p: Parameters (dictionary)

    Returns:
    - power_total: Total power load (float)
    """
    power_lighting = u[3] /p["eta_led"] #checked
    power_cooling = (
        u[1] / p['COP_cool']
        if u[1] > 0
        else abs(u[1])  / p['COP_heat'] # power per clativation area #checked
    )
    power_dehum = u[2]/ p['dehum_eff']/3.6*1e6 #[W m-2]per cultivation area  #checked. p['dehum_eff'] = 0.292 # [kWh/L] rated efficacy of the dehumidifier
    # u[2] * p["A_cul"] * p['dehum_eff'] *1000/3600 # * 0.25   #kg/m^2/s=L/m^2/s , *m^2, *kWh/L ====kwh/s--> W, need *1000/3600 to convert to W
    
    # power_total = power_lighting + power_cooling + power_dehum
    # power_total_vf = power_total * p["A_cul"]
    power_light = power_lighting * p["A_cul"]
    power_hvac = (power_cooling + power_dehum) * p["A_floor"]
    power_total_vf = power_light + power_hvac
    # print("proportion of lighting power:", power_light/power_total_vf*100, power_light, power_hvac)
    
    # print("percent of none lighting power:", (power_cooling+power_dehum)/power_total*100, power_lighting, power_dehum, power_cooling)
    return power_total_vf

def pro_act_D_i(y, nu, carbon_range, temp_range, humidity_range, u_prev_c, u_prev_t, u_prev_h):

    u = np.zeros(nu)

    # # CO2: 
    if y[1] < (carbon_range[0]):  #+100
        # u[0] = np.clip(u_prev_c[-1] + 0.01, 0, 1)
        u[0] = np.clip(u_prev_c[-1] + 0.005*(carbon_range[0]-y[1]), -1, 1) #increase 0.01
        u_prev_c.append(u[0])

    elif y[1] > (carbon_range[1]): #-100
        # u[0] = -1 #0
        u[0] = np.clip(u_prev_c[-1] - 0.01*(y[1] - carbon_range[1]), -1, 1) #-0.01
        u_prev_c.append(u[0])

    else:
        # u[0] = u_prev_c[-1]
        u[0] = np.clip(0.0005*(((y[1] - (carbon_range[0]))/((carbon_range[1]) -
                    (carbon_range[0]))) *0.3 + (-1)), -1, 1)
        u_prev_c.append(u[0])
    

    # Q_cool_s, u[1] related to temperature
    if y[2] > (temp_range[1][1]): #-1
        u[1] = np.clip(u_prev_t[-1] + 0.06*(y[2] - temp_range[1][1]+1), 0, 1) #+ 0.05
        u_prev_t.append(u[1])

    elif y[2] < (temp_range[1][0]): #+1
        #u[1] = 0 #0.09
        # u[1] = 0.45
        u[1] = 0.52
        u_prev_t.append(u[1])

    else:
        # u[1] = u_prev_t[-1]
        u[1] = np.clip(0.21*(((y[2] - (temp_range[1][0]))/((temp_range[1][1]) -
                    (temp_range[1][0]))) *2 + (-1))+0.73, -1, 1)
        # u[1] = 0.8
    
    # phi_cond_h, u[2] related to humidity
    if y[3] > (humidity_range[1] -2): #-1
        # u[2] = 1
        u[2] = np.clip(u_prev_h[-1] + 0.13*(y[3] - humidity_range[1]), -1, 1) #0.1
        u_prev_h.append(u[2])

    elif y[3] < (humidity_range[0]+2): #+1
        u[2] = -0.80
        u_prev_h.append(u[2])

    else:
        u[2] = ((y[3] - (humidity_range[0] +2))/6 ) *2 + (-1)
        # u[2] = u_prev_h[-1]
        # u[2] = 0

    u[3] = 1 

    return u

def pro_act_N_i(y, nu, carbon_range, temp_range, humidity_range, u_prev_c, u_prev_t, u_prev_h):
    # global u_prev_c
    # global u_prev_t
    # global u_prev_h
    u = np.zeros(nu)
    
    # CO2: if CO2 is below 350ppm, operate CO2 injection
    if y[1] < (carbon_range[0]): #+100
        u[0] = np.clip(u_prev_c[-1] + 0.05, -1, 1)
        u_prev_c.append(u[0])
        
    elif y[1] > (carbon_range[1]): #-100
        u[0] = -1
        u_prev_c.append(u[0])

    else:
        u[0] = u_prev_c[-1]
        u_prev_c.append(u[0])

    # Q_cool_s, u[1] related to temperature  18-20
    if y[2] > (temp_range[0][1]-1):  #19
        # u[1] = np.clip(u_prev_t[-1] + 0.03*(y[2] - temp_range[0][1]-3) , 0, 1)#+0.05 (game factor (array))
        u[1] = 0.05
        u_prev_t.append(u[1])

    elif y[2] < (temp_range[0][0]):  #+1(margin/deadband)
        u[1] = np.clip(u_prev_t[-1] - 0.06*(temp_range[0][0]+1 - y[2]), -1, 0) #-0.05
        u_prev_t.append(u[1])

    else:
        # u[1] = 0
        u[1] = 0.1 *(((y[2] - (temp_range[0][0]))/((temp_range[0][1]) - (temp_range[0][0]))) *2 + (-1))
        u_prev_t.append(u[1])
    
    # phi_cond_h, u[2] related to humidity #add difference
    if y[3] > (humidity_range[1] -2): #-1
        u[2] = np.clip(u_prev_h[-1] + 0.1*(y[3] - humidity_range[1]+2), -1, 1)
        u_prev_h.append(u[2])

    elif y[3] < (humidity_range[0] +2): #+1
        u[2] = -1
        u_prev_h.append(u[2])

    else:
        #u[2] = u_prev_h[-1]
        u[2] = ((y[3] - (humidity_range[0] +2))/6 ) *2 + (-1)
        # u[2] = 0

    u[3] = -1
    
    return u

def calculate_pv_output(pv_data, latitude, longitude, ground_albedo, surface_tilt,
                        altitude, solar_constant, module_parameters, system_parameters):
    '''
    calculate the pv power output trajectory and use the data later in the integrated simulation, 
    by calling data point one by one from the trajectory array
    '''
    pv_power_output_array = []
    for i in range(len(pv_data[:, -1])):
        timestamp = pv_data[i, -1]
        ghi = pv_data[i, 0]                # Get the radiation value for specific iteration
        T_ambient = pv_data[i, 1]          # Get the ambient temperature for specific iteration

        # Create instance of PvPowerCalculator
        calculator = PvPowerCalculator(
            latitude,
            longitude,
            ground_albedo,
            timestamp,
            ghi,
            surface_tilt,
            surface_azimuth=180,
            T_ambient=T_ambient,
            altitude=altitude,
            solar_constant=solar_constant,
            module_parameters=module_parameters,
            system_parameters=system_parameters
        )

        pv_power_output = calculator.calculate_total_power_output()        # Calculate total PV power output
        pv_power_output_array.append(pv_power_output)        # Store result in array

    return pv_power_output_array

# %%
def transipiration(p, x):
    phi_transp_h = (1-np.exp(-p["c_pl_d"] * x[0])) * p["c_v_pl_ai"] * (((p["v,0"]*p["v,1"]*p["H2O"])/(p["c_R"] * (x[2] + p["c_T_abs"]))) * np.exp((p["satH2O2"]*x[2])/(p["satH2O3"]+x[2]))-x[3] )
    return max(0, p["A_factor"]*phi_transp_h)

def LUE_denominator(p, x):
    return (1-np.exp(-p["c_pl_d"] * x[0]))

def cal_ver(p, x):
    g_c = -p["photCO2_1"] * x[2]**2 + p["photCO2_2"] * x[2] - p["photCO2_3"]
    g_co2 = 1/((1/p["c_bnd"]) + (1/p["c_stm"]) + (1/g_c))
    return g_c, g_co2
    #return g_c

def cond_heat(p, u):
    Q_cond_heat = p["lat_water"]*u[2] 
    return Q_cond_heat

# %%
# 每个array的长度基于收获时间
def plot_simulation_results(m, h, pv_power_traj, wind_power_traj, p_load, P_battery_history, P_grid_ex_history, P_grid_in_history, P_excess, P_deficit, y_check, u_values, nor_u, d_values, x_values):
    try:
        # Convert time vector to days
        # time = t / 3600 / 24    # where n_days, h = 40, 15 * 60, L = n_days * c, c = 86400s/day t = np.arange(0, L+h, h)
        time = np.arange(0, (m)*h, h) / 3600 / 24     #(m+1)*h/3600/24 

        # Plot PV Power and Load Power with adjusted styles
        plt.figure(figsize=(14, 6))
        plt.step(time, pv_power_traj[:m,0]*1e-6, label='PV Power (MW)', linestyle='-', linewidth=2, where='post')
        plt.step(time, wind_power_traj[:m,0]*1e-6, label='Wind Power (MW)', linestyle='-.', linewidth=2, where='post')
        plt.step(time, p_load[:m,0]*1e-6, label='Load Power (MW)', linestyle='--', linewidth=2, where='post')
        plt.step(time, P_battery_history[:m,0]*1e-6, label='Battery Power (MW)', linestyle='-.', linewidth=1.5, where='post')
        plt.step(time, P_grid_ex_history[:m,0]*1e-6, label='Export Grid Power (MW)', linestyle=':', linewidth=1.5, where='post')
        plt.step(time, P_grid_in_history[:m,0]*1e-6, label='Import Grid Power (MW)', linestyle='-', linewidth=1, alpha=0.7, where='post')
        plt.step(time, P_excess[:m,0]*1e-6, label='Curtailment Power (MW)', linestyle='-.', linewidth=1, alpha=0.7, where='post')
        plt.step(time, P_deficit[:m,0]*1e-6, label='Deficit Power (MW)', linestyle='-.', linewidth=1, alpha=0.7, where='post')

        plt.xlabel('Time (days)', fontsize=10)
        plt.ylabel('Power (MW)', fontsize=10)
        plt.xticks(fontsize=9)
        plt.yticks(fontsize=9)
        # plt.legend(fontsize=10, loc='upper left')
        # plt.title('Power of PV, Wind, Load, Battery, and Grid', fontsize=12)
        # plt.tight_layout()
        # plt.show()
        plt.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.0, 0.5))
        # plt.title('Power of PV, Wind, Load, Battery, and Grid', fontsize=12)
        plt.tight_layout(rect=[0, 0, 0.85, 1])  # Leave space on the right for the legend
        plt.show()        

        # Define constants and labels
        num_states = 4
        num_controls = 4
        '''
        # state_labels = ["dry mass", "CO2dens", "Temperature", "AH"]
        # unit_s = [r'$\mathrm{kg \, m^{-2}}$', r'$\mathrm{kg \, m^{-3}}$', r'$^\circ\mathrm{C}$', r'$\mathrm{kg \, m^{-3}}$']
        # measurement_labels = ["dry mass", "CO2", "Temperature", "RH"]
        # unit_m = [r'$\mathrm{kg \, m^{-2}}$', r'$\mathrm{ppm}$', r'$^\circ\mathrm{C}$', r'$\%$']
        # control_labels = ["CO2", "Q_cool_s", "phi_cond_h", "P_l"]
        # unit_c = [r'$\mathrm{kg \, m^{-2} \, s^{-1}}$', r'$\mathrm{W \, m^{-2}}$', r'$\mathrm{kg \, m^{-2} \, s^{-1}}$', r'$\mathrm{W \, m^{-2}}$']
        # disturbance_labels = ["Solar Radiation", "CO2", "Temperature", "RH"]
        # e_labels = ["Lighting", "Cooling", "Dehumidification", "Total"]
        # unit_e = [r'$\mathrm{W \, m^{-2}}$', r'$\mathrm{W \, m^{-2}}$', r'$\mathrm{W \, m^{-2}}$', r'$\mathrm{W \, m^{-2}}$']

        # def get_label(labels, index):
        #     """Returns the label if index is within range; otherwise returns a generic label."""
        #     return labels[index] if index < len(labels) else f"Trajectory {index + 1}"

        # # Plot measurement state Trajectories (one figure for states)
        # fig1, axs1 = plt.subplots(num_states, 1, figsize=(8, 8 * num_states))
        # fig1.suptitle('Measurement State Trajectories', fontsize=10)
        # for j in range(num_states):
        #     axs1[j].plot(time, y_check[:m+1, j], label=get_label(measurement_labels, j))
        #     axs1[j].set_xlabel('Time (days)', fontsize=10)
        #     axs1[j].set_ylabel(unit_m[j], fontsize=12, rotation=0, labelpad=30, ha='center')
        #     axs1[j].legend()
        #     axs1[j].grid(True)
        # fig1.tight_layout()
        # plt.show()

        # # Plot Denormalized Control Input Trajectories (one figure for controls)
        # fig2, axs2 = plt.subplots(num_controls, 1, figsize=(8, 8 * num_controls))
        # fig2.suptitle('Denormalized Control Input Trajectories', fontsize=10)
        # for k in range(num_controls):
        #     axs2[k].plot(time, u_values[:m+1, k], label=get_label(control_labels, k))
        #     axs2[k].set_xlabel('Time (days)', fontsize=10)
        #     axs2[k].set_ylabel(unit_c[k], fontsize=12, rotation=0, labelpad=30, ha='center')
        #     axs2[k].legend()
        #     axs2[k].grid(True)
        # fig2.tight_layout()
        # plt.show()

        # # # Plot specific disturbance trajectory
        # # plt.figure(figsize=(8, 8))
        # # plt.plot(time, d_values[:m+1, 2], label=get_label(disturbance_labels, 2))
        # # plt.xlabel('Time (days)', fontsize=10)
        # # plt.ylabel('Temperature')
        # # plt.title('Disturbance Trajectory', fontsize=10)
        # # plt.legend()
        # # plt.tight_layout()
        # # plt.show()

        # # # Plot State Trajectories (one figure for states)
        # # fig4, axs4 = plt.subplots(num_states, 1, figsize=(8, 8 * num_states))
        # # fig4.suptitle('State Trajectories', fontsize=10)
        # # for j in range(num_states):
        # #     axs4[j].plot(time, x_values[:m+1, j], label=get_label(state_labels, j))
        # #     axs4[j].set_xlabel('Time (days)', fontsize=10)
        # #     axs4[j].set_ylabel(unit_s[j], fontsize=12, rotation=0, labelpad=30, ha='center')
        # #     axs4[j].legend()
        # #     axs4[j].grid(True)
        # # fig4.tight_layout()
        # # plt.show()

        # # Plot normalized Control Input Trajectories (one figure for controls)
        # fig5, axs5 = plt.subplots(num_controls, 1, figsize=(8, 8 * num_controls))
        # fig5.suptitle('Normalized Control Input Trajectories', fontsize=10)
        # for k in range(num_controls):
        #     axs5[k].plot(time, nor_u[:m+1, k], label=get_label(control_labels, k))
        #     axs5[k].set_xlabel('Time (days)', fontsize=10)
        #     axs5[k].set_ylabel(unit_c[k], fontsize=12, rotation=0, labelpad=30, ha='center')
        #     axs5[k].legend()
        #     axs5[k].grid(True)
        # fig5.tight_layout()
        # plt.show()
        '''
        state_labels = [r"Dry Mass", r"CO$_2$ Concentration", r"Temperature", r"Absolute Humidity"]
        unit_s = [r'$\mathrm{kg \, m^{-2}}$', r'$\mathrm{kg \, m^{-3}}$', r'$^\circ\mathrm{C}$', r'$\mathrm{kg \, m^{-3}}$']
        measurement_labels = [r"Dry Mass", r"CO$_2$ Concentration", r"Temperature", r"Relative Humidity"]
        unit_m = [r'$\mathrm{kg \, m^{-2}}$', r'$\mathrm{ppm}$', r'$^\circ\mathrm{C}$', r'$\%$']
        # control_labels = [r"CO$_2$ Injection Rate", "Q_cool_s", "phi_cond_h", "P_l"]
        control_labels = [
            r"CO$_2$ Injection Rate",  
            r"Cooling Power $Q_{\mathrm{cool}}$",  
            r"Dehumidification Rate $\phi_{\mathrm{cond}}$",  
            r"Light intensity (PPFD) $P_{\mathrm{l}}$"
        ]
        unit_c = [r'$\mathrm{kg \, m^{-2} \, s^{-1}}$', r'$\mathrm{W \, m^{-2}}$', r'$\mathrm{kg \, m^{-2} \, s^{-1}}$', r'$\mu\mathrm{mol \, m^{-2} \, s^{-1}}$']
        # disturbance_labels = ["Solar Radiation", "CO2", "Temperature", "RH"]
        # e_labels = ["Lighting", "Cooling", "Dehumidification", "Total"]
        # unit_e = [r'$\mathrm{W \, m^{-2}}$', r'$\mathrm{W \, m^{-2}}$', r'$\mathrm{W \, m^{-2}}$', r'$\mathrm{W \, m^{-2}}$']


        

        # Set the figure title-----------------new figure----------
                # Create one large figure with 8 subplots arranged in 2 rows × 4 columns
        
        # fig, axs = plt.subplots(2, 4, figsize=(16, 8))  # 2 rows, 4 columns
        # fig.suptitle('State and Control Input Trajectories (Scatter Plots)', fontsize=11)

        # # Plot Measurement State Trajectories (First Row)
        # for j in range(num_states):
        #     axs[0, j].scatter(time, y_check[:m, j], s=2)  # s=10 controls marker size
        #     axs[0, j].set_xlabel('Time (days)', fontsize=10)
        #     axs[0, j].set_ylabel(f"{measurement_labels[j]}\n({unit_m[j]})", fontsize=10, rotation=90, labelpad=10)
        #     axs[0, j].grid(True)
        #     axs[0, j].set_title(f"{measurement_labels[j]}", fontsize=10)

        # # Plot Denormalized Control Input Trajectories (Second Row)
        # for k in range(num_controls):
        #     axs[1, k].scatter(time, u_values[:m, k], s=2)  # s=10 controls marker size
        #     axs[1, k].set_xlabel('Time (days)', fontsize=10)
        #     axs[1, k].set_ylabel(f"{control_labels[k]}\n({unit_c[k]})", fontsize=10, rotation=90, labelpad=10)
        #     axs[1, k].grid(True)
        #     axs[1, k].set_title(f"{control_labels[k]}", fontsize=10)

        # # Adjust layout for better spacing
        # fig.tight_layout(rect=[0, 0, 1, 0.95])  # Leaves space for the title
        # plt.show()  
                
        # code for old version of the code below:
                
        # ! Below is the plot showing trajectories of states and controls.
        fig, axs = plt.subplots(2, 4, figsize=(16, 8))  # 2 rows, 4 columns
        
        fig.suptitle('State and Control Input Trajectories', fontsize=11)

        # Plot Measurement State Trajectories (First Row)
        for j in range(num_states):
            axs[0, j].plot(time, y_check[:m, j])
            axs[0, j].set_xlabel('Time (days)', fontsize=10)
            axs[0, j].set_ylabel(f"{measurement_labels[j]}\n({unit_m[j]})", fontsize=10, rotation=90, labelpad=10)
            axs[0, j].grid(True)
            axs[0, j].set_title(f"{measurement_labels[j]}", fontsize=10)

        # Plot Denormalized Control Input Trajectories (Second Row)
        for k in range(num_controls):
            axs[1, k].plot(time, u_values[:m, k])
            axs[1, k].set_xlabel('Time (days)', fontsize=10)
            axs[1, k].set_ylabel(f"{control_labels[k]}\n({unit_c[k]})", fontsize=10, rotation=90, labelpad=10)
            axs[1, k].grid(True)
            axs[1, k].set_title(f"{control_labels[k]}", fontsize=10)

        # Adjust layout for better spacing
        fig.tight_layout(rect=[0, 0, 1, 0.95])  # Leaves space for the title
        plt.show()
        
        '''
        # ---------------------------new figure---------------------------------------
        
        # Plot specific disturbance trajectory  
        # plt.figure(figsize=(8, 8))
        # plt.plot(time, d_values[:m+1, 2])

        # # Plot measurement state Trajectories (one figure for states)
        # fig1, axs1 = plt.subplots(num_states, 1, figsize=(8, 8 * num_states))
        # fig1.suptitle('Measurement State Trajectories', fontsize=12)
        # for j in range(num_states):
        #     axs1[j].plot(time, y_check[:m+1, j])
        #     axs1[j].set_xlabel('Time (days)', fontsize=10)
        #     axs1[j].set_ylabel(f"{measurement_labels[j]} ({unit_m[j]})", fontsize=12, rotation=90, labelpad=30, ha='center')
        #     # axs1[j].legend()
        #     axs1[j].grid(True)
        # fig1.tight_layout()
        # plt.show()

        # # Plot Denormalized Control Input Trajectories (one figure for controls)
        # fig2, axs2 = plt.subplots(num_controls, 1, figsize=(8, 8 * num_controls))
        # fig2.suptitle('Denormalized Control Input Trajectories', fontsize=12)
        # for k in range(num_controls):
        #     axs2[k].plot(time, u_values[:m+1, k])
        #     axs2[k].set_xlabel('Time (days)', fontsize=10)
        #     axs2[k].set_ylabel(f"{control_labels[k]} ({unit_c[k]})", fontsize=12, rotation=90, labelpad=30, ha='center')
        #     # axs2[k].legend(fontsize=10, loc='upper left')
        #     axs2[k].grid(True)
        # fig2.tight_layout()
        # plt.show()

        # # # Plot specific disturbance trajectory
        # # plt.figure(figsize=(8, 8))
        # # plt.plot(time, d_values[:m+1, 2])
        # # plt.xlabel('Time (days)', fontsize=10)
        # # plt.ylabel('Temperature')
        # # plt.title('Disturbance Trajectory', fontsize=12)
        # # # plt.legend(fontsize=10, loc='upper left')
        # # plt.tight_layout()
        # # plt.show()

        # # # Plot State Trajectories (one figure for states)
        # # fig4, axs4 = plt.subplots(num_states, 1, figsize=(8, 8 * num_states))
        # # fig4.suptitle('State Trajectories', fontsize=10)
        # # for j in range(num_states):
        # #     axs4[j].plot(time, x_values[:m+1, j])
        # #     axs4[j].set_xlabel('Time (days)', fontsize=10)
        # #     axs4[j].set_ylabel(f"{state_labels[j]} ({unit_s[j]})", fontsize=12, rotation=0, labelpad=30, ha='center')
        # #     # axs4[j].legend(fontsize=10, loc='upper left')
        # #     axs4[j].grid(True)
        # # fig4.tight_layout()
        # # plt.show()

        # # Plot normalized Control Input Trajectories (one figure for controls)
        # fig5, axs5 = plt.subplots(num_controls, 1, figsize=(8, 8 * num_controls))
        # fig5.suptitle('Normalized Control Input Trajectories', fontsize=10)
        # for k in range(num_controls):
        #     axs5[k].plot(time, nor_u[:m+1, k])
        #     axs5[k].set_xlabel('Time (days)', fontsize=10)
        #     axs5[k].set_ylabel(f"{control_labels[k]} ({unit_c[k]})", fontsize=12, rotation=90, labelpad=30, ha='center')
        #     # axs5[k].legend(fontsize=10, loc='upper left')
        #     axs5[k].grid(True)
        # fig5.tight_layout()
        # plt.show()
        '''
    except Exception as e:
        print(f"An error occurred during plotting: {e}")


# def calculate_coe(design_para, eval_param, electricity_cost, E_load):
#     # Extract parameters
#     r = eval_param["discount_rate"]
#     project_time = eval_param["project_time"]
#     n_pv = 30  # PV system lifetime (years)
#     n_batt = 5  # Battery system lifetime (years)
#     n_wind = 30  # Wind turbine lifetime (years)
    
#     P_light_fixed = calculate_ppfd(design_para["num_light"], DefineParameters())
#     photo_time = DLI / P_light_fixed * 1e6 /3600
#     light_life_hours = 36000 # hours
#     light_life_years_rounded = math.floor(light_life_hours/photo_time/365)
#     replacement_years = list(range(light_life_years_rounded, project_time + 1, light_life_years_rounded))  

   
#     # Calculate Capital Recovery Factors (CRF)
#     def crf(rate, years):
#         return (rate * (1 + rate)**years) / ((1 + rate)**years - 1)
    
#     crf_project = crf(r, project_time)
#     crf_pv = crf(r, n_pv)
#     crf_batt = crf(r, n_batt)
#     crf_wind = crf(r, n_wind)
#     crf_light = crf(r, light_life_years_rounded)


#     # Initial investment costs
#     C_pv_init = design_para["num_pv"] * eval_param["pv_unit_cost"]
#     C_batt_init = design_para["num_batt"] * eval_param["batt_unit_cost"]
#     C_wind_init = design_para["num_wind"] * eval_param["wind_unit_cost"]
#     C_light_init = design_para["num_light"] * eval_param["led_unit_cost"]
#     CAPEX = C_pv_init + C_batt_init + C_wind_init + C_light_init

#     # Annualized capital costs
#     annualized_pv = C_pv_init * crf_pv
#     annualized_batt = C_batt_init * crf_batt
#     annualized_wind = C_wind_init * crf_wind
#     annualized_light = C_light_init * crf_light
#     annualized_CAPEX = annualized_pv + annualized_batt + annualized_wind + annualized_light
 
 
#     # Annualized replacement cost, lights
#     R_light = sum(C_light_init / (1 + r)**year for year in replacement_years)
#     annualized_replacement_light = R_light * crf_project
#     # Battery replacement costs calculation
#     replacement_years = list(range(5, project_time + 1, 5))  # Replacements every 5 years
#     present_value_replacements = sum(
#         C_batt_init / (1 + r)**year
#         for year in replacement_years
#     )
#     annualized_replacement_batt = present_value_replacements * crf_pv

#     annualized_replacement = annualized_replacement_batt + annualized_replacement_light
    
#     # Annual O&M costs (2% of initial investment)
#     annual_om = 0.02 * (C_pv_init + C_batt_init + C_wind_init)
    
    
    
#     # Total annualized system cost
#     # ASC is the initial CAPEX converted to yearly terms considering the discount rate,
#     # and includes the annualized replacement cost.
#     ASC = (
#         annualized_pv
#         + annualized_batt
#         + annualized_wind
#         + annualized_light
#         # + annualized_replacement
#         # + annual_om
#     )

#     annual_energy_used = sum(E_load) *17
#     # Cost of Electricity (COE)
#     coe = ASC #/ annual_energy_used
    
#     return ASC
