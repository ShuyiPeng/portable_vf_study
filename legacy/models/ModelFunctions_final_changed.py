import numpy as np
import sys
import math

'''
This file contains the model functions for the design study. 
'''

def F(x, u, d, p):                                              
    Q_led_phot = (math.floor(u[3]/p["eta_led"])) * p["eta_PAR"]
    Q_led_q = (math.floor(u[3]/p["eta_led"]))                           # unit: W/m^2 
    Q_env = (p["A_surface"]/p["A_floor"]) * p["U_value"] * (d[1] - x[2])     # unit: W/m^2
    # update some internal states if needed
    Gamma = 5.2e-5
    epsilon = 1.7e-8
    g_c = -p["photCO2_1"] * x[2]**2 + p["photCO2_2"] * x[2] - p["photCO2_3"]
    # g_co2 = 1/((1/p["c_bnd"]) + (1/p["c_stm"]) + (1/g_c))
    # print("g_c and g_co2:", g_c, g_co2)
    phi_led_phot = epsilon* p["par"] * p["rad_rf"]
    phi_phot_max = phi_led_phot * Q_led_phot * g_c * (x[1] - Gamma)/(phi_led_phot * Q_led_phot + g_c * (x[1] - Gamma))
    p["c_pl_d"] = p["c_k"] * p["lar_d"] * (1 - p["c_t"])
    phi_phot_c = (1-np.exp(-p["c_pl_d"] * x[0])) * phi_phot_max # unit: kg m{-2} s{-1}
    phi_resp = (p["resp,s"]*(1-p["c_t"])+p["resp,r"]*p["c_t"]) * x[0] * 2**(0.1 * x[2] - 2.5)
    phi_transp_h = (1-np.exp(-p["c_pl_d"] * x[0])) * p["c_v_pl_ai"] * (((p["v,0"]*p["v,1"]*p["H2O"])/(p["c_R"] * (x[2] + p["c_T_abs"]))) * np.exp((p["satH2O2"]*x[2])/(p["satH2O3"]+x[2]))-x[3])
    phi_transp_h_f = max(0, phi_transp_h)
    Q_transp = p["lat_water"]*phi_transp_h # [W/m^2]

    ki =  np.array([
        p["alfaBeta"]* phi_phot_c - phi_resp,

        1/p["CO2cap"] * ((p["A_factor"] * (-phi_phot_c)) + (p["A_factor"] * (phi_resp /p["alpha"])) + u[0]),

        1/p["aCap"] * (p["A_factor"]*Q_led_q + Q_env - u[1] - (p["A_factor"]*Q_transp)),

        1/p["H2Ocap"] * (p["A_factor"]*phi_transp_h_f - u[2])
        
        ])
    
    
    
    return ki    

def DefineParameters():
    p={}
    p["satH2O2"] = 17.4                 #* (cv2)saturation water vapour parameter 			[-] 					17.4
    p["satH2O3"] = 239                  # (cv3)saturation water vapour parameter 			[°C] 					239
    p["c_R"] = 8314                     # *ideal gas constant 							[J K^{-1} kmol^{-1}] 	8314
    p["c_T_abs"] = 273.15               # *conversion from C to K 						[K] 					273.15
    p["aCap"] = 3e4 					# *effective heat capacity of the greenhouse air [J m^{-2}{gh} °C^{-1}]  3e4 #1005*1.2*10
    p["vCap"] = 1005*1.2#1290 				# *heat capacity per volume of greenhouse air 	[J m^{-3}{gh} °C^{-1}]  1290
    p["c_k"] = 0.9    #Xu
    p["photGamma"] = 7.32e-5 #(phd thesis)	# *(c_T)carbon dioxide compensation point 			[kg{CO2} m^{-3}{air}] 	5.2e-5
    p["c_v_pl_ai"] = 3.6e-3 			# (evap,c,a)canopy transpiration mass transfter coefficent/coefficient of leaf-air vapor flow 			[m s^{-1}] 				3.6e-3
    p["dw_fw"] = 20.98 # calibrated by Xd # used 22.5 all the time               # dry weigth to fresh weight ratio              [unit]                  22.5
    p["lettucePrice"] = 1.08  # euro /kg. exact price needs to be found! 
    p["eta_PAR"] = 0.6 #0.7                    # (GTa-tool=0.6)*efficiency of LED light, check the value [%]            if eta_par=0.6, it means that 60% of the input electrical energy is converted into PAR photons. # nature paper p["eta_PAR"] = 0.52
    p["U_value"] = 0.2                 # *u value/ thermal transmittance (Weidner et al., 2021)                      [-]                     0.3
    p["rack_l"] = 24#22.5#20
    p["rack_w"] = 1.8#1.2
    p["layer"] = 8 #9
    p["num_rack"] = 9 #12
    p["num_room"] = 4 
    p["A_length"] = 50
    p["A_width"] = 50
    p["A_floor"] = p["A_length"] * p["A_width"] #10**2    100**2          # *floor area of the greenhouse, check the value                      [m^2]                            100**2
    p["A_height"] = 6 #5 10                 # the height of the vertical farm                                    [m]                              own assumption
    p["A_surface"] = p["A_floor"] + 2*p["A_length"]*p["A_height"] + 2*p["A_width"]*p["A_height"]  #10**2 + 4*10*5 100**2 + 4*100*10 # *surface area of the greenhouse, check the value                   [m^2]                            100**2 + 4*100*20
    p["A_cul"] = p["rack_l"]*p["rack_w"]*p["layer"]*p["num_rack"]*p["num_room"]                 # cultivation area
    p["A_factor"] = math.floor((p["A_cul"]/p["A_floor"]) * 10)/10     
    p["par"] = 1                       # the ratio of photosynthetically active radiation to total led radiation (from Xu et al. 2020 optimal control of LED) [-]
    p["rad_rf"] = 1                    # the transmission coefficient of the roof for led radiation (from Xu et al. 2020 optimal control of LED) [-]
    p["CO2cap"] = p["A_height"]
    p["H2Ocap"] = p["A_height"] 
    p["lat_water"] = 2.45e6 #2256.4#too low            # latent heat of vaporization of water (ref:)                            [kJ kg^{-1}]           2500 to high #related to temperature

    # Parameters for electricity calculation
    p["COP_cool"] = 4.0
    p["COP_heat"] = 4.0 
    p['dehum_eff'] = 0.292 #L/kWh
    p["c_epsilon"] = 17e-9 #5.82e-9 #17e-9 # 5.82e-9 #17e-9  # this is used in the model
    p["resp,s"] = 3.47e-7
    p["resp,r"] = 1.16e-7
    p["alpha"] = 0.68
    p["v,0"] = 0.85
    p["v,1"] = 611
    p["H2O"] = 18
    p["c_bnd"] = 0.004
    p["c_stm"] = 0.007
    
    # updated parameters from xu's paper
    p["lar_d"]     = 24.63 
    p["alfaBeta"] = 0.51
    p["c_t"] = 0.084                    # 0.084 ratio of root dry mass to lettuce dry mass, Xu, calibrated
    p["photCO2_1"]= 5.11e-6             # *temperature influence on photosynthesis 		[m s^{-1} °C^{-2}]
    p["photCO2_2"]=2.3e-4				# *temperature influence on photosynthesis 		[m s^{-1} °C^{-1}] 		2.3e-4
    p["photCO2_3"]=6.29e-4              # *temperature influence on photosynthesis 		[m s^{-1}] 				6.29e-4
    # p["resp_c"]    = 4.87e-7            # Xu. *(CO2c_a)respiration coefficient 						[s^{-1}]  				4.87e-7
    p["eta_led"] = 3.5 
    p["eta_par"] = 0.553
    
    return p

def PVmoduleParameters():
    module_parameters = {}
    module_parameters["p_max"] = 340           # Max power output in Watts (W)   # each unit is 340Wp
    module_parameters["v_oc"] = 48              # Open-circuit voltage in Volts (V)
    module_parameters["i_sc"] = 8.5             # Short-circuit current in Amperes (A)
    module_parameters["k_v"] = -0.002            # Voltage temperature coefficient in %/°C 
    module_parameters["k_i"] = 0.001             # Current temperature coefficient in %/°C
    # 上面这两个参数好像没有在产品参数里找到coefficient in power, -0.4 in %/°C
    module_parameters["noct"] = 25              # Nominal Operating Cell Temperature in °C
    module_parameters["alpha"] = 0.8            # Module absorption coefficient (dimensionless)
    module_parameters["tau"] = 0.9              # Absorption-transmission product (dimensionless)
    module_parameters["eta_mp_stc"] = 0.18       # Efficiency at STC (dimensionless)
    

    return module_parameters

def PVsysParameters():
    system_parameters = {}
    system_parameters["poa_stc"] = 1000          # Plane-of-array irradiance at standard test conditions (W/m²)
    system_parameters["poa_noct"] = 800          # Plane-of-array irradiance at NOCT (W/m²)
    system_parameters["temp_a_noct"] = 25        # Ambient temperature at NOCT conditions (°C)
    system_parameters["alpha_p"] = -0.004          # Power temperature coefficient in %/°C
    system_parameters["f_pv"] = 0.95             # PV soiling/mismatch factor (dimensionless)
    system_parameters["eta_inv"] = 0.98           # Inverter efficiency (dimensionless)
    
    return system_parameters

def BattParameters():
    batt_param = {}
    batt_param["battery_capacity"] = 4800       # Battery capacity in Wh
    batt_param["SoC_initial"] = 0.5             # Initial State of Charge (50%)
    batt_param["SoC_min"] = 0.05                 # Minimum SoC (5%)
    batt_param["SoC_max"] = 0.95                # Maximum SoC (95%)
    batt_param["P_grid_max_export"] = 0.35e6 #20% of the peak load # 0.35e6#1.65e6 # 0.8*1.734e6 # 0.8* 2.52e6 #800 * 6 * 1e4  # Maximum export power to the grid (in Watts)
    batt_param["P_grid_max_import"] = 2.52e6 #1.734e6 # 2.52e6 #800 * 6 * 1e4  # Maximum import power from the grid (in Watts)
    batt_param["P_battery_max"] = 4800         # Maximum charging power to the battery (in Watts)
    batt_param["eta_c"] = 0.9                   # Charging efficiency
    batt_param["eta_d"] = 0.9                    # Discharging efficiency
    
    return batt_param

def WindParameters():
    wind_param = {}
    wind_param["P_wind_rated"] = 3e5 # https://en.wind-turbine-models.com/turbines/367-enercon-e-32-300#companies        
    return wind_param

def OtherParameters():
    other_param = {}
    other_param["surface_tilt"] = 32            # Surface tilt angle           
    other_param["latitude"] = 52.0037         # Latitude of the location (degrees)
    other_param["longitude"] = 4.32                  # Longitude of the location (degrees) (Bleiswijk)
    other_param["light_nominal_power"] = 54.5   # W, average power of the led nominal power range, 51-58, W
    other_param["sun_elevation"] = 14   # sun elevation angle (degrees) at winter solstice in the Netherlands
    
    return other_param

def EvalParameters():
    eval_param = {}
    # category 1: technical: renewble energy fraction
    # category 2: economic: total investment, electricity cost and revenue, yield revenue  
    # category 3: environmental: land area, co2 emission (grid), co2 emission of facilities (pv and batt), co2 injection (negative)
    eval_param["co2_emission_factor"] = 310 * 1e-3  # kg CO2_e/kWh, 2022, Netherlands
    # https://data.jrc.ec.europa.eu/dataset/919df040-0252-4e4e-ad82-c054896e1641
    # eval_param["co2_price"] = 0.025  # €/tonne CO2, 2022, Netherlands
    eval_param["co2_pv_facilities"] = 32   #pv, 32 g/kWh, battery: 1.75e5 g/kWh
    eval_param["co2_batt_facilities"] = 1.75e5 # g/kWh
    eval_param["batt_unit_cost"] = 1300 #euro
    eval_param["pv_unit_cost"] = 140 #euro
    eval_param["project_time"] = 30
    eval_param["discount_rate"] = 0.05 # 5%
    # eval_param["price_buy_electricity"] = 0.15 # €/kWh, 2022, Netherlands
    # eval_param["price_sell_electricity"] = 0.1 # €/kWh, varying a lot in dynamic market
    eval_param["pv_length"] = 1.8 #m
    eval_param["pv_width"] = 0.9 #m
    eval_param["wind_unit_cost"] = 20000 #m
    
    eval_param["led_unit_cost"] = 30 #estimated value # euro
    eval_param["lettuce_price"] = 1.08 # euro/kg, max NL (notebook)
    
    return eval_param