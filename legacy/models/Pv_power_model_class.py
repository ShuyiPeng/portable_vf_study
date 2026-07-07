from datetime import datetime
import numpy as np
from pvlib import solarposition
import pandas as pd

class PvPowerCalculator:
    """
    A class to calculate solar position, irradiance decomposition, and PV power output.
    """
    def __init__(self, latitude, longitude, ground_albedo, timestamp, ghi, surface_tilt, surface_azimuth, T_ambient, altitude=0, solar_constant=1366.1, module_parameters=None, system_parameters=None):
        self.latitude = latitude
        self.longitude = longitude
        self.ground_albedo = ground_albedo #0.2, 20% ground albedo
        self.timestamp = timestamp
        self.ghi = ghi
        self.surface_tilt = surface_tilt  #30
        self.surface_azimuth = surface_azimuth #180    
        # surface_tilt, surface_azimuth
        # *1 if the tilt and azimuth angles are fixed, simply know the values of them, such as 30 degrees and 180 degrees for south-facing, respectively.
        # *2 if using solar tracking systems, the tilt and azimuth angles will change over time based on sun's position.
        self.T_ambient = T_ambient
        self.altitude = altitude
        self.solar_constant = solar_constant
        
        self.module_parameters = module_parameters or {
            'p_max': 300,           # Rated capacity of the pv array in Watts (W)  (ref: https://www.sciencedirect.com/science/article/pii/S1364032116306712)
            'v_oc': 40.5,           # Open-circuit voltage in Volts (V)
            'i_sc': 8.5,            # Short-circuit current in Amperes (A)
            'k_v': -0.002,          # Voltage temperature coefficient in %/°C
            'k_i': 0.001,           # Current temperature coefficient in %/°C
            'noct': 45,             # Nominal Operating Cell Temperature in °C
            'alpha': 0.8,           # Module absorption coefficient (dimensionless)
            'tau': 0.9,             # Absorption-transmission product (dimensionless)
            'eta_mp_stc': 0.18      # Efficiency at STC (dimensionless, as a fraction)
        }
        self.system_parameters = system_parameters or {
            'poa_stc': 1000,       # Plane-of-array irradiance at standard test conditions (W/m²)
            'poa_noct': 800,       # Plane-of-array irradiance at Nominal Operating Cell Temperature (W/m²)
            'temp_a_noct': 20,     # Ambient temperature at NOCT conditions (°C)
            'alpha_p': -0.004,     # Temperature coefficient of power (%/°C)
            'f_pv': 0.95,          # PV soiling/mismatch/etc. factor (dimensionless)
            'eta_inv': 0.98        # Inverter efficiency (as a fraction, dimensionless)
        }

    def calculate_solar_position(self):
        """
        Calculate solar position (zenith and azimuth angles)
        """
        times = pd.DatetimeIndex([self.timestamp])
        solar_position = solarposition.get_solarposition(
            times, self.latitude, self.longitude, self.altitude
        )
        return solar_position['zenith'].iloc[0], solar_position['azimuth'].iloc[0]

    def decompose_ghi(self, model='erbs'):
        """
        Decompose GHI into DNI and DHI using various models
        """
    # Convert zenith angle to radians
        zen_rad = np.radians(self.calculate_solar_position()[0])
        """
        # Calculate extraterrestrial radiation
        
        # # Option 1: Simplified version using solar constant (with no reference of this simplification in the literature)
        # solar_constant = 1361  # W/m²
        # extra = solar_constant * np.cos(zen_rad)
        
        # Option 2, use the API from pvlib, the method is spencer, the equation can be found also in this 
        # website: https://pvpmc.sandia.gov/modeling-guide/1-weather-design-inputs/irradiance-insolation/extraterrestrial-radiation/
        """
        
        B = solarposition._calculate_simple_day_angle(self.timestamp.timetuple().tm_yday)  # This replaces to_doy
        RoverR0sqrd = (1.00011 + 0.034221 * np.cos(B) + 0.00128 * np.sin(B) +
                    0.000719 * np.cos(2 * B) + 7.7e-05 * np.sin(2 * B))
        extra = self.solar_constant * RoverR0sqrd
        
        # Calculate clearness index
        kt = np.where(extra > 0, self.ghi / extra, 0)
        kt = np.clip(kt, 0, 1)
        
        # Different models for diffuse fraction (DHI/GHI)
        if model.lower() == 'erbs':
            # Erbs model
            diff_frac = np.where(
                kt <= 0.22,
                1.0 - 0.09 * kt,
                np.where(
                    kt <= 0.8,
                    0.9511 - 0.1604 * kt + 4.388 * kt**2 - 
                    16.638 * kt**3 + 12.336 * kt**4,
                    0.165
                )
            )
        
        elif model.lower() == 'reindl':
            # Reindl model
            diff_frac = np.where(
                kt <= 0.3,
                1.020 - 0.254 * kt + 0.0123 * np.cos(zen_rad),
                np.where(
                    kt <= 0.78,
                    1.400 - 1.749 * kt + 0.177 * np.cos(zen_rad),
                    0.486 * kt - 0.182 * np.cos(zen_rad)
                )
            )
        
        elif model.lower() == 'orgill':
            # Orgill and Hollands model
            diff_frac = np.where(
                kt <= 0.35,
                1.0 - 0.249 * kt,
                np.where(
                    kt <= 0.75,
                    1.557 - 1.84 * kt,
                    0.177
                )
            )
        
        else:
            raise ValueError("Invalid model. Choose 'erbs', 'reindl', or 'orgill'")
        
        # Calculate DHI and DNI
        dhi = self.ghi * diff_frac
        dni = np.where(
            np.cos(zen_rad) > 0.01,
            (self.ghi - dhi) / np.cos(zen_rad),
            0
        )
        
        # Apply physical constraints
        dhi = np.clip(dhi, 0, self.ghi)
        dni = np.clip(dni, 0, None)
        
        return dni, dhi

    def calculate_poa_irradiance(self):
        """
        Calculate the plane-of-array (POA) irradiance on a tilted surface.
        """
        # Convert angles to radians
        theta_z_rad = np.radians(self.calculate_solar_position()[0])
        beta_rad = np.radians(self.surface_tilt)
        gamma_rad = np.radians(self.calculate_solar_position()[1] - self.surface_azimuth)
        
        # Calculate angle of incidence (AOI)
        aoi_rad = np.arccos(np.cos(theta_z_rad) * np.cos(beta_rad) +
                        np.sin(theta_z_rad) * np.sin(beta_rad) * np.cos(gamma_rad))
        
        dni, dhi = self.decompose_ghi()
        # Calculate POA beam irradiance
        # poa_beam = dni * np.cos(aoi_rad)
        poa_beam = dni * np.clip(np.cos(aoi_rad), 0, 1)
        
        # Calculate POA diffuse irradiance
        poa_diffuse = dhi * (1 + np.cos(beta_rad)) / 2
        
        # Calculate POA reflected irradiance
        poa_reflected = self.ghi * self.ground_albedo * (1 - np.cos(beta_rad)) / 2
        
        # Calculate total POA irradiance
        # poa_total = poa_beam + poa_diffuse + poa_reflected
        poa_total = np.maximum(0, poa_beam + poa_diffuse + poa_reflected)
        
        return poa_beam, poa_diffuse, poa_reflected, poa_total

    # surface_tilt, surface_azimuth
    # *1 if the tilt and azimuth angles are fixed, simply know the values of them, such as 30 degrees and 180 degrees for south-facing, respectively.
    # *2 if using solar tracking systems, the tilt and azimuth angles will change over time based on sun's position.
    
    def calculate_pv_power(self):
        """
        Calculate the power output of a PV system given the total POA irradiance, 
        ambient temperature, module parameters, and system parameters.

        Returns:
        float: PV system power output (W)
        """
        # Unpack module parameters
        p_max, v_oc, i_sc, k_v, k_i, noct, alpha, tau, eta_mp_stc = (
            self.module_parameters['p_max'],
            self.module_parameters['v_oc'],
            self.module_parameters['i_sc'],
            self.module_parameters['k_v'],
            self.module_parameters['k_i'],
            self.module_parameters['noct'],
            self.module_parameters['alpha'],
            self.module_parameters['tau'],
            self.module_parameters['eta_mp_stc']
        )
        
        # Unpack system parameters
        poa_stc, poa_noct, temp_a_noct, alpha_p, f_pv, eta_inv = (
            self.system_parameters['poa_stc'],
            self.system_parameters['poa_noct'],
            self.system_parameters['temp_a_noct'],
            self.system_parameters['alpha_p'],
            self.system_parameters['f_pv'],
            self.system_parameters['eta_inv']
        )
        
        _, _, _, poa_total = self.calculate_poa_irradiance()
        # Calculate cell temperature
        cell_temp = (self.T_ambient + (noct - temp_a_noct) * (poa_total / poa_noct) *
                    (1 - eta_mp_stc * (1 - alpha_p * noct) / (tau * alpha))) / \
                    (1 + (noct - temp_a_noct) * (poa_total / poa_noct) *
                    (alpha_p * eta_mp_stc / (tau * alpha)))
        
        # # Calculate temperature-corrected voltage and current
        # v_oc_t = v_oc * (1 + k_v * (cell_temp - 25) / 100)
        # i_sc_t = i_sc * (1 + k_i * (cell_temp - 25) / 100)
        
        # Calculate maximum power using the detailed model
        # pv_power = p_max * (poa_total / poa_stc) * \
        #         (1 + alpha_p * (cell_temp - 25)) * f_pv * eta_inv
        
        # temp_corr = np.clip(1 + alpha_p * (cell_temp - 25), 0, None)
        temp_corr = 1 + alpha_p * (cell_temp - 25)
        pv_power = p_max * (poa_total / poa_stc) * temp_corr * f_pv * eta_inv
        # pv_power = np.clip(pv_power, 0, p_max*1.2)
        # pv_power = p_max * (poa_total / poa_stc) * (v_oc_t * i_sc_t / (v_oc * i_sc)) * \
        #         (1 + alpha_p * (cell_temp - 25)) * f_pv * eta_inv        
        
        # pv_power = p_max * (poa_total / poa_stc) * (1 + alpha_p * (cell_temp - 25)) * \
        #     f_pv * eta_inv
        pv_power = min(pv_power, p_max)


        return pv_power

    def calculate_total_power_output(self):
        """
        Calculate the total power output of the PV system directly.

        Returns:
        float: Total PV power output (W)
        """
        return self.calculate_pv_power()



# Example usage
if __name__ == "__main__": 
    latitude = 40.7128   # latitude of the location
    longitude = -74.0060  # longitude of the location
    altitude = 0  # altitude of the location
    ground_albedo = 0.2  # 20% ground albedo
    timestamp = datetime.now()  # current time
    ghi = 1000  # W/m²
    surface_tilt = 22
    surface_azimuth = 180
    T_ambient = 25    
    solar_constant = 1366.1
    
    module_parameters = {
        'p_max': 340,           # Max power output in Watts (W)
        'v_oc': 40.5,           # Open-circuit voltage in Volts (V)
        'i_sc': 8.5,            # Short-circuit current in Amperes (A)
        'k_v': -0.002,          # Voltage temperature coefficient in %/°C
        'k_i': 0.001,           # Current temperature coefficient in %/°C
        'noct': 45,             # Nominal Operating Cell Temperature in °C
        'alpha': 0.8,           # Module absorption coefficient (dimensionless)
        'tau': 0.9,             # Absorption-transmission product (dimensionless)
        'eta_mp_stc': 0.18      # Efficiency at STC (dimensionless, as a fraction)
    }
    system_parameters = {
        'poa_stc': 1000,       # Plane-of-array irradiance at standard test conditions (W/m²)
        'poa_noct': 800,       # Plane-of-array irradiance at Nominal Operating Cell Temperature (W/m²)
        'temp_a_noct': 20,     # Ambient temperature at NOCT conditions (°C)
        'alpha_p': -0.004,     # Power temperature coefficient (%/°C)
        'f_pv': 0.95,          # PV soiling/mismatch/etc. factor (dimensionless)
        'eta_inv': 0.98        # Inverter efficiency (as a fraction, dimensionless)
    }
    
    calculator = PvPowerCalculator(latitude, 
                                   longitude, 
                                   ground_albedo, 
                                   timestamp, 
                                   ghi, 
                                   surface_tilt, 
                                   surface_azimuth, 
                                   T_ambient, 
                                   altitude, 
                                   solar_constant, 
                                   module_parameters, 
                                   system_parameters)
    
    total_power_output = calculator.calculate_total_power_output()
    print("Total PV Power Output:", total_power_output)
