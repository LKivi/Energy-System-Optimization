# -*- coding: utf-8 -*-
"""

Author: Marco Wirtz, Institute for Energy Efficient Buildings and Indoor Climate, RWTH Aachen University, Germany

Created: 01.09.2018

"""

import numpy as np
import math
#import sun
import os

import grid
import soil



def load_params():
    """
    Returns all known data for optmization model.
    """
  
    #%% GENERAL ECONOMIC PARAMETERS
    param = {"interest_rate":  0.05,        # ---,          interest rate
             "observation_time": 20.0,      # a,            project lifetime
             "price_gas": 0.0435,           # kEUR/MWh,     natural gas price
             "price_el": 0.106,             # kEUR/MWh,     electricity price (grid)
             "revenue_feed_in": 0.055,      # kEUR/MWh,     feed-in tariff (electricity)
             "gas_CO2_emission": 0.2,       # t_CO2/MWh,    specific CO2 emissions (natural gas)
             "grid_CO2_emission": 0.657,    # t_CO2/MWh,    specific CO2 emissions (grid)
#             "pv_stc_area": 10000,          # m2,          roof area for pv or stc
             "MIPGap":      0.0001          # ---,          MIP gap
             }


    #%% SOIL PARAMETERS   
    param_soil = {"alpha_soil": 0.8,            #---,       soil surface absorptance
                  "epsilon_soil": 0.9,          #---,       soil surface emissivity
                  "evaprate_soil": 0.6,         #---,       soil surface evaporation rate
                  "lambda_soil": 2,             # W/(m*K),  soil heat conductivity
                  "heatcap_soil": 2e6}          # J/(m^3*K),soil volumetric heat capacity 
    
    param.update(param_soil)
    
    
    #%% ASPHALT LAYER PARAMETERS
    param_asphalt = {"asphaltlayer": 0,          #---,       consideration of asphalt layer? 1 = yes, 0 = no
                     "d_asph": 0.2,              # m,        asphalt layer thickness
                     "alpha_asph": 0.93,         #---,       asphalt surface absorptance
                     "epsilon_asph": 0.88,       #---,       asphalt surface emissivity
                     "evaprate_asph": 0.3,       #---,       asphalt surface evaporation rate
                     "lambda_asph": 0.7,         # W/(m*K),  asphalt heat conductivity
                     "heatcap_asph": 1950400}    # J/(m^3*K),asphalt volumetric heat capacity
    
    param.update(param_asphalt)  
    
    
    #%% PIPE PARAMETERS
    param_pipe = {"grid_depth": 1,                  # m,       installation depth beneath surface
                  "lambda_ins": 0.04,                # W/(m*K), insulation heat conductivity
                  "f_fric": 0.025,                  # ---,     pipe friction factor
                  "dp_pipe": 150,                   # Pa/m,    maximum pipe pressure gradient
                  "c_p":4180,                       # J/(kg*K),fluid specific heat capacity
                  "rho":1000}                       # kg/m^3,  fluid density
    
    param.update(param_pipe)
    
    
    #%% TEMPERATURES
    param_temperatures = {"T_heating_supply": 105,     # °C,   heating supply temperature
                          "T_heating_return": 70,      # °C,   heating return temperature
                          "T_cooling_supply": 6,       # °C,   cooling supply temperature
                          "T_cooling_return": 12}      # °C,   cooling return temperature
    
    param.update(param_temperatures)
    
    
    
    #%% GRID DATA
    # design grid properties for the given input data and parameters
    grid_data = grid.design_grid(param)
    grid.plotGrid
    
 
     #%% LOADS

    dem = {} 
    

    dem_buildings = grid.load_demands(grid_data)
        
    dem["heat"] = dem_buildings["heating"]["sum"]      # MW, heating demand 
    dem["cool"] = dem_buildings["cooling"]["sum"]      # MW, cooling demand   

 
    
#%% THERMAL LOSSES
   
    # calculate heating and cooling losses of the grid
    Losses = soil.calculateLosses(param, grid_data)
    
    anteil = np.sum(Losses["heating_grid"])/(np.sum(dem["heat"])+np.sum(Losses["heating_grid"]))
    print("Anteil Wärmeverluste = " + str(anteil))
    
    # Add losses to demands
    dem["heat"] = dem["heat"] + Losses["heating_grid"]
    dem["cool"] = dem["cool"] + Losses["cooling_grid"]
   
#%%   
    # Improve numeric by deleting very small loads
    eps = 0.01 # MW
    for load in ["heat", "cool"]:
        for k in range(len(dem[load])):
           if dem[load][k] < eps:
              dem[load][k] = 0
    
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    devs = {}

    #%% BOILER
    devs["BOI"] = {"inv_var": 52,       # kEUR/MW_th,       variable investment
                   "eta_th": 0.95,      # ---,              thermal efficiency
                   "life_time": 30,     # a,                operation time
                   "cost_om": 0.01,     # ---,              annual operation and maintenance costs as share of investment
                   }

    #%% COMBINED HEAT AND POWER
    devs["CHP"] = {"inv_var": 570,      # kEUR/MW_el,       variable investment
                   "eta_el": 0.35,      # ---,              electrical efficiency
                   "eta_th": 0.5,       # ---,              thermal efficiency
                   "life_time": 30,     # a,                operation time
                   "cost_om": 0.05,     # ---,              annual operation and maintenance costs as share of investment
                   }   

    #%% ABSORPTION CHILLER
    devs["AC"] = {"inv_var": 78,        # kEUR/MW_th,       variable investment
                  "eta_th": 0.8,        # ---,              thermal efficiency (cooling power / heating power)
                  "life_time": 20,      # a,                operation time
                  "cost_om": 0.05,      # ---,              annual operation and maintenance costs as share of investment
                  }

    #%% COMPRESSION CHILLER
    devs["CC"] = {"inv_var": 78,        # kEUR/MW_th,       variable investment
                  "COP": 5,             # ---,              coefficient of performance
                  "life_time": 20,      # a,                operation time
                  "cost_om": 0.03,      # ---,              annual operation and maintenance costs as share of investment
                  }

    # Calculate annualized investment of every device
    devs = calc_annual_investment(devs, param)   
    return (devs, param, dem)

#%%
#def get_irrad_profile(ele, azim, weather_dict):
#    """
#    Calculates global irradiance on tilted surface from weather file.
#    """
#
#    # Load time series as numpy array
#    dtype = dict(names = ['id','data'], formats = ['f8','f8'])
#    sun_diffuse = np.array(list(weather_dict["Diffuse Horizontal Radiation"].items()), dtype=dtype)['data']
#    sun_global = np.array(list(weather_dict["Global Horizontal Radiation"].items()), dtype=dtype)['data']
#    sun_direct = sun_global - sun_diffuse
#
#    # Define local properties
#    time_zone = 7                # ---,      time zone (weather file works properly with time_zone = 7, although time_zone = 8 is proposed in the weather file)
#    location = (31.17, 121.43)   # degree,   latitude, longitude of location
#    altitude = 7.0               # m,        height of location above sea level
#
#    # Calculate geometric relations
#    geometry = sun.getGeometry(0, 3600, 8760, time_zone, location, altitude)
#    (omega, delta, thetaZ, airmass, Gon) = geometry
#    theta = sun.getIncidenceAngle(ele, azim, location[0], omega, delta)
#
#    theta = theta[1] # cos(theta) is not required
#
#    # Calculate radiation on tilted surface
#    return sun.getTotalRadiationTiltedSurface(theta, thetaZ, sun_direct, sun_diffuse, airmass, Gon, ele, 0.2)

#%%
#def calc_pv(dev, weather_dict):
#    """
#    Calculates photovoltaic output in MW per MW_peak.
#    Model based on http://www.sciencedirect.com/science/article/pii/S1876610213000829, equation 5.
#
#    """
#
#    # Calculate global tilted irradiance in W/m2
#    gti_pv = get_irrad_profile(dev["elevation"], dev["azimuth"], weather_dict)
#
#    # Get ambient temperature from weather dict
#    temp_amb = np.array(list(weather_dict["Dry Bulb Temperature"].items()), dtype=dict(names = ['id','data'], formats = ['f8','f8']))['data']
#
#    temp_cell = temp_amb + gti_pv / dev["solar_noct"] * (dev["temp_cell_noct"] - temp_amb)
#    eta_noct = dev["power_noct"] / (dev["module_area"] * dev["solar_noct"])
#    eta_cell = eta_noct * (1 - dev["gamma"] * (temp_cell - dev["temp_amb_noct"]))
#
#    # Calculate collector area (m2) per installed capacity (MW_peak)
#    area_per_MW_peak = dev["module_area"] / (dev["nom_module_power"] / 1000000)
#
#    # Calculate power generation in MW/MW_peak
#    pv_output = eta_cell * (gti_pv / 1000000) * area_per_MW_peak
#
#    return dict(enumerate(pv_output))

#%%
#def calc_stc(devs, weather_dict):
#    """
#    Calculation of thermal output in MW/MW_peak according to ISO 9806 standard (p. 43).
#
#    """
#
#    dev = devs["STC"]
#
#    # Calculate global tilted irradiance in W/m2
#    gti_stc = get_irrad_profile(dev["elevation"], dev["azimuth"], weather_dict)
#
#    # Get ambient temperature from weather dict
#    temp_amb = np.array(list(weather_dict["Dry Bulb Temperature"].items()), dtype=dict(names = ['id','data'], formats = ['f8','f8']))['data']
#
#    # Calculate heat output in W/m2
#    stc_output_m2 = np.zeros(gti_stc.size)
#    t_norm = (dev["temp_mean"] - temp_amb) / gti_stc
#    eta_th = dev["eta_0"] - dev["a1"] * t_norm - dev["a2"] * t_norm**2 #* gti_stc
#    for t in range(eta_th.size):
#        if not np.isfinite(eta_th[t]):
#            eta_th[t] = 0
#        stc_output_m2[t] = max(eta_th[t] * gti_stc[t], 0)
#
#    # Calculate collector area (m2) per installed capacity (MW_peak)
#    area_per_MW_peak = 1000000 / dev["power_per_m2"]
#
#    # Calculate thermal heat output in MW/MW_peak
#    stc_output = stc_output_m2 * area_per_MW_peak / 1000000
#
#    return dict(enumerate(stc_output))

#%%
#def calc_wind(dev, weather_dict):
#    """
#    Calculation power output from wind turbines in MW/MW_peak.
#    
#    """
#    
#    power_curve = dev["power_curve"]
#    
#    dev["power"] = {}
#    for t in range(len(weather_dict["Wind Speed"])):
#        wind_speed_ground = weather_dict["Wind Speed"][t]
#        wind_speed_shaft = wind_speed_ground * (dev["hub_height"] / dev["ref_height"]) ** dev["expo_a"]
#        
#        # if cases can then be eliminated, if np.interp is used
#        if wind_speed_shaft <= 0:
#            dev["power"][t] = 0
#        elif wind_speed_shaft > power_curve[len(power_curve)-1][0]:
#            print("Warning: Wind speed is " + str(wind_speed_shaft) + " m/s and exceeds wind power curve table.")
#            dev["power"][t] = 0
#    
#        # Linear interpolation
#        
#        # better use: #    res = np.interp(2.5, speed_points, power_points)
#        # do not use this extra function calc_wind, move it directly to wind data section
#        
#        else:
#            for k in range(len(power_curve)):
#                if power_curve[k][0] > wind_speed_shaft:
#                    dev["power"][t] = (power_curve[k][1]-power_curve[k-1][1])/(power_curve[k][0]-power_curve[k-1][0]) * (wind_speed_shaft-power_curve[k-1][0]) + power_curve[k-1][1]
#                    break
#            
#    return dev

#%%
#def calc_COP_AHSP(devs, weather_dict):
#    """
#    Calculation of COP for air source heat pump based on carnot efficiency.
#
#    """
#
#    devs["ASHP"]["COP"] = {}
#    for t in weather_dict["Dry Bulb Temperature"].keys():
#        air_temp = weather_dict["Dry Bulb Temperature"][t]
#        devs["ASHP"]["COP"][t] = devs["ASHP"]["eta"] * (devs["ASHP"]["t_supply"]/(devs["ASHP"]["t_supply"]-(air_temp + 273)))
#    return devs["ASHP"]["COP"]

#%%
def calc_annual_investment(devs, param):
    """
    Calculation of total investment costs per device including replacements and residual value (based on VDI 2067-1, pages 16-17).
    
    Annualized fix and variable investment is returned.
    
    """

    observation_time = param["observation_time"]
    interest_rate = param["interest_rate"]
    q = 1 + param["interest_rate"]

    # Calculate capital recovery factor (=Annuitätenfaktor)
    CRF = ((q**observation_time)*interest_rate)/((q**observation_time)-1)

    for device in devs.keys():

        # If there are no fix costs:
        devs[device]["inv_fix"] = 0
        
        # If there are no replacement costs:
        life_time = devs[device]["life_time"]
        inv_fix_init = devs[device]["inv_fix"]
        inv_var_init = devs[device]["inv_var"]
        inv_fix_repl = devs[device]["inv_fix"]
        inv_var_repl = devs[device]["inv_var"]

        # Number of required replacements
        n = int(math.floor(observation_time / life_time))

        # Inestment for replcaments
        invest_replacements = sum((q ** (-i * life_time)) for i in range(1, n+1))

        # Residual value of final replacement
        res_value = ((n+1) * life_time - observation_time) / life_time * (q ** (-observation_time))

        # Calculate annualized investments       
        if life_time > observation_time:
            devs[device]["ann_inv_fix"] = (inv_fix_init * (1 - res_value)) * CRF 
            devs[device]["ann_inv_var"] = (inv_var_init * (1 - res_value)) * CRF 
        else:
            devs[device]["ann_inv_fix"] = (inv_fix_init + inv_fix_repl * (invest_replacements - res_value)) * CRF
            devs[device]["ann_inv_var"] = (inv_var_init + inv_var_repl * (invest_replacements - res_value)) * CRF 

    return devs