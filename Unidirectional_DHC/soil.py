# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 16:13:13 2018

@author: lkivi
"""

import numpy as np
from scipy.optimize import leastsq
import cmath
import pylab as plt
import json

import parameter
from grid import generateJson


#%%
# Calculate thermal losses
def calculateLosses():
    
    _, param, _ = parameter.load_params()
    generateJson()
    data = json.loads(open("nodes.json").read())
    
    T_soil = calculateSoilTemperature()
    
    Losses = {}
    Losses["heating_grid"] = np.zeros(8760)
    Losses["cooling_grid"] = np.zeros(8760)
    
    for item in data["edges"]:
        d = item["diameter"]
        L = item["distance"]      
        k = (d/2 * 1/param["lambda_ins"] * np.log((d+2*param["t_pipe"]+2*param["t_ins"])/(d+2*param["t_pipe"])))**0.5         # W/(m^2*K)   heat transfer coefficient 
        
        Losses["heating_grid"] = Losses["heating_grid"] + k*np.pi*d*L*((param["T_heating_supply"] - T_soil) + (param["T_heating_return"] - T_soil))/1e6
        Losses["cooling_grid"] = Losses["cooling_grid"] + k*np.pi*d*L*((T_soil - param["T_cooling_supply"]) + (T_soil - param["T_cooling_return"]))/1e6
    
    print(Losses["heating_grid"])
    print(Losses["cooling_grid"])
    
    plt.plot(np.arange(8760)/24, Losses["cooling_grid"])
    plt.show
    
    return Losses
    


#%%
# Calculate and return time series of soil temperature in grid depth
def calculateSoilTemperature(): 
    
    _, param, _ = parameter.load_params()
    
    # Load weather data
    path_weather = "input_data/weather.csv"
    weather = {}
    
    weather["T_air"] = np.loadtxt(open(path_weather, "rb"), delimiter = ",",skiprows = 1, usecols=(0))          # Air temperatur °C
    weather["v_wind"] = np.loadtxt(open(path_weather, "rb"), delimiter = ",",skiprows = 1, usecols=(1))         # Wind Velocity m/s
    weather["r"] = np.loadtxt(open(path_weather, "rb"), delimiter = ",",skiprows = 1, usecols=(2))              # relative humidity -
    weather["G"] = np.loadtxt(open(path_weather, "rb"), delimiter = ",",skiprows = 1, usecols=(3))              # Global radiation W/m^2
    weather["x"] = np.loadtxt(open(path_weather, "rb"), delimiter = ",",skiprows = 1, usecols=(4))              # absolute humidity g/kg
    weather["p"] = np.loadtxt(open(path_weather, "rb"), delimiter = ",",skiprows = 1, usecols=(5))              # air pressure hPa
    
    
    # Calculate T_sky
    hours_day = np.arange(1,24)
    hours_day = np.append(hours_day,0)
    hours_year = []
    for i in range(365):
        hours_year = np.append(hours_year, hours_day)  #hours since midnight
    weather["p_w_dp"] = (weather["x"]/1000 * weather["p"]*100)/(0.622 + weather["x"]/1000)                                                                  # partial water pressure at dew point Pa
    weather["T_dp"] = (243.12*np.log(weather["p_w_dp"]/611.2))/(17.62-np.log(weather["p_w_dp"]/611.2))                                                      # dew point temperatur °C
    weather["T_sky"] = (weather["T_air"]+273.15)*((0.711+0.0056*weather["T_dp"]+0.000073*weather["T_dp"]**2+0.013*np.cos(15*hours_year))**0.25)-273.15      # sky temperature °C
    
    
    # Cosinus Fit of G, T_air and T_sky: X = mean - amp * cos(omega*t - phase)
    G_mean, G_amp, G_phase = cosFit(weather["G"])
    Tair_mean, Tair_amp, Tair_phase = cosFit(weather["T_air"])
    Tsky_mean, Tsky_amp, Tsky_phase = cosFit(weather["T_sky"])
    
    # convective heat transfer at surface W/(m^2*K)
    weather["alpha_conv"] = np.zeros(8760)
    for hour in range(8760):
        if weather["v_wind"][hour] <= 4.88:
            weather["alpha_conv"][hour] = 5.7 + 3.8 * weather["v_wind"][hour]**0.5
        else:
            weather["alpha_conv"][hour] = 7.2*weather["v_wind"][hour]**0.78
    alpha_conv = np.mean(weather["alpha_conv"])
    
    # mean relative air humidity
    r = np.mean(weather["r"])
    
    
    # get ground parameters
    omega = 2*np.pi/365
    
    if param["asphaltlayer"] == 0:       #no asphalt layer, only soil
        alpha_s = param["alpha_soil"]
        epsilon_s = param["epsilon_soil"]
        f = param["evaprate_soil"]
        k = param["lambda_soil"]
        delta_s = (2*(k/param["heatcap_soil"]*3600*24)/omega)**0.5        # damping depth soil m
        delta_soil = delta_s
    else:                               # asphalt layer at surface
        alpha_s = param["alpha_asph"]
        epsilon_s = param["epsilon_asph"]
        f = param["evaprate_asph"]
        k = param["lambda_asph"]
        delta_s = (2*(k/param["heatcap_asph"]*3600*24)/omega)**0.5                       # damping depth asphalt layer m  
        delta_soil = (2*(param["lambda_soil"]/param["heatcap_soil"]*3600*24)/omega)**0.5      # damping depth soil m       
 
  
    # radiation heat transfer at surface W/(m^2*K)
    alpha_rad = 5         # start value
    
    for i in range(1000):
        h_e = alpha_conv*(1+103*0.0168*f) + alpha_rad
        h_r = alpha_conv*(1+103*0.0168*f*r)
        
        Ts_mean = (h_r*Tair_mean+alpha_rad*Tsky_mean+alpha_s*G_mean-0.0168*609*f*alpha_conv*(1-r))/h_e
        
        num = (h_r*Tair_amp+alpha_s*cmath.rect(G_amp,Tair_phase-G_phase)+alpha_rad*cmath.rect(Tsky_amp,Tair_phase-Tsky_phase))
        denom = (h_e+k*((1+1j)/delta_s))
        z = num/denom
        
        Ts_amp = abs(z)
        Ts_phase = Tair_phase + cmath.phase(z)
              
        # recalculate alpha_rad
        omega = 2*np.pi/365/24
        time = np.arange(1,8761)
        weather["T_surface"] = Ts_mean - Ts_amp*np.cos(omega*time-Ts_phase)
        weather["alpha_rad"] = epsilon_s * 5.67e-8 * (weather["T_surface"] + weather["T_sky"]+273.15+273.15)*((weather["T_surface"]+273.15)**2 + (weather["T_sky"]+273.15)**2)
        alpha_rad_new = np.mean(weather["alpha_rad"])
        
        if (alpha_rad_new - alpha_rad)**2 < 1e-5:
            break
        
        alpha_rad = alpha_rad_new
 
    # Calculate soil temperature in grid depth
    d = param["d_asph"] 
    t = param["grid_depth"]
    omega = 2*np.pi/365/24
    time = np.arange(1,8761)
   
    if param["asphaltlayer"] == 0: # no asphalt
        weather["T_soil"] = Ts_mean - Ts_amp*np.exp(-t/delta_soil)*np.cos(omega*time - Ts_phase - t/delta_soil)
    else:   # with asphalt layaer
        if t > d: # grid is below asphalt layer
            weather["T_soil"] = Ts_mean - Ts_amp*np.exp(-d/delta_s)*np.exp(-(t-d)/delta_soil)*np.cos(omega*time - Ts_phase - d/delta_s - (t-d)/delta_soil)
        else: # grid is within asphalt layer
            weather["T_soil"] = Ts_mean - Ts_amp*np.exp(-t/delta_s)*np.cos(omega*time - Ts_phase - t/delta_s)
   
 
    #plt.plot(time/24, weather["T_soil"])
    #plt.show()
    
    T_soil = weather["T_soil"]
    
    return T_soil




#%%
def cosFit(data):
    
    omega = 2*np.pi/8760
    time = np.arange(1,8761)
    
    start_mean = np.mean(data)
    start_amp = np.std(data)* (2**0.5)
    start_phase = 0
    
    func = lambda x: x[0] - x[1]*np.cos(omega*time-x[2]) - data
    mean, amp, phase = leastsq(func, [start_mean, start_amp, start_phase])[0]
    
    #data_fit = mean - amp*np.cos(omega*time - phase)
    #plt.plot(time, data, '.')
    #plt.plot(time, data_fit)
    #plt.show()

    return mean, amp, phase


#%%


calculateLosses()







#%%