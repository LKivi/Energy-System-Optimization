# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 15:12:14 2018

@author: mwi
"""


from pyproj import Proj, transform
import math
import numpy as np
import networkx as nx
import os
import pylab as plt


def load_params(use_case, path_file):
    
    assert (use_case != "FZJ" or use_case != "EON" or use_case != "simple_test"), "Use case '" + use_case + "' not known."
    path_input = path_file + "\\input_data\\" + use_case + "\\"
    print("Using data set: '" + use_case + "'")
    time_steps = range(8760)
    
    nodes = {}       
    
#%% Use case: EON (example data provided by E.ON)
#    if use_case == "EON":
        
#        air_temp = np.loadtxt(path_demand_files + "air_temp.txt")
#        
#        flow_temp = - (0.7471 * air_temp) + 54.75 # source: ectoplanner excel sheet by E.ON
#        # Vorlauftemperatur 
#        outlet_temp_cond = flow_temp
#        inlet_temp_evap = get_grid_temp_hot_pipe("two_point_control", time_steps)
#        COP_carnot = np.abs((outlet_temp_cond+273.15)/(outlet_temp_cond-inlet_temp_evap))
#        COP_HP = 3.3829 * np.log(COP_carnot) - 2.8708
#        print("WARNING: USING WRONG COP DEFINITION FOR TESTING!")
#        COP_CC = COP_HP - 1
#
#        nodes[0] = {"lon":               6.402940, # °,         longitude
#                    "lat":              50.908491, # °,         latitude
#                    "name":              "bldg_0", # ---,       building name is assigned automatically
#                    "heated_area":           1000, # m2,        area that is heated
#                    "spec_heat_dem":          100, # kWh/m2,    specific demand (describes to building standard)
#                    "is_cooled":             True, # ---,      defines if the building is cooled in summer
#                    "bldg_type":    "residential", # either "residential" or "office"
#                    }
#        
#        nodes[0]["heat"] = np.loadtxt(open(path_demand_files + nodes[0]["name"] + "_heating.txt", "rb"), delimiter=",", usecols=(0)) 
#        # if a time series is given the attributes "heated_area", "spec_heat_dem", "is_cooled" and "residential" are ignored
#        nodes[0]["cool"] = np.loadtxt(open(path_demand_files + nodes[0]["name"] + "_cooling.txt", "rb"), delimiter=",", usecols=(0))
#
#        nodes[1] = {"lon":               6.403199, # °,         longitude
#                    "lat":              50.909126, # °,         latitude
#                    "name":              "bldg_1", # ---,       building name is assigned automatically
#                    "heated_area":              0, # m2,        area that is heated
#                    "spec_heat_dem":            0, # kWh/m2,    specific demand (describes to building standard)
#                    "is_cooled":             True, # ---,      defines if the building is cooled in summer
#                    "bldg_type":               "", # either "residential" or "office"
#                    }
#        nodes[1]["heat"] = np.loadtxt(open(path_demand_files + nodes[1]["name"] + "_heating.txt", "rb"), delimiter=",", usecols=(0))
#        nodes[1]["cool"] = np.loadtxt(open(path_demand_files + nodes[1]["name"] + "_cooling.txt", "rb"), delimiter=",", usecols=(0))
        
    if use_case == "FZJ":
        
        # load node data 
        path_nodes = path_input + "nodes.txt"
        path_demands = path_input + "demands\\"
        latitudes =  np.loadtxt(open(path_nodes, "rb"), delimiter = ",", usecols=(0))                       # °,        node latitudes
        longitudes =  np.loadtxt(open(path_nodes, "rb"), delimiter = ",", usecols=(1))                      # °,        node latitudes
        buildings = np.genfromtxt(open(path_nodes, "rb"),dtype = 'str', delimiter = ",", usecols=(3))       # --,       names of considered buildings
                    
#        for index in range(len(latitudes)):
        for index in range(3):
            nodes[index] = {"lat": latitudes[index],
                            "lon": longitudes[index],
                            "name": buildings[index],
                            "dem_heat": np.loadtxt(open(path_demands + buildings[index] + "_heating.txt", "rb"),delimiter = ",", usecols=(0)),
                            "dem_cool": np.loadtxt(open(path_demands + buildings[index] + "_cooling.txt", "rb"),delimiter = ",", usecols=(0))
                            }
        

        nodes = transform_coordinates(nodes)
        
#        for k in range(len(nodes)):
#            plt.plot(nodes[k]["x"], nodes[k]["y"],".")
#        
#        plt.show()


#%% GENERAL PARAMETERS
    param = {"interest_rate":  0.05,        # ---,          interest rate
             "observation_time": 20.0,      # a,            project lifetime
             "price_gas": 0.02824,          # kEUR/MWh,     natural gas price
             "price_cap_gas": 12.149,       # kEUR/(MW*a)   capacity charge for gas grid usage
             "price_el": 0.14506,           # kEUR/MWh,     electricity price
             "price_cap_el": 59.660,        # kEUR/(MW*a)   capacity charge for electricity grid usage
             "self_charge": 0.0272,         # kEUR/MWh      charge on on-site consumption of CHP-generated power   
             "revenue_feed_in": 0.05931,    # kEUR/MWh,      feed-in revenue for CHP-gernerated power
             "gas_CO2_emission": 0.2,       # t_CO2/MWh,    specific CO2 emissions (natural gas)
             "grid_CO2_emission": 0.503,    # t_CO2/MWh,    specific CO2 emissions (grid)
             "MIPGap":      0.0001,         # ---,          MIP gap            
             
             "number_of_balancing_units": 1,
             
             
             "price_cool": 1000,            # kEUR/MWh,     price for cooling power from the district cooling grid
             "price_heat": 1000,            # kEUR/MWh,     price for heating power from the district heating grid
             "costs_piping": 1,             # EUR/m,        costs for earth work and pipe materia per meter pipe installation
             "use_eh_in_bldgs": 0,          # ---,          should electric heaters be used in buildings?
             "op_hours_el_heater": 1000,    # h,            hours in which the eletric heater is operated
             "eta_th_eh": 0.98,             # ---,          thermal efficiency for electric heaters in buildings
             "obj_weight_tac": 0.9,         # ---,          weight for objective function, co2 emission is then 1-obj_weight_tac
             "feasible_TES": 1,             # ---,          are thermal energy storages feasible for BU?
             "feasible_BAT": 1,             # ---,          are batteries feasible for BU?
             "feasible_CTES": 1,            # ---,          are cold thermal energy storages feasible for BU?
             "feasible_BOI": 1,             # ---,          are gas-fired boilers feasible for BU?
             "feasible_from_DH": 1,         # ---,          is a connection to district heating network possible?
             "feasible_from_DC": 1,         # ---,          is a connection to district cooling network possible?
             "feasible_CHP": 1,             # ---,          are CHP units feasible for BU?
             "feasible_EH": 1,              # ---,          are electric heater feasible for BU?
             "feasible_CC": 1,              # ---,          are compression chiller feasible for BU?
             "feasible_AC": 1,              # ---,          are absorbtion chiller feasible for BU?
             }
    
    #%% SOIL PARAMETERS   
    param_soil = {"alpha_soil": 0.8,            #---,       soil surface absorptance
                  "epsilon_soil": 0.9,          #---,       soil surface emissivity
                  "evaprate_soil": 0.7,         #---,       soil surface evaporation rate
                  "lambda_soil": 1.9,           # W/(m*K),  soil heat conductivity
                  "heatcap_soil": 2.4e6}        # J/(m^3*K),soil volumetric heat capacity 
    
    param.update(param_soil)
    
    
    #%% ASPHALT LAYER PARAMETERS
    param_asphalt = {"asphaltlayer": 1,          #---,       consideration of asphalt layer? 1 = yes, 0 = no
                     "d_asph": 0.18,             # m,        asphalt layer thickness
                     "alpha_asph": 0.93,         #---,       asphalt surface absorptance
                     "epsilon_asph": 0.88,       #---,       asphalt surface emissivity
                     "evaprate_asph": 0.3,       #---,       asphalt surface evaporation rate
                     "lambda_asph": 0.7,         # W/(m*K),  asphalt heat conductivity
                     "heatcap_asph": 1950400}    # J/(m^3*K),asphalt volumetric heat capacity
    
    param.update(param_asphalt)  
      
    
    #%% PIPE PARAMETERS
    param_pipe = {"grid_depth": 1.5,                # m,       installation depth beneath surface
                  "lambda_PE": 0.5,                 # W(m*K),  PE heat conductivity
                  "f_fric": 0.025,                  # ---,     pipe friction factor
                  "dp_pipe": 150,                   # Pa/m,    nominal pipe pressure gradient (for network without heat losses)
                  "c_f": 4180,                      # J/(kg*K),fluid specific heat capacity
                  "rho_f": 1000,                    # kg/m^3,  fluid density
                  "t_soil": 0.6}                    # m,       thickness of soil layer around the pipe to calculate heat transfer into ground
                  
    param.update(param_pipe)  
    
    param_pipe_eco = {"inv_earth_work": 300,                 # EUR/m,              preparation costs for pipe installation
                       "inv_pipe_var_per_length": 4272.168,  # EUR/(m^(5/2)*m),    diameter price for PE pipe per metre
                       "inv_pipe_fix_per_length": 0,
                       "pipe_lifetime": 50,                  # a,        pipe life time
                       "cost_om_pipe": 0.01,                 #---,       pipe operation and maintetance costs
                       }
                
    param.update(param_pipe_eco)
    
    


    #%% TEMPERATURES
    param_temperatures = {"T_hot": 20,      # °C,   hot pipe temperature
                          "T_cold": 12,     # °C,   cold pipe temperature
                          }
    
    param.update(param_temperatures)
     
    
    param["COP_HP"] = 5
    param["COP_CC"] = 4
    
    #%% LOAD DEVICE PARAMETER
    
    devs = {}

    
    
    #%% BOILER
    devs["BOI"] = {
                   "eta_th": 0.9,       # ---,    thermal efficiency
                   "life_time": 20,     # a,      operation time (VDI 2067)
                   "cost_om": 0.03,     # ---,    annual operation and maintenance costs as share of investment (VDI 2067)
                   }
    
    
    devs["BOI"]["cap_i"] =  {  0: 0,        # MW_th 
                               1: 0.5,      # MW_th
                               2: 5         # MW_th
                               }
    
    devs["BOI"]["inv_i"] = {    0: 0,       # kEUR
                                1: 33.75,   # kEUR
                                2: 96.2     # kEUR
                                }


    #%% COMBINED HEAT AND POWER - INTERNAL COMBUSTION ENGINE POWERED BY NATURAL GAS
    devs["CHP"] = {
                   "eta_el": 0.419,     # ---,            electrical efficiency
                   "eta_th": 0.448,     # ---,           thermal efficiency
                   "life_time": 15,     # a,               operation time (VDI 2067)
                   "cost_om": 0.08,     # ---,             annual operation and maintenance costs as share of investment (VDI 2067)
                   }   
    
    devs["CHP"]["cap_i"] =  {  0: 0,        # MW_el
                               1: 0.25,     # MW_el
                               2: 1,        # MW_el
                               3: 3         # MW_el
                               }
    
    devs["CHP"]["inv_i"] = {    0: 0,           # kEUR
                                1: 211.15,      # kEUR
                                2: 410.7,       # kEUR
                                3: 707.6        # kEUR
                                } 
    

    
    #%% ABSORPTION CHILLER
    devs["AC"] = {
                  "eta_th": 0.68,       # ---,        nominal thermal efficiency (cooling power / heating power)
                  "life_time": 18,      # a,          operation time (VDI 2067)
                  "cost_om": 0.03,      # ---,        annual operation and maintenance costs as share of investment (VDI 2067)
                  }
    
    devs["AC"]["cap_i"] =   {  0: 0,        # MW_th
                               1: 0.25,     # MW_th
                               2: 1.535,    # MW_th
                               3: 5.115     # MW_th
                               }
    
    devs["AC"]["inv_i"] = {     0: 0,           # kEUR
                                1: 135.5,       # kEUR
                                2: 313.672,     # kEUR
                                3: 619.333      # kEUR
                                } 

    #%% COMPRESSION CHILLER
    devs["CC"] = {
                  "COP": 4,             # ---,             nominal coefficient of performance
                  "life_time": 15,      # a,               operation time (VDI 2067)
                  "cost_om": 0.035,     # ---,             annual operation and maintenance costs as share of investment (VDI 2067)
                  }
    
    
    devs["CC"]["cap_i"] = { 0: 0,       # MW_th
                            1: 0.5,     # MW_th
                            2: 4        # MW_th
                            }
    
    
    devs["CC"]["inv_i"] = { 0: 0,         # kEUR
                            1: 94.95,     # kEUR
                            2: 402.4      # kEUR
                            } 
    
    #%% (HEAT) THERMAL ENERGY STORAGE
    devs["TES"] = {
                   "switch_TES": 0,     # toggle availability of thermal storage
                   "max_cap": 250,      # MWh_th,          maximum thermal storage capacity
                   "min_cap": 0,        # MWh_th,           minimum thermal storage capacity              
                   "sto_loss": 0.005,   # 1/h,              standby losses over one time step
                   "eta_ch": 0.975,     # ---,              charging efficiency
                   "eta_dch": 0.975,    # ---,              discharging efficiency
                   "max_ch": 1000,      # MW,               maximum charging power
                   "max_dch": 1000,     # MW,               maximum discharging power
                   "soc_init": 0.8,     # ---,              maximum initial state of charge
                   "soc_max": 1,        # ---,              maximum state of charge
                   "soc_min": 0,        # ---,              minimum state of charge
                   "life_time": 20,     # a,                operation time (VDI 2067 Trinkwasserspeicher)
                   "cost_om": 0.02,     # ---,              annual operation and maintenance costs as share of investment (VDI 2067 Trinkwasserspeicher)

                   }
    
    devs["TES"]["cap_i"] =   { 0: 0,         # MWh_th,      depends on temperature difference! Q = V * c_p * rho * dT
                               1: 8.128,     # MWh_th
                               2: 40.639,    # MWh_th
                               3: 243.833    # MWh_th
                               }
    
    devs["TES"]["inv_i"] = {    0: 0,              # kEUR
                                1: 147.2,          # kEUR,    includes factor of 1.15 for pressure correction factor due to high temperatures; higher pressure is needed to prevent evaporation
                                2: 410.55,         # kEUR
                                3: 1083.3          # kEUR
                                } 
    
    #%% ELECTRICAL HEATER
    devs["EH"] = {"inv_var": 78,        # kEUR/MW_th,       variable investment
                  "eta_th": 0.9,        # ---,              thermal efficiency
                  "life_time": 20,      # a,                operation time
                  "cost_om": 0.01,      # ---,              annual operation and maintenance costs as share of investment
                  }
    
    #%% HEAT PUMP
    
    
    
#%%        
#    #%% BATTERY
#    devs["BAT"] = {"inv_var": 520,      # kEUR/MWh_el,      variable investment
#                   "max_cap": 50,       # MWh_el,           maximum eletrical storage capacity
#                   "sto_loss": 0,       # 1/h,              standby losses over one time step
#                   "eta_ch": 0.9592,    # ---,              charging efficiency
#                   "eta_dch": 0.9592,   # ---,              discharging efficiency
#                   "soc_init": 0.8,     # ---,              maximum initial relative state of charge
#                   "soc_max": 0.8,        # ---,              maximum relative state of charge
#                   "soc_min": 0.2,        # ---,              minimum relative state of charge
#                   "life_time": 10,     # a,                operation time
#                   "cost_om": 0.02,     # ---,              annual operation and maintenance costs as share of investment
#                   }



#    #%% COLD THERMAL ENERGY STORAGE
#    devs["CTES"] = {"inv_var": 11.7,    # kEUR/MWh_th,      variable investment
#                    "max_cap": 5000,      # MWh_th,           maximum thermal storage capacity
#                    "sto_loss": 0.005,  # 1/h,              standby losses over one time step
#                    "eta_ch": 0.975,    # ---,              charging efficiency
#                    "eta_dch": 0.975,   # ---,              discharging efficiency
#                    "soc_init": 0.8,    # ---,              maximum initial state of charge
#                    "soc_max": 1,       # ---,              maximum state of charge
#                    "soc_min": 0,       # ---,              minimum state of charge
#                    "life_time": 20,    # a,                operation time
#                    "cost_om": 0.01,    # ---,              annual operation and maintenance costs as share of investment
#                    }
    
#    #%% CONNECTION TO DISTRICT COOLING NETWORK
#    devs["from_DC"] = {"inv_var": 11.7,    # kEUR/MW_th,      variable investment
#                       "max_cap": 5000,    # MW_th,           maximum thermal storage capacity
#                       "min_cap": 0,       # MW_th,           minimum thermal storage capacity              
#                       "eta_th": 0.99,     # ---,             discharging efficiency
#                       "life_time": 50,    # a,               operation time
#                       "cost_om": 0.01,    # ---,             annual operation and maintenance costs as share of investment
#                       }

#    #%% CONNECTION TO DISTRICT HEATING NETWORK
#    devs["from_DH"] = {"inv_var": 11.7,    # kEUR/MW_th,      variable investment
#                       "max_cap": 5000,    # MW_th,           maximum thermal storage capacity
#                       "min_cap": 0,       # MW_th,           minimum thermal storage capacity              
#                       "eta_th": 0.99,     # ---,             discharging efficiency
#                       "life_time": 50,    # a,               operation time
#                       "cost_om": 0.01,    # ---,             annual operation and maintenance costs as share of investment
#                       }

    # Calculate annualized investment of every device
    devs = calc_annual_investment(devs, param)

    return nodes, param, devs, time_steps

def get_grid_temp_hot_pipe(mode, time_steps):
    if mode == "two_point_control":
        grid_temp_hot_pipe = np.zeros(len(time_steps))
        grid_temp_summer = 16
        grid_temp_winter = 22
        grid_temp_hot_pipe[0:3754] = grid_temp_winter
        grid_temp_hot_pipe[3754:7040] = grid_temp_summer
        grid_temp_hot_pipe[7040:8760] = grid_temp_winter
        
        with open("D:\\mwi\\Gurobi_Modelle\EctoPlanner\\temp.txt", "w") as outfile:
            for t in time_steps:
                outfile.write(str(round(grid_temp_hot_pipe[t],3)) + "\n")   
        
    return grid_temp_hot_pipe

#%%
def calc_pipe_costs(nodes, edges, edge_dict_rev, param):
    """
    Calculate variable and fix costs for every edge.
    """
    c_fix = {}
    c_var = {}
    for e in edges:
        x1, y1 = nodes[edge_dict_rev[e][0]]["x"], nodes[edge_dict_rev[e][0]]["y"]
        x2, y2 = nodes[edge_dict_rev[e][1]]["x"], nodes[edge_dict_rev[e][1]]["y"]
        c_fix[e] = (param["inv_earth_work"] + param["inv_pipe_fix_per_length"]) * math.sqrt((x1-x2)**2 + (y1-y2)**2)
        c_var[e] = param["inv_pipe_var_per_length"] * math.sqrt((x1-x2)**2 + (y1-y2)**2)
    
    print("Mindestkapazitaet vorsehen fuer Rohre")
    
    param["inv_pipe_fix"] = c_fix
    param["inv_pipe_var"] = c_var
    return param


#%%
def get_edge_dict(n):
    compl_graph = nx.complete_graph(n)                                                      # Creates graph with n nodes 0 to n-1 and edges between every pair of nodes
    edge_list = list(compl_graph.edges(data=False))                                         # get list of edges
    edge_dict = {(edge_list[k][0], edge_list[k][1]): k for k in range(len(edge_list))}      # dicts indcluding edge numbers
    edge_dict_rev = {k: (edge_list[k][0], edge_list[k][1]) for k in range(len(edge_list))}
    edges = range(len(edge_list))                                                           # list containing edge indices
    return edge_dict, edge_dict_rev, edges, compl_graph

#%%
def transform_coordinates(nodes):
    outProj = Proj(init='epsg:25832')   # ETRS89 / UTM zone 32N
    inProj = Proj(init='epsg:4258')     # Geographic coordinate system: EPSG 4326
    
    # get x- and y- coordinates and find minimal values of each
    min_x, min_y = transform(inProj,outProj,nodes[0]["lon"],nodes[0]["lat"])
    for n in range(len(nodes)):
        nodes[n]["x"],nodes[n]["y"] = transform(inProj,outProj,nodes[n]["lon"],nodes[n]["lat"])
        if nodes[n]["x"] < min_x:
            min_x = nodes[n]["x"]
        if nodes[n]["y"] < min_y:
            min_y = nodes[n]["y"]
    
    # Shift coordinate system by minimal x- and y- value        
    for n in range(len(nodes)):
        nodes[n]["x"] = nodes[n]["x"] - min_x
        nodes[n]["y"] = nodes[n]["y"] - min_y
        
    return nodes

#%%
def calc_annual_investment(devs, param):
    """
    Calculation of total investment costs including replacements (based on VDI 2067-1, pages 16-17).

    Parameters
    ----------
    dev : dictionary
        technology parameter
    param : dictionary
        economic parameters

    Returns
    -------
    annualized fix and variable investment
    """

    observation_time = param["observation_time"]
    interest_rate = param["interest_rate"]
    q = 1 + param["interest_rate"]

    # Calculate capital recovery factor
    CRF = ((q**observation_time)*interest_rate)/((q**observation_time)-1)

    # Calculate annuity factor for each device
    for device in devs.keys():
        
        # Get device life time
        life_time = devs[device]["life_time"]

        # Number of required replacements
        n = int(math.floor(observation_time / life_time))
        
        # Inestment for replcaments
        invest_replacements = sum((q ** (-i * life_time)) for i in range(1, n+1))

        # Residual value of final replacement
        res_value = ((n+1) * life_time - observation_time) / life_time * (q ** (-observation_time))

        # Calculate annualized investments       
        if life_time >= observation_time:
            devs[device]["ann_factor"] = (1 - res_value) * CRF 
        else:
            devs[device]["ann_factor"] = ( 1 + invest_replacements - res_value) * CRF 

    return devs

## CODE SNIPPETS for FZJ data
    

#further parameters for topology optimization
#"inv_earth_work":              200,    # EUR/m, fix costs per meter length (earth work/construction)
#             "inv_material_var":            0.093,  # EUR/m/d # costs per meter length and diameter (pipe costs)
#             "inv_material_fix":            29.71,  # EUR/m # costs per meter length (pipe costs)
#             "number_of_balancing_units": 1,      # maximum number of nodes in which balancing units can be build
#    "T_supply_pipe": 18,           #
#             "T_return_pipe": 12,           # 
#             "c_P_water": 4.18,             # kJ/(kg K),    specific heat capacity for water
#             
             
      
#    nodes = transform_coordinates(nodes)
#    
#    # Load demands
#
#    ending = {"power": "_electricity.txt", "heat": "_heating.txt", "cool": "_cooling.txt"}
#    build = {}
#    path_load = {}
#    for n in range(len(nodes)):
#        build_name = nodes[n]["name"]
#        for com in nodes[n]["commodities"]:
#                path_load[com] = path_demand_files + build_name + ending[com]
#                build[com] = np.loadtxt(open(path_load[com], "rb"), delimiter=",", usecols=(0))
#            
#                # filter outlier (manually checked for the two buildings)
#                if build_name == "1520":
#                    for k in range(len(build[com])):
#                        if build[com][k] > 1000:
#                            build[com][k] = build[com][k-1] 
#                
#                if build_name == "1613":
#                    for k in range(len(build[com])):
#                        if build[com][k] > 1000:
#                            build[com][k] = build[com][k-1] 
#                            
#                nodes[n][com] = build[com]
#                print("Building " + build_name + " (" + com + ") loaded successfully. ")
        

    #%% Use case: FZJ (Forschungszentrum Juelich)
    
#    if use_case == "FZJ":
#        pass
        # path_weather_file = str(os.path.dirname(os.path.realpath(__file__)) + "\\input_data\\TRY2015_37335002675500_Jahr.csv")
#        path_demand_files = "input_data\\processed_data_2018-07-03_21-34-53\\"
        
#%% Use case: Simple test (Simple use case for experimenting based on data of FZJ)
    
#    if use_case == "simple_test":
#        
#        nodes[0] = {"lon":      1, # longitude
#                    "lat":      0, # latitude 
#                    "name":  "test_bldg_0",
#                    "commodities": ["heat", "cool", "power"],
#                    }
#        
#        nodes[1] = {"lon":      1, # longitude
#                    "lat":      1, # latitude 
#                    "name":  "test_bldg_1",
#                    "commodities": ["heat", "cool", "power"],
#                    }
#        
#        nodes[2] = {"lon":      0, # longitude
#                    "lat":      0, # latitude 
#                    "name":  "test_bldg_2",
#                    "commodities": ["heat", "cool", "power"],
#                    }
#        
#        nodes[3] = {"lon":      0, # longitude
#                    "lat":      1, # latitude 
#                    "name":  "test_bldg_3",
#                    "commodities": ["heat", "cool", "power"],
#                    } 
        
        #    else:
#            
#        nodes[0] = {"lon":      6.402940, # longitude
#                    "lat":      50.908491, # latitude 
#                    "bldg_id":  "1510",
#                    "commodities": ["heat", "cool", "power"],
#                    }
#                        
#        nodes[1] = {"lon":      6.403199, # longitude
#                    "lat":      50.909126, # latitude 
#                    "bldg_id":  "1560",
#                    "commodities": ["heat", "cool", "power"],
#                    }
#        
#        nodes[2] = {"lon":      6.402109, # longitude
#                    "lat":      50.908323, # latitude 
#                    "bldg_id":  "1570",
#                    "commodities": ["heat", "cool", "power"],
#                    }
#        
#        nodes[3] = {"lon":      6.401926, # longitude
#                    "lat":      50.908205, # latitude 
#                    "bldg_id":  "1580",
#                    "commodities": ["heat", "cool", "power"],
#                    }
        
    # 1513 auf 1580 drauf rechnen
        
#    nodes[4] = {"lon":      6.403670, # longitude
#                "lat":      50.907662, # latitude 
#                "bldg_id":  "1514",
#                "dem": {0: -1, 1: 2}
#                }
    
#    nodes[5] = {"lon":      6.403767, # longitude
#                "lat":      50.908444, # latitude 
#                "bldg_id":  "1522",
#                }
#    
#    nodes[6] = {"lon":      6.404016, # longitude
#                "lat":      50.908603, # latitude 
#                "bldg_id":  "1590",
#                }
#    
#    nodes[7] = {"lon":      6.404968, # longitude
#                "lat":      50.908360, # latitude 
#                "bldg_id":  "1520",
#                }
#    
#    nodes[8] = {"lon":      6.405162, # longitude
#                "lat":      50.908781, # latitude 
#                "bldg_id":  "1615",
#                }
#    
#    nodes[9] = {"lon":      6.406938, # longitude
#                "lat":      50.908541, # latitude 
#                "bldg_id":  "0410",
#                }
#    
#    nodes[10] = {"lon":     6.406624, # longitude
#                "lat":      50.909455, # latitude 
#                "bldg_id":  "1630",
#                }
#    
#    nodes[11] = {"lon":     6.405187, # longitude
#                "lat":      50.910320, # latitude 
#                "bldg_id":  "1613",
#                }
#    
#    nodes[12] = {"lon":     6.404774, # longitude
#                "lat":      50.910150, # latitude 
#                "bldg_id":  "1690",
#                }
#    
#    nodes[13] = {"lon":     6.404428, # longitude
#                "lat":      50.909751, # latitude 
#                "bldg_id":  "1660",
#                }
nodes, param, devs, time_steps = load_params("FZJ", str(os.path.dirname(os.path.realpath(__file__))))