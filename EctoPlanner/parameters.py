# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 15:12:14 2018

@author: mwi
"""


from pyproj import Proj, transform
import math
import numpy as np

def load_params(use_case, path_file):
    
    assert (use_case != "FZJ" or use_case != "EON" or use_case != "simple_test"), "Use case '" + use_case + "' not known."
    path_demand_files = path_file + "\\input_data\\" + use_case + "\\"
    print("Using data set: '" + use_case + "'")
    time_steps = range(8760)
    
    nodes = {}       
    
#%% Use case: EON (example data provided by E.ON)
    if use_case == "EON":
        
        air_temp = np.loadtxt(path_demand_files + "air_temp.txt")
        
        flow_temp = - (0.7471 * air_temp) + 54.75 # source: ectoplanner excel sheet by E.ON
        # Vorlauftemperatur 
        outlet_temp_cond = flow_temp
        inlet_temp_evap = get_grid_temp_hot_pipe("two_point_control", time_steps)
        COP_carnot = np.abs((outlet_temp_cond+273.15)/(outlet_temp_cond-inlet_temp_evap))
        COP_HP = 3.3829 * np.log(COP_carnot) - 2.8708
        print("WARNING: USING WRONG COP DEFINITION FOR TESTING!")
        COP_CC = COP_HP - 1

        nodes[0] = {"lon":               6.402940, # °,         longitude
                    "lat":              50.908491, # °,         latitude
                    "name":              "bldg_0", # ---,       building name is assigned automatically
                    "heated_area":           1000, # m2,        area that is heated
                    "spec_heat_dem":          100, # kWh/m2,    specific demand (describes to building standard)
                    "is_cooled":             True, # ---,      defines if the building is cooled in summer
                    "bldg_type":    "residential", # either "residential" or "office"
                    }
        
        nodes[0]["heat"] = np.loadtxt(open(path_demand_files + nodes[0]["name"] + "_heating.txt", "rb"), delimiter=",", usecols=(0)) 
        # if a time series is given the attributes "heated_area", "spec_heat_dem", "is_cooled" and "residential" are ignored
        nodes[0]["cool"] = np.loadtxt(open(path_demand_files + nodes[0]["name"] + "_cooling.txt", "rb"), delimiter=",", usecols=(0))

        nodes[1] = {"lon":               6.403199, # °,         longitude
                    "lat":              50.909126, # °,         latitude
                    "name":              "bldg_1", # ---,       building name is assigned automatically
                    "heated_area":              0, # m2,        area that is heated
                    "spec_heat_dem":            0, # kWh/m2,    specific demand (describes to building standard)
                    "is_cooled":             True, # ---,      defines if the building is cooled in summer
                    "bldg_type":               "", # either "residential" or "office"
                    }
        nodes[1]["heat"] = np.loadtxt(open(path_demand_files + nodes[1]["name"] + "_heating.txt", "rb"), delimiter=",", usecols=(0))
        nodes[1]["cool"] = np.loadtxt(open(path_demand_files + nodes[1]["name"] + "_cooling.txt", "rb"), delimiter=",", usecols=(0))

#        nodes = transform_coordinates(nodes)

    param = {"interest_rate":  0.05,        # ---,          interest rate
             "observation_time": 20.0,      # a,            project lifetime
             "price_gas": 0.0435,           # kEUR/MWh,     natural gas price
             "price_el": 0.106,             # kEUR/MWh,     electricity price (grid)
             "revenue_feed_in": 0.055,      # kEUR/MWh,     feed-in tariff (electricity)
             "gas_CO2_emission": 0.2,       # t_CO2/MWh,    specific CO2 emissions (natural gas)
             "grid_CO2_emission": 0.657,    # t_CO2/MWh,    specific CO2 emissions (chinese electricity grid mix)
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

    param["COP_HP"] = COP_HP
    param["COP_CC"] = COP_CC
    
    #%% LOAD DEVICE PARAMETER
    
    devs = {}

    #%% BOILER
    devs["BOI"] = {"inv_var": 52,       # kEUR/MW_th,       variable investment
                   "eta_th": 0.95,      # ---,              thermal efficiency
                   "life_time": 30,     # a,                operation time
                   "cost_om": 0.01,     # ---,              annual operation and maintenance costs as share of investment
                   }

    #%% CHP - INTERNAL COMBUSTION ENGINE
    devs["CHP"] = {"inv_var": 570,      # kEUR/MW_el,       variable investment
                   "eta_el": 0.35,      # ---,              electrical efficiency
                   "eta_th": 0.5,       # ---,              thermal efficiency
                   "life_time": 30,     # a,                operation time
                   "cost_om": 0.05,     # ---,              annual operation and maintenance costs as share of investment
                   }
    
    #%% ELECTRICAL HEATER
    devs["EH"] = {"inv_var": 78,        # kEUR/MW_th,       variable investment
                  "eta_th": 0.9,        # ---,              thermal efficiency
                  "life_time": 20,      # a,                operation time
                  "cost_om": 0.01,      # ---,              annual operation and maintenance costs as share of investment
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
        
    #%% BATTERY
    devs["BAT"] = {"inv_var": 520,      # kEUR/MWh_el,      variable investment
                   "max_cap": 50,       # MWh_el,           maximum eletrical storage capacity
                   "sto_loss": 0,       # 1/h,              standby losses over one time step
                   "eta_ch": 0.9592,    # ---,              charging efficiency
                   "eta_dch": 0.9592,   # ---,              discharging efficiency
                   "soc_init": 0.8,     # ---,              maximum initial relative state of charge
                   "soc_max": 0.8,        # ---,              maximum relative state of charge
                   "soc_min": 0.2,        # ---,              minimum relative state of charge
                   "life_time": 10,     # a,                operation time
                   "cost_om": 0.02,     # ---,              annual operation and maintenance costs as share of investment
                   }

    #%% THERMAL ENERGY STORAGE
    devs["TES"] = {"inv_var": 11.7,     # kEUR/MWh_th,      variable investment
                   "max_cap": 5000,     # MWh_th,           maximum thermal storage capacity
                   "sto_loss": 0.005,   # 1/h,              standby losses over one time step
                   "eta_ch": 0.975,     # ---,              charging efficiency
                   "eta_dch": 0.975,    # ---,              discharging efficiency
                   "soc_init": 0.8,     # ---,              maximum initial state of charge
                   "soc_max": 1,        # ---,              maximum state of charge
                   "soc_min": 0,        # ---,              minimum state of charge
                   "life_time": 20,     # a,                operation time
                   "cost_om": 0.01,     # ---,              annual operation and maintenance costs as share of investment
                   }

    #%% COLD THERMAL ENERGY STORAGE
    devs["CTES"] = {"inv_var": 11.7,    # kEUR/MWh_th,      variable investment
                    "max_cap": 5000,      # MWh_th,           maximum thermal storage capacity
                    "sto_loss": 0.005,  # 1/h,              standby losses over one time step
                    "eta_ch": 0.975,    # ---,              charging efficiency
                    "eta_dch": 0.975,   # ---,              discharging efficiency
                    "soc_init": 0.8,    # ---,              maximum initial state of charge
                    "soc_max": 1,       # ---,              maximum state of charge
                    "soc_min": 0,       # ---,              minimum state of charge
                    "life_time": 20,    # a,                operation time
                    "cost_om": 0.01,    # ---,              annual operation and maintenance costs as share of investment
                    }
    
    #%% CONNECTION TO DISTRICT COOLING NETWORK
    devs["from_DC"] = {"inv_var": 11.7,    # kEUR/MW_th,      variable investment
                       "max_cap": 5000,    # MW_th,           maximum thermal storage capacity
                       "min_cap": 0,       # MW_th,           minimum thermal storage capacity              
                       "eta_th": 0.99,     # ---,             discharging efficiency
                       "life_time": 50,    # a,               operation time
                       "cost_om": 0.01,    # ---,             annual operation and maintenance costs as share of investment
                       }

    #%% CONNECTION TO DISTRICT HEATING NETWORK
    devs["from_DH"] = {"inv_var": 11.7,    # kEUR/MW_th,      variable investment
                       "max_cap": 5000,    # MW_th,           maximum thermal storage capacity
                       "min_cap": 0,       # MW_th,           minimum thermal storage capacity              
                       "eta_th": 0.99,     # ---,             discharging efficiency
                       "life_time": 50,    # a,               operation time
                       "cost_om": 0.01,    # ---,             annual operation and maintenance costs as share of investment
                       }

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

def calc_pipe_costs(nodes, edges, edge_dict_rev, param):
    """
    Calculate variable and fix costs for every edge.
    """
    c_fix = {}
    c_var = {}
    for e in edges:
        x1, y1 = nodes[edge_dict_rev[e][0]]["x"], nodes[edge_dict_rev[e][0]]["y"]
        x2, y2 = nodes[edge_dict_rev[e][1]]["x"], nodes[edge_dict_rev[e][1]]["y"]
        c_fix[e] = (param["inv_earth_work"] + param["inv_material_fix"]) * math.sqrt((x1-x2)**2 + (y1-y2)**2)
        c_var[e] = param["inv_material_var"]
    
    print("Mindestkapazitaet vorsehen fuer Rohre")
    param["c_fix"] = c_fix
    param["c_var"] = c_var
    return param

def get_edge_dict(n):
    compl_graph = nx.complete_graph(n)
    edge_list = list(compl_graph.edges(data=False))
    edge_dict = {(edge_list[k][0], edge_list[k][1]): k for k in range(len(edge_list))}
    edge_dict_rev = {k: (edge_list[k][0], edge_list[k][1]) for k in range(len(edge_list))}
    edges = range(len(edge_list))
    return edge_dict, edge_dict_rev, edges, compl_graph

def transform_coordinates(nodes):
    outProj = Proj(init='epsg:25832')   # ETRS89 / UTM zone 32N
    inProj = Proj(init='epsg:4258')     # Geographic coordinate system: EPSG 4326
    for n in range(len(nodes)):
        nodes[n]["x"],nodes[n]["y"] = transform(inProj,outProj,nodes[n]["lon"],nodes[n]["lat"])
        
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

    for device in devs.keys():

#        for TONGLI area:
        devs[device]["inv_fix"] = 0
        
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
    