# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 11:07:27 2018

@author: mwi
"""

import os
import parameters
import bldg_balancing
import design_network_topology
import design_balancing_unit
import post_processing
import datetime
import numpy as np

#%% Define paths
path_file               = str(os.path.dirname(os.path.realpath(__file__)))
dir_results             = path_file + "\\Results" + str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

# Choose use case
#use_case = "simple_test"
#use_case = "EON"
use_case = "FZJ"

#TODO: imeplement PV in optimization, and: normalize summands in objective function (two pre-optimizations necessary)
# maybe only cost optimization

# Load parameters
nodes, param, devs, time_steps = parameters.load_params(use_case, path_file)

# Calculate intra-balancing (within buildings)
#nodes = bldg_balancing.calc_residuals(nodes, param, time_steps, dir_results)


# Abschätzung für Bedarfs-Massenströme an Knoten (tatsächliche Werte werden später in bldg_balancing berechnet!!!)
for k in range(len(nodes)):
    nodes[k]["mass_flow"] = np.zeros(8760)
    for t in range(8760):
        nodes[k]["mass_flow"][t] = (nodes[k]["dem_heat"][t] - nodes[k]["dem_cool"][t])*1e3/(param["c_f"]*(param["T_hot"] - param["T_cold"]))

# Optimize network topology
design_network_topology.design_network(nodes, param, time_steps, dir_results)

# Calculate inter-balancing and design balancing unit
#design_balancing_unit.design_balancing_unit(nodes, devs, param, time_steps, dir_results)


# Post-processing
#print("\n-----Post-processing:-----")
#post_processing.calc_diversity_index(nodes, time_steps)
#post_processing.plot_residual_heat(dir_results)
#post_processing.plot_total_power_dem_bldgs(dir_results)
#post_processing.plot_power_dem_HP_EH_CC(dir_results)
#post_processing.plot_demands(nodes, dir_results)
#post_processing.plot_COP_HP_CC(param, dir_results)

#        post_processing.plot_ordered_load_curve(heat_dem_sort, hp_capacity, eh_capacity, param, nodes[n]["name"], time_steps, dir_results)
#        post_processing.plot_bldg_balancing(nodes[n], time_steps, param, dir_results)
#        post_processing.save_balancing_results(nodes[n], time_steps, dir_results)