# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 20:28:12 2018

@author: mwi
"""

import gurobipy as gp
import matplotlib.pyplot as plt
import networkx as nx
import load_params
import post_processing

def design_network(nodes, param, time_steps, dir_results):
    
    print("HIER MUESSEN NOCH PUMPARBEITEN BERUECKSICHTIGT WERDEN in der ZielFn, sonst hat man keinen vernuenftigen Trade-Off zwischen Invest und Betrieb.")

    # Create a new model
    model = gp.Model("Ectogrid_topology")
    
        
    #%% CREATE VARIABLES
    
    edge_dict, edge_dict_rev, edges, compl_graph = load_params.get_edge_dict(len(nodes))
    param = load_params.calc_pipe_costs(nodes, edges, edge_dict_rev, param)
    node_list = range(len(nodes))
    
    x = model.addVars(edges, vtype="B", name="x") # Purchase decision binary variables (1 if device is installed, 0 otherwise)
        
    ksi = model.addVars(node_list, vtype="B", name="balancing_unit") # Binary decision: balancing unit installed in node
    
    cap = model.addVars(edges, vtype="C", name="nominal_edge_capacity") # Mass flow capacity of edge
     
    m_dot = model.addVars(edges, time_steps, vtype="C", lb=-100, name="mass_flow_pipe") # Mass flow in pipe in every time step (from 0->1 positive)
    
    m_bal = model.addVars(node_list, time_steps, vtype="C", lb=-100, name="mass_flow_balancing") # Mass flow from warm supply pipe to cold return pipe
    
    obj = model.addVar(vtype="C", name="total_costs") # Objective function
        
    
    #%% DEFINE OBJECTIVE FUNCTION
    model.update()
    model.setObjective(sum(x[edge] * param["c_fix"][edge] + cap[edge] * param["diam_per_cap"] * param["c_var"][edge] for edge in edges), gp.GRB.MINIMIZE)
    
    #%% CONSTRAINTS
    # Node balance
    list_edge_id_used = []
    
    for node in node_list:
        adjacent_edges = list(compl_graph.edges(node, data=False))
        ids_plus_sign = []
        ids_minus_sign = []
        for e in adjacent_edges:
            if e[0] > e[1]:
                e = (e[1], e[0])
            edge_id = edge_dict[e]
            if edge_id not in list_edge_id_used:
                ids_plus_sign.append(edge_id)
                list_edge_id_used.append(edge_id)
            else:
                ids_minus_sign.append(edge_id)
            
        for t in time_steps:
            model.addConstr(sum(m_dot[k,t] for k in ids_plus_sign) - sum(m_dot[k,t] for k in ids_minus_sign) 
                            + nodes[node]["mass_flow"][t] + m_bal[node,t] == 0)
            
    # Maximum number of balancing units
    model.addConstr(sum(ksi[node] for node in node_list) <= param["number_of_balancing_units"], "number_balancing_units")
    
    # Balancing power is only at balancing nodes possible
    for node in node_list:
        for t in time_steps:
            model.addConstr(m_bal[node,t] <= ksi[node] * 1000)
            model.addConstr(-m_bal[node,t] <= ksi[node] * 1000)
            
    # Help constraint
    model.addConstr(ksi[0] == 1)
        
    # Mass flow on edge must not exceed pipe capacity   
    for edge in edges:
        for t in time_steps:
            model.addConstr(m_dot[edge,t] <= cap[edge])
            model.addConstr(-m_dot[edge,t] <= cap[edge])
            model.addConstr(m_dot[edge,t] <= x[edge] * 1000)
            model.addConstr(-m_dot[edge,t] <= x[edge] * 1000)
    
    #%% RUN OPTIMIZATION
            
    model.optimize()
    
    
    #%% EVALUATE RESULTS
    
    if model.Status in (3,4) or model.SolCount == 0:  # "INFEASIBLE" or "INF_OR_UNBD"
            model.computeIIS()
            model.write("model.ilp")
            print('Optimization result: No feasible solution found.')
        
    else:
        model.write("model.lp")
        model.write("model.prm")
        model.write("model.sol")
        
        print("Optimization done.\n")
        
    #%% Create plots and result files
    print("Use tuples as index of edges")
    post_processing.save_network_data(nodes, cap)
    post_processing.plot_network(nodes, cap)