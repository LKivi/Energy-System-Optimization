# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 15:48:51 2018

@author: lkivi
"""

import numpy as np
import json
import pylab as plt
from pyproj import Proj, transform


#%%
# Calculate diameters for heating and cooling grid
def design_grid(param):
 
    # standard pipes (ISO group 1, series 1, thickness range D)
    # inner diameters
    diameters = [0.007, 0.0103, 0.014, 0.0177, 0.0233, 0.0297, 0.0378, 0.0437, 0.0557, 0.0709, 0.0831, 0.1079, 0.1325, 0.1603, 0.2101, 0.263, 0.3127, 0.3444, 0.3938, 0.4444, 0.4954]
    
    
    data = generateJson()
    dem = load_demands(data)    
    
    grid_styles = ["heating", "cooling"]
    
    for style in grid_styles:
        
        for edge in data["edges"]:
            supplied_buildings = list_supplied_buildings(data, edge)

            # sum up the demands of the buildings supplied by that edge        
            dem_buildings = np.zeros(8760)
            for building in supplied_buildings:
                 dem_buildings = dem_buildings + dem[style][building]
                 
            # find maximum value of load on the pipe
            pipe_load_max = np.max(dem_buildings)
            
            # calculate maximum mass flow in the pipe
            m_max = pipe_load_max*1e6/(param["c_p"]*(abs(param["T_"+style+"_supply"] - param["T_"+style+"_return"])))
            
            # calculate pipe diameter for given pressure gradient R
            d = ((8*m_max**2*param["f_fric"])/(param["rho"]*np.pi**2*param["dp_pipe"]))**0.2
            
            # choose next bigger diameter from list
            for d_norm in diameters:
                if d_norm >= d:
                    d = d_norm
                    break
     
            # write pipe diameter into json array
            edge["diameter_"+style] = d
            
        
    # save new json-file in project folder
    with open("nodes.json", "w") as f: json.dump(data, f, indent=4, sort_keys=True)
    
    return data


#%%
# generate json-file of the network using the input files nodes.txt and edges.txt
# pipe diameters are initialized with 0    
def generateJson():
    
    data_dict = {}
    
    path_nodes = "input_data/nodes.txt"     # contains node properties: latidude, longitude, name and type (supply, building, node)
    path_edges = "input_data/edges.txt"     #   
    
    nodes = {}
       
    nodes["lat"] = np.loadtxt(open(path_nodes, "rb"), delimiter = ",", usecols=(0))*np.pi/180           # rad,      node latitudes
    nodes["long"] = np.loadtxt(open(path_nodes, "rb"), delimiter = ",", usecols=(1))*np.pi/180          # rad,      node longitudes
    nodes["type"] = np.genfromtxt(open(path_nodes, "rb"),dtype = 'str', delimiter = ",", usecols=(2))   # --,       node type
    nodes["name"] = np.genfromtxt(open(path_nodes, "rb"),dtype = 'str', delimiter = ",", usecols=(3))   # --,       node name
    
    # Earth radius
    r = 6371000
    
    # supply node serves as reference node (x=0, y=0)
#    for i in np.arange(np.size(nodes["lat"])):
#        if nodes["type"][i] == "supply":
#            lat_ref = nodes["lat"][i]
#            long_ref = nodes["long"][i]
    
    # find minimal lat/long
    lat_ref = np.min(nodes["lat"])
    long_ref = np.min(nodes["long"])
    
    # transform lat/long to xy-coordinates 
    nodes["x"] = r*np.arccos(np.sin(nodes["lat"])**2 + np.cos(nodes["lat"])**2 * np.cos(nodes["long"] - long_ref))
    nodes["y"] = r*np.arccos(np.sin(nodes["lat"])*np.sin(lat_ref) + np.cos(nodes["lat"])*np.cos(lat_ref))
    # replace nan entries by 0
    nodes["x"] = np.nan_to_num(nodes["x"])
    nodes["y"] = np.nan_to_num(nodes["y"])
    
    # shift x/y-coordinates so that supply node is at x = 0, y = 0
    for i in np.arange(np.size(nodes["x"])):
        if nodes["type"][i] == "supply":
            supply_x = nodes["x"][i]
            supply_y = nodes["y"][i] 
    nodes["x"] = nodes["x"] - supply_x
    nodes["y"] = nodes["y"] - supply_y
    
    edges = {}
    edges["node_0"] = np.genfromtxt(open(path_edges, "rb"),dtype = 'str', delimiter = ",", usecols=(0))
    edges["node_1"] = np.genfromtxt(open(path_edges, "rb"),dtype = 'str', delimiter = ",", usecols=(1))
      
    nodes_list = []
    for i in range(np.size(nodes["x"])):
        nodes_list.append({"name": nodes["name"][i], "x": nodes["x"][i], "y": nodes["y"][i], "type": str(nodes["type"][i])})
    
    edges_list = []
    for i in range(np.size(edges["node_0"])):
        
        index_0 = np.where(nodes["name"] == edges["node_0"][i])[0][0]
        index_1 = np.where(nodes["name"] == edges["node_1"][i])[0][0]
        
        length = ((nodes["x"][index_1] - nodes["x"][index_0])**2 + (nodes["y"][index_1] - nodes["y"][index_0])**2)**0.5
        edges_list.append({"name": edges["node_0"][i] + "-" + edges["node_1"][i],
                           "node_0": edges["node_0"][i],
                           "node_1": edges["node_1"][i],
                           "length": length,
                           "diameter_heating": 0,
                           "diameter_cooling": 0})
    
    data_dict = {"nodes": nodes_list,
                 "edges": edges_list}
        
    # save json-file in project folder
    with open("nodes.json", "w") as f: json.dump(data_dict, f, indent=4, sort_keys=True)
    
    return data_dict
    
 
#%%    
def plotGrid():
 
    data = json.loads(open('nodes.json').read())
    
    for item in data["nodes"]:
        if item["type"] == "supply":
            r = 8
            color = 'r'
            width = 1
            order = 15
        if item["type"] == "node":
            r = 2
            color = 'k'
            width = 0.5
            order = 10
        if item["type"] == "building":
            r = 5
            color = 'g'
            width = 1
            order = 15
            
        phi = np.arange(50)*2*np.pi/50
        x = r*np.cos(phi) + item["x"]
        y = r*np.sin(phi) + item["y"]
        plt.plot(x,y,'k', linewidth = width, zorder = 1)
        plt.fill(x,y,color, zorder = order)
  
    
    for item in data["edges"]:
       
        x_0, y_0 = findXY(data,item["node_0"])
        x_1, y_1 = findXY(data,item["node_1"])
        
        plt.plot([x_0, x_1], [y_0, y_1], 'r', zorder = 5)
    
    plt.grid(zorder = 0)
    plt.show()


#%%
# finds x- and y-coordinate of a node out of json file by name
def findXY(data, name):
    
    found = 0
    
    for item in data["nodes"]:
        
        if item["name"] == name:
            x = item["x"]
            y = item["y"]
            found = 1
            
    if found == 0:
        print("Can't retrieve node coordinates to plot grid edges")
        exit()
        
    return x,y
   

#%% finds all buildings that are supplied by a specific edge
def list_supplied_buildings(data, edge):
    
    # initialize array of end points with end point of the input edge
    endings = [edge["node_1"]]
    
    # initialize list of buildings
    supplied_buildings = []
    
    for i in range(1000):
        
        # check if the found ending points are buildings; add the found buildings to the buildings-array
        for iEnding in range(np.size(endings)):
            nodeName = endings[iEnding]
            for item in data["nodes"]:
                if item["name"] == nodeName and item["type"] == "building":
                    supplied_buildings.append(nodeName)        
        
        # set end points to new start points
        starts = endings
         
        #reset ending nodes
        endings = []
        
        #find all edges beginning with any entry of starts and get their ending points
        for iStart in range(np.size(starts)):
            nodeName = starts[iStart]
        
            for item in data["edges"]:
                if item["node_0"] == nodeName:
                    endings.append(item["node_1"])
        
        # if no new edges are found, the buildings array is returned
        if endings == []:
            return supplied_buildings
 

#%% loads demand arrays
def load_demands(data):
    
    path_demands = "input_data/demands/"
    dem = {}
    dem["heating"] = {}
    dem["cooling"] = {}
    
    dem["heating"]["sum"] = np.zeros(8760)
    dem["cooling"]["sum"] = np.zeros(8760)
    
    # collect building names out of json-data
    buildings = []
    for item in data["nodes"]:
        if item["type"] == "building":
            buildings.append(item["name"])
    
    # get loads of each building and sum up 
    for name in buildings:
        dem["heating"][name] = np.loadtxt(open(path_demands + name + "_heating.txt", "rb"), delimiter = ",", usecols=(0))/1000      # MW,   heating load of building
        dem["heating"]["sum"] = dem["heating"]["sum"] + dem["heating"][name]
        dem["cooling"][name] = np.loadtxt(open(path_demands + name + "_cooling.txt", "rb"), delimiter = ",", usecols=(0))/1000      # MW,   cooling load of building
        dem["cooling"]["sum"] = dem["cooling"]["sum"] + dem["cooling"][name]
    
    
    return dem
        
    
    
        
        