# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 15:48:51 2018

@author: lkivi
"""

import numpy as np
import json
import pylab as plt
from shapely.geometry import Point


def generateJson():
    
    data_dict = {}
    
    path_nodes = "input_data/nodes.txt"  
    path_edges = "input_data/edges.txt"
    
    nodes = {}
       
    nodes["lat"] = np.loadtxt(open(path_nodes, "rb"), delimiter = ",", usecols=(0))*np.pi/180           # rad,      node latitudes
    nodes["long"] = np.loadtxt(open(path_nodes, "rb"), delimiter = ",", usecols=(1))*np.pi/180          # rad,      node longitudes
    nodes["type"] = np.genfromtxt(open(path_nodes, "rb"),dtype = 'str', delimiter = ",", usecols=(2))   # --,       node type
    
    # Earth radius
    r = 6371000
    
    # reference coordinates (x=0, y=0)
    lat_ref = np.amin(nodes["lat"])
    long_ref = np.amin(nodes["long"])
    
    # calculate xy-coordinates 
    nodes["x"] = r*np.arccos(np.sin(nodes["lat"])**2 + np.cos(nodes["lat"])**2 * np.cos(nodes["long"] - long_ref))
    nodes["y"] = r*np.arccos(np.sin(nodes["lat"])*np.sin(lat_ref) + np.cos(nodes["lat"])*np.cos(lat_ref))
    # replace nan entries by 0
    nodes["x"] = np.nan_to_num(nodes["x"])
    nodes["y"] = np.nan_to_num(nodes["y"])
    
    edges = {}
    edges["node_0"] = np.genfromtxt(open(path_edges, "rb"),dtype = None, delimiter = ",", usecols=(0))
    edges["node_1"] = np.genfromtxt(open(path_edges, "rb"),dtype = None, delimiter = ",", usecols=(1))
    
    
    nodes_list = []
    for i in range(np.size(nodes["x"])):
        nodes_list.append({"name": str(i), "x": nodes["x"][i], "y": nodes["y"][i], "type": str(nodes["type"][i])})
    
    edges_list = []
    for i in range(np.size(edges["node_0"])):
        distance = ((nodes["x"][edges["node_1"][i]] - nodes["x"][edges["node_0"][i]])**2 + (nodes["y"][edges["node_1"][i]] - nodes["y"][edges["node_0"][i]])**2)**0.5
        edges_list.append({"name": str(edges["node_0"][i]) + "-" + str(edges["node_1"][i]),
                           "node_0": int(edges["node_0"][i]),
                           "node_1": int(edges["node_1"][i]),
                           "distance": distance,
                           "diameter": 0.5})
    
    data_dict = {"nodes": nodes_list,
                 "edges": edges_list}
        
    
    with open("nodes.json", "w") as f: json.dump(data_dict, f, indent=4, sort_keys=True)
    
    data_dict = json.loads(open("nodes.json").read())
       
    
def plotGrid():
     
    data = json.loads(open("nodes.json").read())
 
    for item in data["nodes"]:
        if item["type"] == "supply":
            r = 8
            color = 'r'
            width = 1
        if item["type"] == "node":
            r = 3
            color = 'g'
            width = 0.5
        phi = np.arange(50)*2*np.pi/50
        x = r*np.cos(phi) + item["x"]
        y = r*np.sin(phi) + item["y"]
        plt.plot(x,y,'k', linewidth = width, zorder = 1)
        plt.fill(x,y,color, zorder = 10)
    
    for item in data["edges"]:
        plt.plot([data["nodes"][item["node_0"]]["x"],data["nodes"][item["node_1"]]["x"]],[data["nodes"][item["node_0"]]["y"],data["nodes"][item["node_1"]]["y"]], 'r', zorder = 5)
    
    plt.grid(zorder = 0)
    plt.show()


