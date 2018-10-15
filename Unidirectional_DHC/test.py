# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 22:46:17 2018

@author: lkivi
"""
import cmath
import numpy as np
import grid

data = grid.generateJson()

buildings = {}

for item in data["edges"]:
    buildings[item["name"]] = grid.listBuildings(data, item)
    
print(buildings)