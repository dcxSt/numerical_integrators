#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 15:16:59 2020

@author: steve
"""
# N BODY SYSTEM NO OBJECTS

# %% IMPORTS 
import numpy as np
import matplotlib.pyplot as plt

# %% CONSTANTS
G = 1 # gravitational constant

class System:
    G = 1
    H = 0.01
    STEPS = 10000 
    # canonical symplectic form
    J = np.array([[0,0,0,-1,0,0],
                   [0,0,0,0,-1,0],
                   [0,0,0,0,0,-1],
                   [1,0,0,0,0,0],
                   [0,1,0,0,0,0],
                   [0,0,1,0,0,0]])
    
    
    def __init__(self,Y0,masses,names):
        self.Y0 = Y0
        self.Y = Y0[:]
        self.masses = masses
        self.trace = None
        self.names = names
        
    # GETTERS
    
    def get_energy_planet(self,y):
        
    def get_energy(self)
    


# %% 

if __name__=='__main__':
    # initialize instance of system with stars, mass and name 
