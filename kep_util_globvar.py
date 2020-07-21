#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 12:09:39 2020

This script defines utility functions and global constants

It also defines first integral functions, the reason these are lumped together is because 
we need the first integral functions to define some of the global constants

The import statements are written here for ease of access
from kep_util_globvar import G,M1,M2,M,M_REDUCED,ETA,Y0,H,STEPS,NABLA_LIN_X,NABLA_LIN_Y,ENERGY_0,TOTAL_MOMENTUM_0,TOTAL_ANG_MOMENTUM_0,J,Jinv
from kep_util_globvar import normsq,supervec_dot,supervec_norm,to_supervec,k_factor
from kep_util_globvar import get_energy,get_total_angular_momentum,get_total_linear_momentum_abs,get_lin_mom_x,get_lin_mom_y

@author: steve
"""

import numpy as np
import math



# Initializes Global Variables and Constants 

# %% GLOBAL CONSTANTS

# contants
G = 1 # gravitational constant
M1 = 1.0
M2 = 1.4
M = M1+M2
M_REDUCED = M1*M1 / M
ETA = -G*M1*M2 
Y0 = np.array([np.array([0.4,0.0]),np.array([-0.4,-0.0]),
               np.array([0.,-1.0]),np.array([0.,1.0])])# cartesian (p1,p2,q1,q2)

H = 0.0001 # timestep 
STEPS = 50000


NABLA_LIN_X = np.array([np.array([1,0]),np.array([1,0]),np.array([0,0]),np.array([0,0])])
NABLA_LIN_Y = np.array([np.array([0,1]),np.array([0,1]),np.array([0,0]),np.array([0,0])])
# go into CM frame by setting net momentum to zero? 

J = np.array(((0,0,0,0, 1,0,0,0),
              (0,0,0,0,0, 1,0,0),
              (0,0,0,0,0,0, 1,0),
              (0,0,0,0,0,0,0, 1),
              (-1,0,0,0,0,0,0,0),
              (0,-1,0,0,0,0,0,0),
              (0,0,-1,0,0,0,0,0),
              (0,0,0,-1,0,0,0,0)))

Jinv = np.array(((0,0,0,0,-1,0,0,0),
                 (0,0,0,0,0,-1,0,0),
                 (0,0,0,0,0,0,-1,0),
                 (0,0,0,0,0,0,0,-1),
                 ( 1,0,0,0,0,0,0,0),
                 (0, 1,0,0,0,0,0,0),
                 (0,0, 1,0,0,0,0,0),
                 (0,0,0, 1,0,0,0,0)))


# %% GENERIC UTILITY FUNCTIONS

# helper function, calculates absolute value squared of vector 
def normsq(v): 
    return sum([i**2 for i in v]) 

# this function computes the dot products for my supervectors 
# each entry of the vector is it's self a vector, can think of this like a 2nd order field extension
def supervec_dot(u,v):
    return sum([np.dot(i,j) for i,j in zip(u,v)])

# same as supervec dot but takes one param and dots it with its self
def supervec_norm(u):
    return sum([np.dot(i,j) for i,j in zip(u,u)])

# takes streched format vector converts it back to supervec
def to_supervec(y):
    yvec = []
    for i in range(0,len(y),2):
        yvec.append(np.array(y[i:i+2]))
    return np.array(yvec)

# this function returns zero when the reduced orbit is circular
def k_factor(y):
    # compute the relative velocity squared
    v_squared = normsq(y[1] / M2 - y[0] / M1)
    r_abs = math.sqrt(normsq(y[3] - y[2]))
    return v_squared - G*M / r_abs

# %% FIRST INTEGRALS, functions that return scalars (field elements) 

# evaluates hamiltonian
def get_energy(y):
    r = np.array([y[2][0]-y[3][0] , y[2][1]-y[3][1]])#r from 1 to 0
    rabs = np.sqrt(normsq(r))
    p1sq,p2sq = normsq(y[0]),normsq(y[1])
    potential = - G*M1*M2 / rabs
    return 0.5*p1sq/M1 + 0.5*p2sq/M2 + potential 

# evaluates total angular momentum
def get_total_angular_momentum(y):
    # calculate it for each of the planets
    l1 = np.linalg.det(np.array([y[0],y[2]]))
    l2 = np.linalg.det(np.array([y[1],y[3]]))
    return l1 + l2 

# evaluates total linear momentum 
def get_total_linear_momentum_abs(y):
    # calculate the net linear momentum of the system (abs value) 
    return np.sqrt(normsq(y[0]+y[1]))

# evaluates linear mometnum in x direction
def get_lin_mom_x(y):
    return y[0][0] + y[1][0]

# evaluate linear momentum in y direction
def get_lin_mom_y(y):
    return y[0][1] + y[1][1]


# %% INITIALIZE GLOBAL CONSTANTS THAT REQUIRE FUNCTIONS 
ENERGY_0 = get_energy(Y0)
TOTAL_MOMENUM_0 = get_total_linear_momentum_abs(Y0)
TOTAL_ANG_MOMENTUM_0 = get_total_angular_momentum(Y0)    



