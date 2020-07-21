#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 12:04:44 2020

Here are stored functions which return derivatives of certain values
things like Jacobians, Hessians, Flows and Nablas are defined here

import statement for ease of reference:
from kep_derivative_functions import nabla_lin_y,nabla_lin_x,nabla_l,nabla_q,nabla_H
from kep_derivative_functions import partial_Hqqq,partial_Hpp,partial_Hqq,hessian_H
from kep_derivative_functions import gradient,modified_gradient_en_ang_attractor,modified_gradient_energy_attractor

@author: steve
"""

import numpy as np
import math
from kep_util_globvar import G,M1,M2,ETA,ENERGY_0,TOTAL_ANG_MOMENTUM_0
from kep_util_globvar import normsq,supervec_norm
from kep_util_globvar import get_energy,get_total_angular_momentum


# %% GRADIANT FUNCTIONS, GRADIENTS OF FIRST INTEGRALS (NABLA)

# returns y' = J^{-1}\nabla H given y
# y is [p1,p2,q1,q2]
def gradient(y):
    # assume 2 body problem
    # here gradient is a bit of misnomer, it's the J^{-1}\nabla H that I label grad y
    r = np.array([y[2][0]-y[3][0] , y[2][1]-y[3][1]])#r from 1 to 0
    rabs = np.sqrt(normsq(r))
    grady = []
    grady.append(-r*G*M1*M2*(rabs**(-3)))
    grady.append(r*G*M1*M2*(rabs**(-3)))
    grady.append(y[0]/M1)
    grady.append(y[1]/M2)
    return np.array(grady)

# uses ENERGY_0 turns the energy manifold into an attractor
def modified_gradient_energy_attractor(y):
    r = np.array([y[2][0]-y[3][0] , y[2][1]-y[3][1]])#r from 1 to 0
    rabs = np.sqrt(normsq(r))
    grady = []
    grady.append(-r*G*M1*M2*(rabs**(-3)))
    grady.append(r*G*M1*M2*(rabs**(-3)))
    grady.append(y[0]/M1)
    grady.append(y[1]/M2) 
    grady = np.array(grady)
    # an attracive term is added along nabla H
    energy_y = get_energy(y)
    attracting_term = nabla_H(y) 
    # attracting_term = attracting_term / supervec_norm(attracting_term)
    attracting_term *= (ENERGY_0 - energy_y) * math.exp(ENERGY_0 - energy_y) /5
    
    return grady + attracting_term 

# additionall attracting term to the ENERGY_0 and ANGULAR_M_0 manifolds
def modified_gradient_en_ang_attractor(y):
    r = np.array([y[2][0]-y[3][0] , y[2][1]-y[3][1]])#r from 1 to 0
    rabs = np.sqrt(normsq(r))
    grady = []
    grady.append(-r*G*M1*M2*(rabs**(-3)))
    grady.append(r*G*M1*M2*(rabs**(-3)))
    grady.append(y[0]/M1)
    grady.append(y[1]/M2) 
    grady = np.array(grady)
    n_grady = math.sqrt(supervec_norm(grady))
    # an attractive term as in the naive approach, first energy
    energy_y = get_energy(y)
    energy_attracting_term = nabla_H(y)
    energy_attracting_term *= (ENERGY_0 - energy_y) * math.exp(ENERGY_0 - energy_y)
    e_att_n = np.sqrt(supervec_norm(energy_attracting_term))
    # make the attracting term smaller than the initial term (1/10th at most)
    if e_att_n * 10 > n_grady:
        energy_attracting_term *= n_grady / (10 * e_att_n) 
    # then angular momentum
    ang_y = get_total_angular_momentum(y)
    ang_attracting_term = nabla_l(y)
    ang_attracting_term *= (TOTAL_ANG_MOMENTUM_0 - ang_y) * math.exp(TOTAL_ANG_MOMENTUM_0 - ang_y)
    ang_att_n = np.sqrt(supervec_norm(energy_attracting_term))
    # make the attracting term smaller if it's too big
    if ang_att_n * 10 > n_grady:
        ang_attracting_term *= n_grady / (10 * e_att_n)
    
    return grady + energy_attracting_term + ang_attracting_term 
    
    
    

# returns the gradient of the hamiltonian
def nabla_H(y):
    r = np.array([y[2][0]-y[3][0] , y[2][1]-y[3][1]])# r from 1 to 0 
    rabs = np.sqrt(normsq(r))
    nabla_h = []
    nabla_h.append(y[0]/M1)
    nabla_h.append(y[1]/M2)
    nabla_h.append(r*G*M1*M2*(rabs**(-3)))
    nabla_h.append(-r*G*M1*M2*(rabs**(-3)))
    return np.array(nabla_h)


# returns the hessian matrix of the hamiltonian at y
def hessian_H(y):
    # see section 3.1 kepler problem in report write-up
    # compute xi1 and xi2
    xi = math.sqrt(np.dot(y[2]-y[3],y[2]-y[3]))
    xito3 = xi**3
    xito5 = xi**5
    xi1 = y[3][0] - y[2][0]
    xi2 = y[3][1] - y[2][1]
    # we define each column manually (also symmetric so columns = rows), it's only eight by eight
    col1 = (1/M1,0,0,0,0,0,0,0)
    col2 = (0,1/M1,0,0,0,0,0,0)
    col3 = (0,0,1/M2,0,0,0,0,0)
    col4 = (0,0,0,1/M2,0,0,0,0)
    col5 = (0,0,0,0, -ETA/xito3 + ETA/xito5*xi1**2 , ETA/xito5*xi1*xi2 , ETA/xito3 - ETA/xito5*xi1**2 , -ETA/xito5*xi1*xi2)
    col6 = (0,0,0,0, ETA/xito5*xi1*xi2 , -ETA/xito3 + ETA/xito5*xi2**2 , -ETA/xito5*xi1*xi2 , ETA/xito3 - ETA/xito5*xi2**2)
    col7 = (0,0,0,0, ETA/xito3 - ETA/xito5*xi1**2 , -ETA/xito5*xi1*xi2 , -ETA/xito3 + ETA/xito5*xi1**2 , ETA/xito5*xi1*xi2)
    col8 = (0,0,0,0, -ETA/xito5*xi1*xi2 , ETA/xito3 - ETA/xito5*xi2**2 , ETA/xito5*xi1*xi2 , -ETA/xito3 + ETA/xito5*xi2**2)
    return np.array((col1,col2,col3,col4,col5,col6,col7,col8))
    


# returns 4x4 matrix, lower right hand block of hessian
def partial_Hqq(y):
    # see section 3.1 kepler problem in report write-up
    # compute xi1 and xi2
    xi = math.sqrt(np.dot(y[2]-y[3],y[2]-y[3]))
    xito3 = xi**3
    xito5 = xi**5
    xi1 = y[3][0] - y[2][0]
    xi2 = y[3][1] - y[2][1]
    # we define each column manually (also symmetric so columns = rows), it's only four by four
    col1 = (-ETA/xito3 + ETA/xito5*xi1**2 , ETA/xito5*xi1*xi2 , ETA/xito3 - ETA/xito5*xi1**2 , -ETA/xito5*xi1*xi2)
    col2 = (ETA/xito5*xi1*xi2 , -ETA/xito3 + ETA/xito5*xi2**2 , -ETA/xito5*xi1*xi2 , ETA/xito3 - ETA/xito5*xi2**2)
    col3 = (ETA/xito3 - ETA/xito5*xi1**2 , -ETA/xito5*xi1*xi2 , -ETA/xito3 + ETA/xito5*xi1**2 , ETA/xito5*xi1*xi2)
    col4 = (-ETA/xito5*xi1*xi2 , ETA/xito3 - ETA/xito5*xi2**2 , ETA/xito5*xi1*xi2 , -ETA/xito3 + ETA/xito5*xi2**2)
    return np.array((col1,col2,col3,col4))

# returns 4x4 matrix, upper right hand block of hessian
def partial_Hpp(y):
    col1 = (1/M1,0,0,0) 
    col2 = (0,1/M1,0,0) 
    col3 = (0,0,1/M2,0) 
    col4 = (0,0,0,1/M2) 
    return np.array((col1,col2,col3,col4)) 

# returns 4x4x4 tensor
def partial_Hqqq(y):
    return

# takes the gradient of hamiltonian only wrt q1 and q2
def nabla_q(q):
    # assume 2bp
    # takes q returns it's gradient, which is negative the derivative of p
    r = np.array([q[0][0]-q[1][0] , q[0][1]-q[1][1]])#r from 1 to 0
    rabs = np.sqrt(normsq(r))
    nablaq = []
    nablaq.append(-r*G*M1*M2*(rabs**(-3)))# vector
    nablaq.append(r*G*M1*M2*(rabs**(-3)))
    return nablaq

# i think this one is more correct - angular momentum 
def nabla_l(y):
    p1,p2,q1,q2 = y[0],y[1],y[2],y[3]
    nablal = []
    nablal.append(np.array([q1[1] , -q1[0]])) # partial p1 
    nablal.append(np.array([q2[1] , -q2[0]])) 
    nablal.append(np.array([ -p1[1] , p1[0]])) 
    nablal.append(np.array([ -p2[1] , p2[0]])) 
    return np.array(nablal)

# gradient of the total linear momentum in the x direction
# the level sets of this first integral are hyper-planes, so quite easy to figure out
def nabla_lin_x(y):
    return np.array([np.array([1,0]),np.array([1,0]),np.array([0,0]),np.array([0,0])])

# gradient of the total linear momentum in the y direction
def nabla_lin_y(y):
    return np.array([np.array([0,1]),np.array([0,1]),np.array([0,0]),np.array([0,0])])

