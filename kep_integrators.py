#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 12:04:45 2020

import statement for ease of access

@author: steve
"""

import numpy as np
from kep_util_globvar import M1,M2,H,J
from kep_derivative_functions import nabla_q,nabla_H,nabla_l
from kep_derivative_functions import partial_Hpp,partial_Hqq,hessian_H,hessian_l
from kep_derivative_functions import gradient,modified_gradient_en_ang_attractor,modified_gradient_energy_attractor



# %% BASIC NUMERICAL FLOW FUNCTIONS
    
# advances solution forward one timestep using the explicit euler method
def exp_euler(y):
    # calculate the gradient
    grady = gradient(y) 
    y_next = y + H*grady 
    return y_next 

# the gradient is modified so that the energy manifold is an attractor
def exp_euler_modified_energy_attractor(y):
    grady_mod = modified_gradient_energy_attractor(y)
    y_next = y + H*grady_mod
    return y_next

# the gradient is modified so that the energy and angular momentum manifolds are attractors
def exp_euler_modified_energy_ang_attractor(y):
    grady_mod = modified_gradient_en_ang_attractor(y) 
    y_next = y + H*grady_mod
    return y_next 

# advances solution forward one timestep using Stromer-Verlet scheme (p8)
def stromer_verlet(y): 
    grady = gradient(y) 
    # STEP 1 : \dot q_{n+1/2} = \dot q_n + h/2 * \dot p_n/m
    p1plushalf = y[0] + 0.5*grady[0]*H
    p2plushalf = y[1] + 0.5*grady[1]*H
    # STEP2 : q_{n+1} = q_n + h\dot q_{n+1/2}
    q1_next = y[2] + H*p1plushalf/M1
    q2_next = y[3] + H*p2plushalf/M2
    # STEP 3 : p_{n+1} = p_{n+1/2} - h/2 \frac{\partial H(p,q)}{\partial q}
    nablaq_next = nabla_q([q1_next,q2_next])
    p1_next = p1plushalf + 0.5*H* nablaq_next[0]
    p2_next = p2plushalf + 0.5*H* nablaq_next[1]
    y_next = np.array([p1_next,p2_next,q1_next,q2_next])
    
    return y_next

# advances solution one timestep using explicit trapezium rule (runge kutta) p28
def exp_trapezium(y):
    k1 = gradient(y)
    k2 = gradient(y + H*k1)
    y_next = y + 0.5*H*(k1+k2)
    return y_next

# advances solution one timestep using explicit midpoint rule (runge kutta) p28
def exp_midpoint(y):
    k1 = gradient(y)
    k2 = gradient(y + H*k1*0.5)
    y_next = y + H*k2
    return y_next

# with u=(p1,p2) and v=(q1,q2) we use syplectic euler method, easy because
# u' = a(v) and v' = b(u) , single variable dependence
def syplectic_euler(y):
    grady = gradient(y)
    u,v,u_prime = y[:2],y[2:],grady[:2]
    # first advance u_n to u_n+1 by explicit euler
    u_next = u + H*u_prime
    # then advance v_n to v_n+1 by implicit euler
    grady_2 = gradient(np.array([u_next[0],u_next[1],v[0],v[1]]))
    v_next_prime = grady_2[2:]
    v_next = v + H*v_next_prime
    y_next = np.concatenate([u_next,v_next])
    return y_next

def fourth_order_kutta(y):
    # equation 1.8 left p30 - numerical methods book 
    k1 = gradient(y)
    k2 = gradient(y + 0.5*H*k1)
    k3 = gradient(y + 0.5*H*k2)
    k4 = gradient(y + H*k3)
    y_next = y + H/6*(k1 + 2*k2 + 2*k3 + k4) 
    return y_next 


# %% UTILITY FUNCTIONS FOR MODIFIED EQ INTEGRATORS

# returns the scalar lambda_2 as defined in subsection 'Modified Equations for Numerical Flow of projection methods'
# in section 'Backward Error Analysis' (5) of the report
def lambda2(y,J,hessian_beta,nabla_beta,nabla_hamil,d2):
    hess_b = hessian_beta(y)
    nab_b = nabla_beta(y).flatten()
    nab_H = nabla_hamil(y).flatten()
    jinv_nab_H = -J @ nab_H
    
    first_term = 0.5 * np.dot(hess_b @ jinv_nab_H , jinv_nab_H)
    second_term = np.dot(nab_b , d2(y))
    return (first_term + second_term) / np.dot(nab_b,nab_b)

def lambda3(y,J,threetensor_beta,hessian_beta,nabla_hamil,nabla_beta,d3,d2):
    return

# returns term in the equation that comes from energy projection
def get_o2_projection_term_energy(y,d2):
    lam2 = lambda2(y,J,hessian_beta = hessian_H,
                   nabla_beta = nabla_H, nabla_hamil = nabla_H,
                   d2 = d2) 
    return -nabla_H(y).flatten() * lam2 
    
# returns term in the equation that comes from angular momentum projection 
def get_o2_projection_term_ang_mom(y,d2):
    lam2 = lambda2(y,J,hessian_beta = hessian_l,
                   nabla_beta = nabla_l, nabla_hamil = nabla_H,
                   d2 = d2)
    return -nabla_l(y).flatten() * lam2 


# %% NUMERICAL FLOW FUNCTIONS
# helper function computes second term of numerical flow
def d2_exp_euler(y):
    # this one is trivial
    return np.array((0,0,0,0,0,0,0,0))

def d3_exp_euler(y):
    return np.array((0,0,0,0,0,0,0,0))

# NOT YET WRITTEN
def d2_stromer_verlet(y):
    nab_H = nabla_H(y).flatten() #p11,p12,p21,p22,q11,q12,q21,q22 
    ham_p,ham_q = nab_H[:4],nab_H[4:] 
    ham_pp,ham_qq = partial_Hpp(y),partial_Hqq(y)
    return - 0.5 * np.concatenate([ham_qq @ ham_p , ham_pp @ ham_q])

def d3_stromer_verlet(y):
    return

# %% MODIFIED GRAIENT FUNCTIONS
def mod_flow_o2_no_proj(y,d2,h,flattened=True):
    f = gradient(y)
    nab_H = nabla_H(y).flatten() #p11,p12,p21,p22,q11,q12,q21,q22 
    ham_p,ham_q = nab_H[:4],nab_H[4:] 
    ham_pp,ham_qq = partial_Hpp(y),partial_Hqq(y) 
    # construct the second order term
    f2 = 0.5 * np.concatenate([ham_qq @ ham_p , ham_pp @ ham_q]) + d2(y).flatten()
    if flattened==True:
        return f + h*f2 # (warning, no reformatting here, still flat)
    m1 = f + h*f2 
    return np.array((m1[0:2],m1[2:4],m1[4:6],m1[6:])) 

def mod_flow_o2_energy_proj(y,d2,h):
    m1 = mod_flow_o2_no_proj(y,d2,h)
    proj_term = get_o2_projection_term_energy(y,d2) # add the energy projection term 
    m2 = m1 + h*proj_term 
    m2 = np.array((m2[0:2],m2[2:4],m2[4:6],m2[6:]))
    return m2

def mod_flow_o2_ang_proj(y,d2,h):
    m1 = mod_flow_o2_no_proj(y,d2,h)
    proj_term = get_o2_projection_term_ang_mom(y,d2)
    m2 = m1 + h*proj_term
    m2 = np.array((m2[0:2],m2[2:4],m2[4:6],m2[6:]))
    return m2

def mod_flow_o2_energy_and_ang_proj(y,d2,h):
    m1 = mod_flow_o2_no_proj(y,d2,h)
    proj_term_energy = get_o2_projection_term_energy(y,d2)
    proj_term_ang_mom = get_o2_projection_term_ang_mom(y,d2)
    m2 = m1 + h*(proj_term_energy + proj_term_ang_mom)
    m2 = np.array((m2[0:2],m2[2:4],m2[4:6],m2[6:]))
    return m2

# %% EXPLICIT EULER (SMALL STEP) NUMERICAL FLOW FUNCTIONS FOR MODIFIED EQUATIONS
# the equations are modified: the standard as well as projection terms added

def fourth_order_kutta_modified_flow(y,d2=d2_exp_euler,h=0.01,flow=mod_flow_o2_no_proj):
    k1 = flow(y)
    k2 = flow(y + 0.5*H*k1) 
    k3 = flow(y + 0.5*H*k2)
    k4 = flow(y + H*k3) 
    y_next = y + H/6*(k1 + 2*k2 + 2*k3 + k4)
    return y_next

def slow_modified_flow_basic_order2_no_proj(y,d2=d2_exp_euler,h=0.01):
    # h should be roughly 1 or 2 orders of magnitude bigger thagradientn H
    # just returns the point, times H times the modified gradient 
    f = gradient(y)
    nab_H = nabla_H(y).flatten() #p11,p12,p21,p22,q11,q12,q21,q22 
    ham_p,ham_q = nab_H[:4],nab_H[4:] 
    ham_pp,ham_qq = partial_Hpp(y),partial_Hqq(y) 
    # construct the second order term and reformat it so that it's in the stupid format
    f2 = 0.5 * np.concatenate([ham_qq @ ham_p , ham_pp @ ham_q]) + d2(y).flatten()
    f2 = np.array((f2[0:2],f2[2:4],f2[4:6],f2[6:]))
    
    y_next = y + H*( f + h*f2 ) 
    return y_next

def modified_flow_basic_order3(y,d2,d3,h=0.01):
    return 

def modified_energy_proj_order2(y,d2=d2_exp_euler,h=0.01):
    f = gradient(y)
    nab_H = nabla_H(y).flatten()
    ham_p,ham_q = nab_H[:4],nab_H[4:]
    ham_pp,ham_qq = partial_Hpp(y),partial_Hqq(y)
    # construct second order term f_2 with the projection term
    f2 = 0.5 * np.concatenate([ham_qq @ ham_p , ham_pp @ ham_q]) + d2(y).flatten() # modified eq term
    
    f2 += get_o2_projection_term_energy(y,d2) # add the energy projection term 
    
    # format it correctly
    f2 = np.array((f2[0:2],f2[2:4],f2[4:6],f2[6:]))
    
    y_next = y + H*( f + h*f2 )
    return y_next 

def modified_ang_proj_order2(y,h=0.01):
    return

def modified_energy_ang_proj_order2(y,h=0.01):
    return



