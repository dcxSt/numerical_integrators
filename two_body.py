#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 13:40:57 2020

@author: steve
"""
# TWO BODY PROBLEM

# %% IMPORTS
import numpy as np 
import matplotlib.pyplot as plt 
import itertools
import math

# %% GLOBAL CONSTANTS

# contants
G = 1 # gravitational constant
M1 = 1.0
M2 = 1.4
M = M1+M2
M_REDUCED = M1*M1 / M
Y0 = np.array([np.array([.4,0.]),np.array([-.4,0.]),
               np.array([0.,-1.0]),np.array([0.,1.0])])# cartesian (p1,p2,q1,q2)

# EPSILON = 10.0 # see projection method # depricated 
H = 0.01 # timestep 
STEPS = 100000
ENERGY_0 = None # get_energy(Y0) 
TOTAL_MOMENUM_0 = None # get_total_linear_momentum_abs(Y0) 
TOTAL_ANG_MOMENTUM_0 = None # get_total_angular_momentum(Y0) 

NABLA_LIN_X = np.array([np.array([1,0]),np.array([1,0]),np.array([0,0]),np.array([0,0])])
NABLA_LIN_Y = np.array([np.array([0,1]),np.array([0,1]),np.array([0,0]),np.array([0,0])])
# go into CM frame by setting net momentum to zero? 



# %% GLOBAL VARIABLES

energy_arr = []
linear_momentum_arr = []
net_lin_mom_x_arr = []
net_lin_mom_y_arr = []
ang_momentum_arr = []
lev_set_uvec = {"energy":[],"angular_m":[],"linear_x":[],"linear_y":[]} # dict of the angles between first integrals
position_arr = [] # [Y0[2:]]
velocity_arr = [] # np.array([Y0[0]/M1 , Y0[1]/M2])
time = 0
time_arr = []
radius_arr = []
k_factor_arr = []
overflow_count = 0


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


# %% FIRST INTEGRALS

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


# %% STAND ALONE UPDATE DTA

# helper for update_dta, stores the directions each level set is pointing
def update_lev_set_uvec(y,first_int_nablas):
    global lev_set_uvec
    for nabla,name in first_int_nablas:
        nablay = nabla(y)
        u_vec = nablay / np.sqrt(supervec_norm(nablay))
        lev_set_uvec[name].append(u_vec)
    return

# helper function for keeping track of constants and stuff over time
def update_dta(y,first_int_nablas=[(nabla_H,"energy"),
                                   (nabla_l,"angular_m"),
                                   (nabla_lin_x,"linear_x"),
                                   (nabla_lin_y,"linear_y")]):
    global position_arr,energy_arr,ang_momentum_arr,linear_momentum_arr,time_arr,radius_arr
    # update 
    time_arr.append(time) 
    position_arr.append(y[2:])# update the position array 
    velocity_arr.append(np.array([y[0]/M1 , y[1]/M2]))
    energy_arr.append(get_energy(y)) 
    ang_momentum_arr.append(get_total_angular_momentum(y)) 
    linear_momentum_arr.append(get_total_linear_momentum_abs(y)) 
    net_lin_mom_x_arr.append(get_lin_mom_x(y)) 
    net_lin_mom_y_arr.append(get_lin_mom_y(y))
    update_lev_set_uvec(y,first_int_nablas)
    k_factor_arr.append(k_factor(y))
    
    
    r = np.array([y[2][0]-y[3][0] , y[2][1]-y[3][1]])#r from 1 to 0
    rabs = np.sqrt(normsq(r))
    radius_arr.append(rabs)


# %% FUNCTIONS THAT DISPLAY THINGS
def display_trajectories():
    global position_arr
    position_arr = np.array(position_arr)# here i'm using the global variable pos ar
    m1x = position_arr.T[0][0]
    m2x = position_arr.T[0][1]
    m1y = position_arr.T[1][0]
    m2y = position_arr.T[1][1]
    position_arr = list(position_arr)# turn it back into a list, just for good practice
    
    fig,ax = plt.subplots(figsize=(10,10))
    ax.plot(m1x,m1y)
    ax.plot(m2x,m2y)
    plt.title('Trajectories   :   h={} , steps={}'.format(H,STEPS),fontsize=17)
    plt.show()
    
def display_trajectories_relative(method_name=None):
    fig,ax = plt.subplots(figsize=(10,10))
    
    plot_relative_trajectories()
    
    if method_name: plt.title('Relative trajectories : h={} , steps={}\nmethod={}'.format(H,STEPS,method_name),fontsize=17)
    else: plt.title('Relative trajectories : h={} , steps={}'.format(H,STEPS),fontsize=17)
    plt.show()
    
def display_relative(method_name=None):
    fig,ax = plt.subplots(1,2,figsize=(10,20))
    
    plt.subplot(121)
    plot_relative_trajectories()
    if method_name: plt.title('Relative trajectories : h={} , steps={}\nmethod={}'.format(H,STEPS,method_name),fontsize=17)
    else: plt.title('Relative trajectories : h={} , steps={}'.format(H,STEPS),fontsize=17)
    
    plt.subplot(122)
    plot_relative_velocities()
    if method_name: plt.title('Relative velocities : h={} , steps={}\nmethod={}'.format(H,STEPS,method_name),fontsize=17)
    else: plt.title('Relative velocities : h={} , steps={}'.format(H,STEPS),fontsize=17)
    
    plt.show()
    
def plot_relative_trajectories():
    global position_arr
    position_arr = np.array(position_arr)
    m1x = position_arr.T[0][0]
    m2x = position_arr.T[0][1]
    m1y = position_arr.T[1][0]
    m2y = position_arr.T[1][1]
    relative_position = [m1x-m2x , m1y-m2y]
    
    plt.plot(relative_position[0],relative_position[1],'-',alpha=.7,linewidth=0.5)
    return
    
def plot_relative_velocities():
    global velocity_arr
    velocity_arr = np.array(velocity_arr)
    m1vx = velocity_arr.T[0][0]
    m2vx = velocity_arr.T[0][0]
    m1vy = velocity_arr.T[1][0]
    m2vy = velocity_arr.T[1][1]
    relative_velocity = [m1vx-m2vx , m1vy-m2vy]
    
    plt.plot(relative_velocity[0],relative_velocity[1],'-',alpha=.7,linewidth=0.5)
    return
    
def display_total_energy():
    # global energy_arr
    fig,ax = plt.subplots(figsize=(8,8))
    ax.plot(time_arr,energy_arr)
    plt.title('net energy\nh={} , steps={}'.format(H,STEPS),fontsize=20)
    # ax.plot(time_arr,linear_momentum_arr)
    # ax.plot(time_arr,angular)
    
def display_total_angular_momentum(): 
    fig,ax = plt.subplots(figsize=(8,8)) 
    ax.plot(time_arr,ang_momentum_arr) 
    plt.title('net angular momentum\nh={} , steps={}'.format(H,STEPS),fontsize=20)
    plt.xlabel('time') 
    plt.ylabel('net linear momentum of system') 
    
def display_total_linear_momentum():
    fig,ax = plt.subplots(figsize=(8,8))
    ax.plot(time_arr,linear_momentum_arr)
    plt.title('net linear momentum\nh={} , steps={}'.format(H,STEPS), fontsize=20)
    plt.xlabel('time')
    plt.ylabel('net linear momentum of system')
    
def plot_invarient_level_set_angles(ls_names=None):
    if not ls_names: ls_names = lev_set_uvec.keys()
    for i,j in itertools.combinations(ls_names,r=2):
        if i != j:
            u_i,u_j = lev_set_uvec[i],lev_set_uvec[j]
            # calculate the angle between them and display them
            angleij = [math.acos(abs(supervec_dot(u_i[k],u_j[k])))*180/np.pi for k in range(len(u_i))] # they should all be the same size
            # plot the angle
            plt.plot(time_arr,angleij,label="{} and {}".format(i,j))
    plt.legend()
    plt.title("Angles between invarient level sets",fontsize=16) 
    plt.xlabel('time',fontsize=14) 
    plt.ylabel('angle',fontsize=14) 
    
    
def display_invarients():
    # the problem here is that they all need to be scalled - ill just do them on different plots for now
    fig,ax = plt.subplots(3,2,figsize=(10,10))
    plt.subplot(321)
    plt.plot(time_arr,energy_arr,label='energy')
    # # scale the radius and do a comparison
    # min_e = min(energy_arr)
    # radius_arr_scaled = np.array(radius_arr) * (max(energy_arr)-min_e)/max(radius_arr) + min_e*np.ones(len(radius_arr))
    # plt.plot(time_arr,radius_arr_scaled,label='radius (scaled)')
    
    plt.xlabel('time')
    plt.ylabel('net energy of the system')    
    plt.title('energy\nh={} , steps={}'.format(H,STEPS))
    plt.legend(fontsize=14)
    
    plt.subplot(322)
    plt.plot(time_arr,ang_momentum_arr,label='angular momentum')
    plt.title('angular momentum',fontsize=16)
    plt.xlabel('time',fontsize=14)
    plt.ylabel('net ang momentum of system',fontsize=10)
    plt.legend()
    
    plt.subplot(323)
    plt.plot(time_arr,linear_momentum_arr,label='linear moment between them and display um\nh={} , steps={}'.format(H,STEPS))
    plt.plot(time_arr,net_lin_mom_x_arr,label='lin momentum x')
    plt.plot(time_arr,net_lin_mom_y_arr,label='lin momentum y')
    plt.title('linear momentum')
    plt.xlabel('time',fontsize=14)
    plt.ylabel('Linear Momentum\nMethod : {}'.format(method.__name__),fontsize=14)
    plt.legend()
    
    plt.subplot(324)
    
    plt.title('----',fontsize=15)
    plt.xlabel('----',fontsize=14)
    plt.ylabel('----')
    plt.legend()
    
    plt.subplot(325)
    plt.plot(time_arr,k_factor_arr)
    plt.title('k factor\nmeasures the eccentricity, zero for circular',fontsize=15)
    plt.xlabel('time',fontsize=14)
    plt.ylabel('net linear momentum y')
    
    plt.subplot(326)
    plot_invarient_level_set_angles(["energy","angular_m"])
    
    plt.tight_layout()
    

# %% NUMERICAL FLOW FUNCTIONS
    
# advances solution forward one timestep using the explicit euler method
def exp_euler(y):
    # calculate the gradient
    grady = gradient(y) 
    y_next = y + H*grady 
    return y_next 

def exp_euler_modified_energy(y):
    grady_mod = modified_gradient_energy_attractor(y)
    y_next = y + H*grady_mod
    return y_next

def exp_euler_modified_energy_ang(y):
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

# %% PROJECTIONS

# projects y onto the energy manifold
def energy_projection(y):
    nabla_h = nabla_H(y)
    nabla_h_flat = np.ndarray.flatten(nabla_h)
    lam = (ENERGY_0 - get_energy(y))/(np.dot(nabla_h_flat,nabla_h_flat))
    y_twiddle = y + lam*nabla_h
    return y_twiddle
    
    
# naive should be same as other if manifold tanjent spaces are perpendicular 
# this method can do with some optimization (/ refactoring) 
def naive_first_integral_projection(y,first_integrals=[(get_energy,nabla_H),
                                               (get_total_angular_momentum,nabla_l)]):
    y_proj=y[:]
    # apply each projection one after the other 
    for i in first_integrals:
        fint,nabla_fint = i[0],i[1](y_proj) 
        lambda_i = fint(Y0) - fint(y_proj) 
        lambda_i /= supervec_norm(nabla_fint)
        y_proj = y_proj + lambda_i*nabla_fint 
        
    return y_proj
        
# same as naive but does it in k steps (is also k times slower!), also it's only effective if 
# there are multiple first integrals, no point if there is just one 
def iterated_first_integral_projection(y,k=2,first_integrals=[(get_energy,nabla_H)]):
    y_proj = y[:]
    # apply k times 
    for j in range(k,0,-1):
        # apply each projection one after the other 
        for i in first_integrals: 
            fint,nabla_fint = i[0],i[1](y_proj) 
            lambda_i = fint(Y0) - fint(y_proj) 
            lambda_i /= supervec_norm(nabla_fint) 
            y_proj = y_proj + (lambda_i / j) * nabla_fint 
    
    return y_proj


def first_integral_projection_2(y,first_integrals=[(get_energy,nabla_H),
                                                   (get_lin_mom_x,nabla_lin_x),
                                                   (get_lin_mom_y,nabla_lin_y)]):
    global overflow_count
    
    # first compute the gradients
    fivag = [(i[0](y),i[0](Y0),i[1](y)) for i in first_integrals]
    finty,fint0,beta = [j[1] for j in fivag],[j[0] for j in fivag],[j[2] for j in fivag]
    beta_prime = [j[:] for j in beta] # copy of the gradients to be modified
    
    # just a check to make sure they all have same energy a
    
    # for each j, project all the i's out of the j surface
    for j in range(len(fivag)):
        for i in range(len(fivag)):
            if j != i:
                try:
                    beta_i_prime = beta_prime[i] - supervec_dot(beta[j],beta_prime[i]) / supervec_dot(beta[j],beta_prime[j]) * beta_prime[j]
                    beta_prime[i] = beta_i_prime 
                except OverflowError("overflow, probably the manifolds have small angle"):
                    overflow_count+=1 
    
    y_proj = y[:] 
    # project along each axis
    for e0,ey,b_i_p,b_i in zip(finty,fint0,beta_prime,beta):
        try:
            lambda_i = (e0 - ey) / supervec_dot(b_i_p,b_i)
            y_proj += lambda_i * b_i_p 
        except OverflowError("overflow in computing lambda_i"):
            overflow_count+=1
            
    return y_proj


# General first integral projection 
# takes list of tuples (first integral , gradient function)
# MATH HERE IS A LITTLE FAULTY, NEEDS REEVALUATING, IN LATEX TOO
### DEPRICATED
# def first_integral_projection(y,first_integrals=[(get_energy,nabla_H),
#                                                (get_total_angular_momentum,nabla_l)]):
#     # first compute the gradients 
#     fivag = [(i[0](y),i[1](y)) for i in first_integrals] # first integral values and gradients
    
#     # keep track of overflow errors 
#     global overflow_count
    
#     # next compute the adjusted gradients for each of them 
#     nabla_primes = []
#     for i in range(len(fivag)):
#         nabla_i_prime = fivag[i][1]
#         for j in range(len(fivag)):
#             if i!=j:
#                 nabla_j = fivag[j][1]
#                 try:
#                     difference = supervec_dot(nabla_i_prime,nabla_j) / supervec_dot(nabla_j,nabla_j) * nabla_j
#                     nabla_i_prime -= difference
#                 except OverflowError:
#                     print('Overflow error in computing nabla_i_prime at step {}'.format(len(time_arr)))
#                     overflow_count+=1

#         nabla_primes.append(nabla_i_prime)
    
#     # now compute the values lambda_i'
#     lambda_primes = [] 
#     for i in range(len(nabla_primes)):
#         try:
#             numerator = first_integrals[i][0](Y0) - fivag[i][0]
#             # to ensure there is no blow up we add an epsilon term to the denominator 
#             denominator = np.abs(supervec_dot(nabla_primes[i],fivag[i][1])) + EPSILON*np.abs(numerator)
#             lambda_i_prime = numerator / denominator
#         except:
#             print('Overflow error in computing lambda_i_prime at step {}, setting lambda_i_prime to zer0'.format(len(time_arr)))
#             overflow_count+=1 
#             lambda_i_prime = 0 
        
#         lambda_primes.append(lambda_i_prime)
        
#     # now project 
#     y_projected  = y + sum([i*j for i,j in zip(lambda_primes,nabla_primes)])
#     return y_projected

def project_invarient_manifold_standard_newton_iter(y,k=2,first_integrals=[(get_energy,nabla_H),
                                                             (get_total_angular_momentum,nabla_l)]):
    # define g and g', g : \R^n \to \R^m 
    def g(y):
        g = []
        for fint,nabla in first_integrals:
            g.append(fint(y)-fint(Y0))
        return np.array(g)
    # def nablag(y):
    #     gprime = []
    #     for fint,nabla in first_integrals: 
    #         gprime.append(nabla(y)) 
    #     return np.array(gprime)
    def nablag_flat(y):
        gprime = []
        for fint,nabla in first_integrals:
            gprime.append(np.hstack(nabla(y)))
        return np.array(gprime)
    
    y_proj = y[:]
    
    # definte initial guess for lambda
    lambda_vec = []
    for i in first_integrals: 
        fint,nabla = i[0],i[1](y_proj) 
        lambda_i = fint(Y0) - fint(y_proj) 
        lambda_i /= supervec_norm(nabla) 
        lambda_vec.append(lambda_i) 
    lambda_vec = np.array(lambda_vec) 
    
    # do a couple newton iterations
    nablag_y_flat = nablag_flat(y)
    try:
        csm = np.linalg.inv(np.dot(nablag_y_flat,nablag_y_flat.T)) # constant square matrix
        for i in range(k):
            delta_lambda = - np.dot(csm , g(y + to_supervec(np.dot(nablag_y_flat.T,lambda_vec))))
            lambda_vec = lambda_vec + delta_lambda 
        
        # project y 
        y_proj = y + to_supervec(np.dot(nablag_y_flat.T , lambda_vec))
    except:
        print("Singular matrix, projection not possible")
        
    return y_proj 
    


# %% DISPLAY AND COMPARE THE INTEGRATORS
    
# 
def display_compare_methods(data):
    plt.subplots(2,2,figsize=(10,10))
    
    # display energy
    plt.subplot(221)
    for times,energies,name in zip(data['time'],data['energy'],data['name']):
        plt.plot(times,energies,label=name,alpha=.6)
    plt.title('Energy : h={} , steps={}'.format(H,STEPS),fontsize=16)
    plt.ylabel('Energy')
    plt.xlabel('Time')
    plt.legend()
    
    # display trajectories
    plt.subplot(222)
    for times,positions,name in zip(data['time'],data['position'],data['name']):
        m1x = positions.T[0][0]
        m2x = positions.T[0][1]
        m1y = positions.T[1][0]
        m2y = positions.T[1][1]
        relative_position = [m1x-m2x , m1y-m2y]
        
        plt.plot(relative_position[0],relative_position[1],label=name,alpha=.6)
    plt.title('Relative trajectories : h={} , steps={}'.format(H,STEPS),fontsize=16)
    plt.ylabel('Relative Y')
    plt.xlabel('Relative X')
    plt.legend()
    
    # display linear momentum
    plt.subplot(223)
    for times,momentums,name in zip(data['time'],data['lin_momentum'],data['name']):
        plt.plot(times,momentums,label=name,alpha=.6)
    plt.title('Linear Momenta : h={} , steps={}'.format(H,STEPS),fontsize=16)
    plt.ylabel('Linear Momentum')
    plt.xlabel('Time')
    plt.legend()
    
    # display angular momentum
    plt.subplot(224)
    for times,ang_momentums,name in zip(data['time'],data['ang_momentum'],data['name']):
        plt.plot(times,ang_momentums,label=name,alpha=.75)
    plt.title('Angular Momenta : h={} , steps={}'.format(H,STEPS),fontsize=16)
    plt.ylabel('Angular Momentum')
    plt.xlabel('Time')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    
def compare_methods(list_of_methods):
    global time,energy_arr,linear_momentum_arr,ang_momentum_arr,position_arr,time_arr,radius_arr
    data = {'name':[] , 'energy':[] , 'lin_momentum':[] , 'ang_momentum':[] ,
            'position':[] , 'time':[] , 'radius':[]}
    for method in list_of_methods:
        y = Y0[:]
        time=0
        for i in range(STEPS):
            y = method(y)
            time+=H
            update_dta(y)
        
        # save all the data in a big dictionary of arrays
        data['name'].append(method.__name__)
        data['energy'].append(np.array(energy_arr[:]))
        data['lin_momentum'].append(np.array(linear_momentum_arr[:]))
        data['ang_momentum'].append(np.array(ang_momentum_arr[:]))
        data['position'].append(np.array(position_arr[:]))
        data['time'].append(np.array(time_arr[:]))
        data['radius'].append(np.array(radius_arr[:]))
        
        # reset the global vairables for a new run
        energy_arr = []
        linear_momentum_arr = []
        ang_momentum_arr = []
        position_arr = [] # [Y0[2:]]
        time = 0
        time_arr = []
        radius_arr = []
        
    return data

def compare_energy_projection(method):
    global time,energy_arr,linear_momentum_arr,ang_momentum_arr,position_arr,time_arr,radius_arr
    data = {'name':[] , 'energy':[] , 'lin_momentum':[] , 'ang_momentum':[] ,
            'position':[] , 'time':[] , 'radius':[]}
    
    # first without and then with projection
    for boolian_var in (False,True):
        y = Y0[:]
        time=0
        for i in range(STEPS):
            y = method(y)
            time+=H
            if boolian_var==True: y = energy_projection(y)
            update_dta(y)
        
        # save all the data in a big dictionary of arrays
        if boolian_var==True:
            data['name'].append(method.__name__+' With Energy Projection')
        else: data['name'].append(method.__name__+' No Projection')
        data['energy'].append(np.array(energy_arr[:]))
        data['lin_momentum'].append(np.array(linear_momentum_arr[:]))
        data['ang_momentum'].append(np.array(ang_momentum_arr[:]))
        data['position'].append(np.array(position_arr[:]))
        data['time'].append(np.array(time_arr[:]))
        data['radius'].append(np.array(radius_arr[:]))
        
        # reset the global vairables for a new run 
        energy_arr = []
        linear_momentum_arr = []
        ang_momentum_arr = []
        position_arr = [] # [Y0[2:]]
        time = 0
        time_arr = []
        radius_arr = []
    
    return data

def display_relative_velocity():
    return


def define_starting_momentum(E,L,q1,q2,p1):
    global Y0
    # solve ang momentum for p21 and sub into energy eq 
    a = np.linalg.norm()
        
# %% INITIALIZE CONSTANTS THAT REQUIRE FUNCTIONS
ENERGY_0 = get_energy(Y0)
TOTAL_MOMENUM_0 = get_total_linear_momentum_abs(Y0)
TOTAL_ANG_MOMENTUM_0 = get_total_angular_momentum(Y0)

# %% main
# if __name__ == "__main__":
#     # choose a bunch of starting positions, 
#     # make sure they are all elliptical with the same energy and angular momentum
#     # to do this you can just rotate them
#     # y1 = np.array([np.array([.8,0.]),np.array([-.8,0.]),
#     #             np.array([0.,-0.3]),np.array([0.,0.3])])
#     y1 = np.array([np.array([.4,0.]),np.array([-.4,0.]),
#                np.array([0.,-1.0]),np.array([0.,1.0])])
    
#     Y0 = y1[:]
#     y_arr = []
#     for i in range(5):
#         yi = []
#         # the following two lines are only valid for when y1[0][1] and y1[1][1] are zero
#         yi.append(np.array([y1[0][0]*math.cos(i*math.pi/3) , y1[0][0]*math.sin(i*math.pi/3)]))
#         yi.append(np.array([y1[1][0]*math.cos(i*math.pi/3) , y1[1][0]*math.sin(i*math.pi/3)]))
#         yi.append(y1[2])
#         yi.append(y1[3])
#         y_arr.append(yi)
#     # advance each 100 time units, or 10000 time-steps with projections and exp euler method
#     energies = [get_energy(y) for y in y_arr]
    
#     method = exp_euler 
    
#     steps_per_leap = 7000
#     H = 0.01 
#     for i in range(15): 
#         traces = [] 
#         everything_traces = []
#         for k in range(len(y_arr)): 
#             y = y_arr[k]
#             trace2 = []
#             # time evolve it x steps 
#             for j in range(steps_per_leap): 
#                 y = method(y) 
#                 # projection 
#                 y = first_integral_projection_2(y,first_integrals=[(get_energy,nabla_H),
#                                                     (get_total_angular_momentum,nabla_l)])
                
#                 # calculate the relative velocity and append it to the traces
#                 if j % 11==0:
#                     trace2.append(y[2:])
            
#             # redefine y_arr to update it and update traces 
#             y_arr[k] = y 
#             traces.append(trace2[:]) 
#         # display the traces 
#         plt.figure() 
#         for trace in traces: 
#             # get the velocities 
#             trace2 = np.array(trace2)
#             m1x = trace2.T[0][0]
#             m2x = trace2.T[0][1]
#             m1y = trace2.T[1][0]
#             m2y = trace2.T[1][1]
#             relative_position = [m1x-m2x , m1y-m2y]
#             plt.plot(relative_position[0],relative_position[1],'x',alpha=.5)
#         plt.show() 
    
    
    
if __name__ == "__main__":
    # data = compare_methods([stromer_verlet_timestep,
    #                         fourth_order_kutta])
    # data = compare_energy_projection(syplectic_euler)
    # display_compare_methods(data)
    
    y = Y0[:]
    for i in range(STEPS):
        # method = exp_euler 
        # method = exp_euler_modified_energy_ang 
        method = exp_euler 
        y = method(y)
        
        # y = fourth_order_kutta(y)
        # y = exp_euler(y)
        # y = exp_trapezium(y)
        # y = exp_midpoint(y)
        # y = syplectic_euler(y)
        
        # projection step
        # y = energy_projection(y)
        
        # y = naive_first_integral_projection(y,first_integrals=[(get_energy,nabla_H),
        #                                                         (get_total_angular_momentum,nabla_l)])
                                                                # (get_lin_mom_x,nabla_lin_x),
                                                                # (get_lin_mom_y,nabla_lin_y),
                                                                # (get_total_angular_momentum,nabla_l)])
        
        
        # y = naive_first_integral_projection(y,first_integrals=[(get_total_angular_momentum,nabla_l)])
        # y = first_integral_projection_2(y,first_integrals=[(get_energy,nabla_H),
        #                                     (get_total_angular_momentum,nabla_l)])
                                                            #(get_lin_mom_x,nabla_lin_x),
                                                            #(get_lin_mom_y,nabla_lin_y)])
        
        # y= iterated_first_integral_projection(y,k=5,first_integrals=[(get_energy,nabla_H),
        #                                                              (get_total_angular_momentum,nabla_l)])
        
        y = project_invarient_manifold_standard_newton_iter(y,k=2,first_integrals=[(get_energy,nabla_H),
                                                                                    (get_total_angular_momentum,nabla_l)])
        
        # update every k steps 
        k=10
        if i%k==0:
            time+=H*k
            update_dta(y)
        
    print("overflow occurence : {}".format(overflow_count)) 
    
    # DISPLAY
    # before displaying you may want to take off some of the data from the trajcetories so that you can see what the 'stable orbit' looks like
    # CHUCK = 300000
    # try:
    #     position_arr = position_arr[CHUCK:]
    # except:
    #     print("position array too small to chuck anything, not gonna do it")
    display_relative(method_name = method.__name__)
    # display_total_energy() 
    display_invarients()
    plt.show()
    
