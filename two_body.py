#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 13:40:57 2020

@author: steve
"""
# TWO BODY PROBLEM

# %% imports
import numpy as np
import matplotlib.pyplot as plt

# %% define constants

# contants
G = 1 # gravitational constant
M1 = 1.0
M2 = 1.2
Y0 = np.array([np.array([.4,0.]),np.array([-.4,0.]),
               np.array([0.,-1.]),np.array([0.,1.])])# cartesian (p1,p2,q1,q2)

H = 0.01 # timestep
STEPS = 3000

# go into CM frame by setting net momentum to zero


"""
the hamiltonian is H(y) = p1^2/2m1 + p2^2/2m2 - GM1M2/(q1-q2)
y' = inv J nabla H = grad y
where J is the canonical syplectic form (ASM)
"""

"""
once you figure out how to code the hamiltonian make this into

"""

"""
idea for a display function, shows you all the relative radii (relative distances betwwen objects)
"""

# %% 

energy_arr = []
linear_momentum_arr = []
ang_momentum_arr = []
position_arr = [] # [Y0[2:]]
time = 0
time_arr = []
radius_arr = []


### FUNCTIONS THAT CALCULATE THINGS

# helper function, calculates absolute value squared of vector
def normsq(v):
    return sum([i**2 for i in v])

# return y' given y
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
    grady = np.array(grady)
    return grady

def grad_q(q):
    # assume 2bp
    # takes q returns it's gradient, which is negative the derivative of p
    r = np.array([q[0][0]-q[1][0] , q[0][1]-q[1][1]])#r from 1 to 0
    rabs = np.sqrt(normsq(r))
    gradq = []
    gradq.append(-r*G*M1*M2*(rabs**(-3)))# vector
    gradq.append(r*G*M1*M2*(rabs**(-3)))
    return gradq

def get_energy(y):
    r = np.array([y[2][0]-y[3][0] , y[2][1]-y[3][1]])#r from 1 to 0
    rabs = np.sqrt(normsq(r))
    p1sq,p2sq = normsq(y[0]),normsq(y[1])
    potential = - G*M1*M2 / rabs
    return 0.5*p1sq/M1 + 0.5*p2sq/M2 + potential 

def get_total_angular_momentum(y):
    # calculate it for each of the planets
    l1 = np.linalg.det(np.array([y[0],y[2]]))
    l2 = np.linalg.det(np.array([y[1],y[3]]))
    return l1 + l2

def get_total_linear_momentum_abs(y):
    # calculate the net linear momentum of the system (abs value)
    return np.sqrt(normsq(y[0]+y[1]))


def update_dta(y):
    global position_arr,energy_arr,ang_momentum_arr,linear_momentum_arr,time_arr,radius_arr
    # update 
    time_arr.append(time)
    position_arr.append(y[2:])# update the position array
    energy_arr.append(get_energy(y))
    ang_momentum_arr.append(get_total_angular_momentum(y))
    linear_momentum_arr.append(get_total_linear_momentum_abs(y))
    
    r = np.array([y[2][0]-y[3][0] , y[2][1]-y[3][1]])#r from 1 to 0
    rabs = np.sqrt(normsq(r))
    radius_arr.append(rabs)


### FUNCTIONS THAT DISPLAY THINGS
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
    
def display_invarients():
    # the problem here is that they all need to be scalled - ill just do them on different plots for now
    fig,ax = plt.subplots(2,2,figsize=(10,10))
    plt.subplot(221)
    plt.plot(time_arr,energy_arr,label='energy')
    # scale the radius and do a comparison
    min_e = min(energy_arr)
    radius_arr_scaled = np.array(radius_arr) * (max(energy_arr)-min_e)/max(radius_arr) + min_e*np.ones(len(radius_arr))
    plt.plot(time_arr,radius_arr_scaled,label='radius (scaled)')
    
    plt.xlabel('time')
    plt.ylabel('net energy of the system')    
    plt.title('energy\nh={} , steps={}'.format(H,STEPS))
    plt.legend(fontsize=14)
    
    plt.subplot(222)
    plt.plot(time_arr,ang_momentum_arr,label='angular momentum')
    plt.title('angular momentum\nh={} , steps={}'.format(H,STEPS),fontsize=16)
    plt.xlabel('time',fontsize=14)
    plt.ylabel('net angular momentum abs of system',fontsize=14)
    plt.legend()
    
    plt.subplot(223)
    plt.plot(time_arr,linear_momentum_arr,label='linear momentum\nh={} , steps={}'.format(H,STEPS))
    plt.title('linear momentum')
    plt.xlabel('time',fontsize=14)
    plt.ylabel('net linear momentum abs of system',fontsize=14)
    plt.legend()
    
    plt.subplot(224)
    plt.plot()
    
    plt.tight_layout()
    

# %%
    
# advances solution forward one timestep using the explicit euler method
def exp_euler_timestep(y):
    # calculate the gradient
    grady = gradient(y)
    y_next = y + H*grady
    
    return y_next

# advances solution forward one timestep using Stromer-Verlet scheme (p8)
def stromer_verlet_timestep(y):
    grady = gradient(y)
    # STEP 1 : \dot q_{n+1/2} = \dot q_n + h/2 + \dot p_n/m
    p1plushalf = y[0] + 0.5*grady[0]*H
    p2plushalf = y[1] + 0.5*grady[1]*H
    # STEP2 : q_{n+1} = q_n + h\dot q_{n+1/2}
    q1_next = y[2] + H*p1plushalf/M1
    q2_next = y[3] + H*p2plushalf/M2
    # STEP 3 : p_{n+1} = p_{n+1/2} - h/2 \frac{\partial H(p,q)}{\partial q}
    gradq_next = grad_q([q1_next,q2_next])
    p1_next = p1plushalf + 0.5*H* gradq_next[0]
    p2_next = p2plushalf + 0.5*H* gradq_next[1]
    y_next = np.array([p1_next,p2_next,q1_next,q2_next])
    
    return y_next

# advances solution one timestep using explicit trapezium rule (runge kutta)
def exp_trapezium(y):
    k1 = gradient(y)
    k2 = gradient(y + H*k1)
    y_next = y + 0.5*H*(k1+k2)
    return y_next

    
# %% main
if __name__ == "__main__":
    y = Y0[:]
    for i in range(STEPS):
        # y = exp_euler_timestep(y)
        # y = stromer_verlet_timestep(y)
        y = exp_trapezium(y)
        
        # update the time
        time+=H
    
        update_dta(y)
    display_trajectories()
    # display_total_energy()
    display_invarients()
    
    
    
    





