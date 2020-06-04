#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 09:32:13 2020

@author: steve
"""
# HENON HEILES MODEL AND POINTCARE CUTS

# %% IMPORTS 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

# the following two imports require us to re-define 'nabla_H' and 'gradient'
from two_body import energy_projection

# %% GLOBAL CONSTANTS

ENERGY_0 = 1/8 # 1/12 and 1/8 are in book p16
q20,p20, = 0.3,0.3
q10 = 0.0 # always 0
p10 = np.sqrt(2*ENERGY_0 - p20**2 - (0.5*(q10**2 + q20**2) + q10**2*q20 - q20**3/3))
Y0 = np.array([p10 , p20 , q10 , q20]) 
H = 0.01 # timestep
STEPS = 30000

# TOTAL_MOMENTUM = 
# TOTAL_ANG_MOMENTUM = 

J = np.matrix([[0,0,1,0],[0,0,0,1],[-1,0,0,0],[0,-1,0,0]])

# %% GLOBAL VARIABLES
cuts = [np.array([p20,q20])] # pointcarre cuts elements of form [p2,q2]
position_arr = [np.array([q10,q20])] # so that we can see where the dude goes

# %% FUNCTIONS THAT CALCULATE THINGS, LIKE GRADIENT ETC.

# it's smarter (more concises and legible) to define nabla H and use this when computing the gradient
def nabla_H(y):
    nabla_h = [y[0] , y[1] , 
               y[2] + 2*y[2]*y[3] , 
               y[3] + y[2]**2 - y[3]**2]
    return np.array(nabla_h)

# gradient is defined by the equation y' = J^{-1}\nabla H 
def gradient(y):
    grady = np.asarray(np.matmul(J.I,nabla_H(y)))[0]
    return grady

# nabla only of the q part of y
def nabla_q(q):
    nablaq = [q[0] + 2*q[0]*q[1] , 
              q[1] + q[0]**2 - q[1]**2]
    return np.array(nablaq) 
    
    
    

# %% FIRST INTEGRALS and other quantities

# evaluates hamiltonian
def get_energy(y):
    p1sq,p2sq,q1,q2 = y[0]**2 , y[1]**2 , y[2] , y[3]
    kinetic = 0.5*(p1sq + p2sq)
    potential = 0.5*(q1**2 + q2**2) + q1**2*q2 - q2**3/3
    return kinetic + potential

# returns the potential energy
def get_potential(y):
    q1,q2 = y[2] , y[3]
    potential = 0.5*(q1**2 + q2**2) + q1**2*q2 - q2**3/3
    return potential

# return kinetic energy 
def get_kinetic(y):
    p1sq,p2sq = y[0]**2 + y[1]**2
    kinetic = 0.5*(p1sq + p2sq)
    return kinetic
    

# %% UPDATE DATA (NOT NECESSARY FOR POINTCARE CUTS)


# %% NUMERICAL FLOW FUNCTIONS

# numerical flow operator for the explicit euler method
# UNVERIFIED
def exp_euler(y):
    # calculate the gradient
    grady = gradient(y)
    y_next = y + H*grady
    return y_next

# numerical flow operator for the stromer verlet scheme
# UNVERIFIED
def stromer_verlet(y):
    grady = gradient(y)
    # STEP 1 : \dot q_{n+1/2} = \dot q_n + h/2 * \dot p_n/m    [m=1 :) ]
    p1plushalf = y[0] + 0.5*grady[0]*H
    p2plushalf =  y[1] + 0.5*grady[1]*H
    # STEP2 : q_{n+1} = q_n + h\dot q_{n+1/2}
    q1_next = y[2] + H*p1plushalf
    q2_next = y[3] + H*p2plushalf
    # STEP 3 : p_{n+1} = p_{n+1/2} - h/2 \frac{\partial H(p,q)}{\partial q}
    nablaq_next = nabla_q([q1_next,q2_next])
    p1_next = p1plushalf - 0.5 * H * nablaq_next[0]
    p2_next = p2plushalf - 0.5 * H * nablaq_next[1]
    y_next = np.array([p1_next,p2_next,q1_next,q2_next])
    # print('ping',end=' ') # trace
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



# %% DISPLAY AND COMPARE THE INTEGRATORS

# helper method, calculates next n cuts
def get_next_n_cuts(y,method,n=20):
    q1,p1 = y[2],y[0]
    new_cuts = []
    while len(new_cuts)< n:
        # numerical flow
        y_next = method(y)
        q1_next = y_next[2]
        p1_next = y_next[0]
        
        # if there is a cut, record it
        if q1_next*q1 < 0 and p1 > 0:
            newcut = 0.5*(np.array([y[1],y[3]]) + np.array([y_next[1],y_next[3]]))
            new_cuts.append(newcut)
        
        y,q1,p1 = y_next,q1_next,p1_next 
        
    # return the new cuts you calculatednet_cuts
    return y,new_cuts
            

# helper method for below, saves figure and 
def save_four_cut_fig(multi_cuts,list_of_methods,net_cuts):
    # assumes there are four cuts to plotnet_cuts
    for i,(cuts,method) in enumerate(zip(multi_cuts,list_of_methods)):
        plt.subplot(2,2,i+1)
        plt.plot(np.array(cuts).T[0],np.array(cuts).T[1],'.k',markersize=0.8)
        plt.xlabel('Momentum')
        plt.ylabel('Position 2')
        plt.title('Pointcare cuts : {} : {} cuts\nEnergy = {}  :  stepsize = {}'.format(method.__name__ ,net_cuts, ENERGY_0 , H))
    plt.tight_layout()
    plt.savefig('E:{},cuts:{}.png'.format(ENERGY_0,net_cuts))
    print('just saved {}'.format(net_cuts),end='\t')
    plt.clf()
    

# DEPRICATED
# compares four integrators by simultaneously evaluating them 
def compare_methods(list_of_methods , stepsize=20 , total_cuts=50000):
    # time evolve all of them individually 
    # initialize the y's
    multi_y = [Y0[:],Y0[:],Y0[:],Y0[:]]
    multi_cuts = [cuts[:],cuts[:],cuts[:],cuts[:]]
    net_cuts = 0
    
    # keep looping though until the desired amount of cuts is reached
    plt.subplots(2,2,figsize=(10,10))
    while net_cuts < total_cuts:
        for index,(y,method) in enumerate(zip(multi_y,list_of_methods)):
            # obtain stepsize new cuts for each of them
            y,new_cuts = get_next_n_cuts(y,method,n=stepsize)
            multi_y[index] = y
            for i in new_cuts: multi_cuts[index].append(i)
        net_cuts += stepsize
        
        # saves figure with cuts
        save_four_cut_fig(multi_cuts,list_of_methods,net_cuts)
        
        # the smarter thing to do is save the actual cut data, and you can save over the thing, no need to save multiple ones...
        # I WILL IMPLEMENT THIS IN OTHER METHOD
            
    

# calculates large amount of cuts for specified method and saves arrays
def compute_and_save_cuts(method,stepsize = 200,total_cuts = 5000):
    """
    method : the proagation method or numerical flow function 
    stepsize : the number of cuts you calculate before each save 
    """
    global cuts
    cuts = []
    net_cuts = 0
    y = Y0[:]
    last_save_fname = None
    print('Computing {} cuts for Energy = {} using the {} method'.format(total_cuts,ENERGY_0,method.__name__))
    while net_cuts < total_cuts:
        y,next_cuts = get_next_n_cuts(y,method,n=stepsize)
        for i in next_cuts: cuts.append(i)
        net_cuts += stepsize 
        print('saving {}'.format(net_cuts))
        save_fname = 'pointcare_cuts_energy_{}_method_{}_cuts_{}'.format(ENERGY_0,method.__name__,net_cuts)
        np.save(save_fname , np.array(cuts))
        
        if last_save_fname: # replace it to prevent cluttering
            os.remove(last_save_fname)
        last_save_fname = save_fname+'.npy'
    print('DONE!')
    
    
# to test to see if one method is working properly, time propagate and plot the trajectory
def test_method(method,steps=10000):
    global position_arr
    y=Y0[:]
    for i in range(steps):
        y = method(y)
        position_arr.append(y[2:])
    position_arr = np.array(position_arr).T
    plt.plot(position_arr[0],position_arr[1])
    plt.title('method={},energy={}\nsteps={},stepsize={}'.format(method.__name__,ENERGY_0,steps,H))
    plt.show()

# %% MAIN

if __name__=='__main__':
    # test method to see if working properly 
    test_method(stromer_verlet,steps = 30000) 
    
# if __name__=='__main__': 
#     # compute a whole bunch of cuts for each method 
#     # compute_and_save_cuts(exp_euler) 
#     # for i in np.linspace(1/12,1/8,10): 
#     ENERGY_0 = 1/8 # 1/12 and 1/8 are in book p16 
#     q20,p20, = 0.3,0.3 
#     q10 = 0.0 # always 0 
#     p10 = np.sqrt(2*ENERGY_0 - p20**2 - (0.5*(q10**2 + q20**2) + q10**2*q20 - q20**3/3))
#     Y0 = np.array([p10 , p20 , q10 , q20]) 
#     # compute_and_save_cuts(syplectic_euler)
#     compute_and_save_cuts(stromer_verlet,stepsize=20)
#     # compute_and_save_cuts(exp_euler)
    



# if __name__=='__main__':
#     compare_methods([syplectic_euler,
#                      syplectic_euler,
#                      syplectic_euler,
#                      syplectic_euler],
#                     total_cuts=10000)
    

# if __name__=='__main__':
#     y = Y0[:]
#     q1 = y[2]
#     p1 = y[0]
#     # for i in range(STEPS):
#     plt.figure()
#     method = syplectic_euler
#     while True:
#         y_next = syplectic_euler(y)
#         q1_next = y_next[2]
#         p1_next = y_next[0]
        
#         # if there is a cut, record it
#         if q1*q1_next < 0 and p1>0:
#             newcut = 0.5*(np.array([y[1],y[3]]) + np.array([y_next[1],y_next[3]]))
#             cuts.append(newcut)
#             bool_var = True
            
#         y,q1,p1 = y_next , q1_next , p1_next
            
#         if len(cuts) % 50 == 0 and bool_var == True:
#             bool_var = False
#             print(len(cuts),end='cuts ,\t')
#             # display the cuts every time you get a new 100 cuts
#             # plt.figure()
#             plt.plot(np.array(cuts).T[0],np.array(cuts).T[1],'.k',label='{}'.format(len(cuts)),
#                       markersize=.8)
#             plt.xlabel('momentum 2')
#             plt.ylabel('position 2')
#             # plt.legend(fontsize=8)
#             plt.title('Pointcare cuts : {}\nEnergy = {}  :  {} cuts'.format(method.__name__,ENERGY_0,len(cuts)))
#             plt.show()
#             plt.clf()
            
        
                
        
    
    
    
    
    
    