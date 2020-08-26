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

from kep_util_globvar import M1,M2,Y0,H,STEPS
from kep_util_globvar import normsq,supervec_dot,supervec_norm,k_factor
from kep_util_globvar import get_energy,get_total_angular_momentum,get_total_linear_momentum_abs,get_lin_mom_x,get_lin_mom_y

from kep_derivative_functions import nabla_lin_y,nabla_lin_x,nabla_l,nabla_H

# from kep_derivative_functions import *
import kep_integrators as integrators 
import kep_projection_operators as proj

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


# %% UPDATE DTA

# helper for update_dta, stores the directions each level set is pointing
def update_lev_set_uvec(y,first_int_nablas):
    global lev_set_uvec
    for nabla,name in first_int_nablas:
        nablay = nabla(y)
        u_vec = nablay / np.sqrt(supervec_norm(nablay))
        lev_set_uvec[name].append(u_vec)
    return

def reset_dta():
    global time,y,velocity_arr,net_lin_mom_y_arr,net_lin_mom_x_arr,position_arr,energy_arr,ang_momentum_arr,linear_momentum_arr,time_arr,radius_arr,k_factor_arr
    time_arr = []
    position_arr = []
    velocity_arr = []
    energy_arr = []
    ang_momentum_arr = []
    linear_momentum_arr = []
    net_lin_mom_x_arr = []
    net_lin_mom_y_arr =[]
    k_factor_arr = []
    radius_arr = []
    y=Y0[:]
    time=0
    for name in lev_set_uvec.keys():
        lev_set_uvec[name] = []
    
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
    
def display_relative(steps_sep=0.5,method_name=None,projection_type='None'):
    fig,ax = plt.subplots(1,2,figsize=(20,10))
    
    plt.subplot(121)
    plot_relative_trajectories(steps_sep=steps_sep)
    if method_name: plt.title('Relative trajectories : h={} , steps={}\nmethod={}  proj={}'.format(H,STEPS,method_name,projection_type),fontsize=17)
    else: plt.title('Relative trajectories : h={} , steps={}'.format(H,STEPS),fontsize=17)
    
    plt.subplot(122)
    plot_relative_velocities(steps_sep=steps_sep)
    if method_name: plt.title('Relative velocities : h={} , steps={}\nmethod={}  proj={}'.format(H,STEPS,method_name,projection_type),fontsize=17)
    else: plt.title('Relative velocities : h={} , steps={}'.format(H,STEPS),fontsize=17)
    
    plt.show()
    
def plot_relative_trajectories(steps_sep=None):
    global position_arr
    position_arr = np.array(position_arr)
    m1x = position_arr.T[0][0]
    m2x = position_arr.T[0][1]
    m1y = position_arr.T[1][0]
    m2y = position_arr.T[1][1]
    relative_position = [m1x-m2x , m1y-m2y]
    
    cutoff=None
    if isinstance(steps_sep,tuple):# can take steps_sep as rational number quotien like (1,4) is one quarter
        cutoff = int(len(position_arr)*steps_sep[0]/steps_sep[1])
    elif isinstance(steps_sep,int):
        cutoff = steps_sep
    elif isinstance(steps_sep,float):# should be < 1
        cutoff = int(steps_sep*len(position_arr))
        
    if cutoff:
        plt.plot(relative_position[0][:cutoff],relative_position[1][:cutoff],'-',alpha=.7,linewidth=0.5,label='first part')
        plt.plot(relative_position[0][cutoff:],relative_position[1][cutoff:],'-',alpha=.7,linewidth=0.5,label='latter part')
        plt.legend()
    else:
        plt.plot(relative_position[0],relative_position[1],'-',alpha=.7,linewidth=0.5)
    return
    
def plot_relative_velocities(steps_sep=None):
    global velocity_arr
    velocity_arr = np.array(velocity_arr)
    m1vx = velocity_arr.T[0][0]
    m2vx = velocity_arr.T[0][1]
    m1vy = velocity_arr.T[1][0]
    m2vy = velocity_arr.T[1][1]
    relative_velocity = [m1vx-m2vx , m1vy-m2vy]
    
    cutoff=None
    if isinstance(steps_sep,tuple):
        cutoff = int(len(velocity_arr)*steps_sep[0]/steps_sep[1])
    elif isinstance(steps_sep,int):
        cutoff = steps_sep
    elif isinstance(steps_sep,float):
        cutoff = int(steps_sep*len(velocity_arr))
    
    if cutoff:
        plt.plot(relative_velocity[0][:cutoff],relative_velocity[1][:cutoff],'-',alpha=.7,linewidth=0.5,label='first part')
        plt.plot(relative_velocity[0][cutoff:],relative_velocity[1][cutoff:],'-',alpha=.7,linewidth=0.5,label='latter part')
        plt.legend() 
    else:
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
    fig,ax = plt.subplots(3,2,figsize=(12,15))
    plt.subplot(321)
    plt.plot(time_arr,energy_arr,label='energy')
    # # scale the radius and do a comparison
    # min_e = min(energy_arr)
    # radius_arr_scaled = np.array(radius_arr) * (max(energy_arr)-min_e)/max(radius_arr) + min_e*np.ones(len(radius_arr))
    # plt.plot(time_arr,radius_arr_scaled,label='radius (scaled)')
    
    plt.xlabel('time')
    plt.ylabel('net energy of the system')    
    plt.title('energy\nh={} , steps={}'.format(H,STEPS),fontsize=14)
    plt.legend(fontsize=14)
    
    plt.subplot(322)
    plt.plot(time_arr,ang_momentum_arr,label='angular momentum')
    plt.title('angular momentum',fontsize=14)
    plt.xlabel('time',fontsize=14)
    plt.ylabel('net ang momentum of system',fontsize=10)
    plt.legend()
    
    plt.subplot(323)
    plt.plot(time_arr,linear_momentum_arr,label='net linear moment')
    plt.plot(time_arr,net_lin_mom_x_arr,label='lin mom x')
    plt.plot(time_arr,net_lin_mom_y_arr,label='lin mom y')
    try:
        plt.title('Linear Momentum\nMethod : {}'.format(method.__name__),fontsize=14)
    except:
        plt.title('Linear Momentum',fontsize=14)
    plt.xlabel('time',fontsize=12)
    plt.ylabel('Linear Momentum',fontsize=14)
    plt.legend()
    
    plt.subplot(324)
    plot_relative_trajectories(steps_sep=0.5)
    # edited to use projection name, global variable
    try:
        plt.title('Relative trajectories\nProjection : {}'.format(projection_name),fontsize=14)
    except:
        plt.title('Relative trajectories',fontsize=14)
    plt.xlabel('relative position x',fontsize=12)
    plt.ylabel('relative position y',fontsize=12)
    
    plt.subplot(325)
    plt.plot(time_arr,k_factor_arr)
    plt.title('k factor\nmeasures the eccentricity, zero for circular',fontsize=14)
    plt.xlabel('time',fontsize=14)
    plt.ylabel('net linear momentum y')
    
    plt.subplot(326)
    plot_invarient_level_set_angles(["energy","angular_m"])
    
    plt.tight_layout() 
    




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

def display_relative_velocity():
    return

# under construction 
def define_starting_momentum(E,L,q1,q2,p1):
    global Y0
    # solve ang momentum for p21 and sub into energy eq 
    # a = np.linalg.norm()
    return
        


# %% main

y = Y0[:]
projection_dic = {"None":None,
               "Naive":proj.naive_projection,
               "Iterated":proj.iterated_projection,
               "Parallel":proj.parallel_projection,
               "Standard":proj.standard_projection} 

method_dic = {"exp euler":integrators.exp_euler,
              "stromer verlet":integrators.stromer_verlet}

proj_dic = {"no proj":[()],
                 "energy":[(get_energy,nabla_H)],
                 "ang mom":[(get_total_angular_momentum,nabla_l)],
                 "both":[(get_energy,nabla_H),(get_total_angular_momentum,nabla_l)]}


# method = integrators.fourth_order_kutta 
# method = integrators.exp_euler 
# method = integrators.exp_trapezium 
# method = integrators.exp_midpoint 
# method = integrators.syplectic_euler 
# method = integrators.stromer_verlet 
# method = integrators.slow_modified_flow_basic_order2_no_proj 
# method = integrators.modified_energy_proj_order2

if __name__=="__main__":
    # REGULAR OLD TESTING THE INTEGRATORS
    k = 10
    method = integrators.exp_euler
    for i in range(STEPS):
        y = method(y)
        if i%k==0:
            time += H*k
            update_dta(y)
            
    display_invarients()
    plt.show()

# if __name__ == "__main__":
#     # TESTING THE MODIFIED EQUATIONS
#     # FIRST INTEGRATE USING NORMAL INTEGRATOR
#     proj_man_name = "energy"
#     proj_manifolds = proj_dic[proj_man_name]
    
    
#     # choose the integration scheme 
#     method_name = "exp euler" 
#     method = method_dic[method_name]
    
#     # choose projection type if any 
#     projection_name = "Naive" 
#     projection = projection_dic[projection_name]
    
#     k=10 # number of steps before update (for display purposes) 
    
    
    
#     # # integrate in STEPS iterations 
#     # for i in range(STEPS): 
#     #     y = method(y) 
        
#     #     # projection step
#     #     if projection_name!="None":
#     #         y = projection(y,first_integrals = proj_dic[proj_man_name])
            
#     #     # update every k steps 
#     #     if i%k==0:
#     #         time+=H*k
#     #         update_dta(y)
        
#     # print("overflow occurence : {}".format(overflow_count)) 
    
#     # display_invarients()
#     # # plt.savefig("./figures/gallary/invarients_configx_{}_{}_h={}_STEPS={}.png".format(method.__name__,projection_name,H,STEPS))
#     # plt.show() 
        

#     # INTEGRATE USING MODIFIED EQUATION, with step-size R times smaller
#     # R = 1
#     # h = H 
#     # H = int(H/R)
#     # STEPS = int(R*STEPS)
#     # reset_dta()
    
#     h = 0.01
    
#     for i in range(STEPS):
#         y = integrators.fourth_order_kutta_modified_flow(y , d2_name = method_name,
#                                             h=h, flow_name = proj_man_name)
        
#         # update every k steps 
#         if i%k==0:
#             time+=H*k
#             update_dta(y)
            
            
#     display_invarients()
#     # plt.savefig("./figures/gallary/invarients_configx_{}_{}_h={}_STEPS={}.png".format(method.__name__,projection_name,H,STEPS))
#     plt.show()
    
    
    
    
    
    
    
#     # DISPLAY
#     # before displaying you may want to take off some of the data from the trajcetories so that you can see what the 'stable orbit' looks like
#     # CHUCK = 300000
#     # try:
#     #     position_arr = position_arr[CHUCK:]
#     # except:
#     #     print("position array too small to chuck anything, not gonna do it")
    
#     # display_relative(steps_sep=0.5,method_name = method.__name__,projection_type=projection_name)
#     # display_total_energy() 








