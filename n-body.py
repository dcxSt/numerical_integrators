#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 01:17:07 2020

@author: steve
"""
# N BODY PROBLEM

# %% IMPORTS 
import numpy as np
import matplotlib.pyplot as plt

# %% CLASS SYSTEM 
class System:
    # Class attributes
    G = 1
    H = 0.001
    STEPS = 50000
    # canonical symplectic form
    J = np.array([[0,0,0,-1,0,0],
                   [0,0,0,0,-1,0],
                   [0,0,0,0,0,-1],
                   [1,0,0,0,0,0],
                   [0,1,0,0,0,0],
                   [0,0,1,0,0,0]])
    
    def __init__(self,Y0,masses,names=None):
        # Object Attributes
        self.Y0 = Y0 # 2d array
        self.Y = Y0[:]
        self.masses = masses # list of floats
        self.trace = None # gets made from planet traces
        self.energies = []
        #self.trace = [np.array(Y0[:])] 
        
        #Y0 in matrix form, masses is a list of masses 
        if len(Y0) != len(masses): 
            raise ValueError("It seems like you starting positions and number of planets don't match") 
        
        
        if not names: 
            names = [i for i in range(len(masses))] 
            
        # make all the planets 
        self.planets = [] 
        for y0,m,name in zip(Y0,masses,names): 
            # initiate a pointmass, or planet at position y0 
            self.planets.append(Planet(y0,m,name)) 
            
            
    # GETTERS
    # INVARIENTS AND SCALAR QUANTITIES 
    def get_energy(self):
        energy = sum([planet.get_energy(self.planets) for planet in self.planets])
        return energy
    
    def update_energies(self):
        self.energies.append(self.get_energy())
    
    # return a 3-vector
    def get_net_lin_momentum(self):
        return sum([i[:3] for i in self.Y])
    
    def get_angular_momentum(self):
        return sum([planet.get_ang_mom() for planet in self.planets])
    
    # GRADIENTS AND VECTOR QUANTITIES
    def get_nabla_H(self):
        # calculate nabla h for each of the planets and put them into a matrix (list of arrays)
        return np.array([p.nabla_H(self.planets) for p in self.planets])
    
    # returns a 3 by n matrix and compute the nabla 
    def nabla_q(self,q_matrix=None):
        if not isinstance(q_matrix,np.ndarray) :
            return np.array([p.nabla_q(self.planets) for p in self.planets])
        # if a q_matrix is given calculate as if the planets where in the positions specified by it
        else:
            sh = q_matrix.shape
            pot_grad = np.zeros(sh[0]*sh[1]).reshape(sh)
            for idx,(planet_1,q1) in enumerate(zip(self.planets,q_matrix)):
                for planet_2,q2 in zip(self.planets,q_matrix):
                    if planet_1 != planet_2:
                        pot_grad[idx] = pot_grad[idx] - System.G * planet_1.m * planet_2.m / np.linalg.norm(planet_1.radius(planet_2))**3 * planet_1.radius(planet_2)
            return pot_grad
    
    def get_flow_derivative(self): 
        fd = np.array([planet.flow_derivative(self.planets) for planet in self.planets]) 
        return fd 
    
    # returns nabla l x and y and z, each is it's own matrix
    def nabla_l_xyz(self):
        nabla_lx = np.array([p.nabla_lx() for p in self.planets])
        nabla_ly = np.array([p.nabla_ly() for p in self.planets])
        nabla_lz = np.array([p.nabla_lz() for p in self.planets])
        return nabla_lx,nabla_ly,nabla_lz 
    
    # AND SETTERS 
    # add a planet to the system 
    def add_planet(self,y0,m,name=None): 
        if len(y0)!=6 or len(m)!=6 : raise ValueError 
        
        self.Y0.append(y0) 
        self.masses.append(m) 
        return 
    
    def remove_planet(self,name):
        for planet in self.planets:
            if name == planet.name:
                self.planets.remove(planet)
                return 
        print("Planet {} doesn't exist, check for typos".format(name))
        return
            
    # updates meta information for each planet after one integration numerical flow step
    def update_planets(self):
        for planet,y_next in zip(self.planets,self.Y):
            planet.update_y(y_next) 
        return 
            
    # takes the traces y componants of each planet and makes a big matrix trace for the whole system
    def make_trace(self):
        print('Calculating Trace...',end="\t")
        self.trace = [] 
        for planet in self.planets:
            self.trace.append(planet.trace)
        self.trace = np.array(self.trace) 
        self.trace = np.transpose(self.trace,axes=[1,0,2]) 
        print('Done')
        return
        
    # NUMERICAL FLOWS 
        
    # takes y_n return y_n+1
    def explicit_euler_timestep(self): 
        flow_derivative = self.get_flow_derivative()
        Y_next = np.array(self.Y) + System.H * flow_derivative 
        self.Y = Y_next
        return
    
    # stomer verlet timestep
    # assumes that Y and each planet's y are in agreement
    def stromer_verlet_timestep(self):
        flow_derivative = self.get_flow_derivative() # 6 by n matrix 
        # STEP 1 
        x0 = self.Y.T[:3].T # trace 
        p_plushalf = self.Y.T[:3].T + 0.5 * System.H * flow_derivative.T[:3].T
        v_plushalf = np.array([i/p.m for i,p in zip(p_plushalf,self.planets)])
        # STEP 2 
        x1 = self.Y.T[3:].T # trace 
        q_next = self.Y.T[3:].T + System.H * v_plushalf 
        # STEP 3 
        nabla_q_next = self.nabla_q(q_next)
        p_next = p_plushalf - 0.5 * System.H * nabla_q_next
        x3 = self.Y # trace 
        self.Y = np.concatenate([p_next,q_next],axis=1)
        x4=self.Y # trace 
        return
        
    
    # DISPLAY
    def display_xy_projection(self):
        plt.figure(figsize=(10,10))
        for planet in self.planets:
            planet.display_xy_projection()
        plt.legend()
        plt.title("xy projection of trajectories",fontsize=16)
        plt.show()
        return
    
    def display_projections_and_energy(self):
        plt.subplots(2,2,figsize=(10,10))
        plt.subplot(221)
        for planet in self.planets:
            planet.display_xy_projection()
        plt.legend()
        plt.title("xy plane trajectories",fontsize=16)
        plt.xlabel("x")
        plt.ylabel("y")
        
        plt.subplot(222)
        for planet in self.planets:
            planet.display_yz_projection()
        plt.legend()
        plt.title("yz plane trajectories")
        plt.xlabel("y")
        plt.ylabel("z")
        
        plt.subplot(223)
        for planet in self.planets:
            planet.display_zx_projection()
        plt.legend()
        plt.title("zx plane trajectories")
        plt.xlabel("z")
        plt.ylabel("x")
        
        plt.subplot(224)
        # later we can add these two lines if this might be intereting, dunno if it's interesting yet...
        # for planet in self.planets:
        #     planet.display_energy()
        plt.plot(self.energies)
        plt.title("total energy of system")
        plt.xlabel("Time in steps of {}".format(System.H))
        plt.ylabel("Energy")
        
        
        plt.show()
    
    # UTILITY 
    def cm_frame(Y,masses):
        # returns Y in the CM frame
        px,py,pz = Y.T[0],Y.T[1],Y.T[2]
        pnetx,pnety,pnetz = sum(px),sum(py),sum(pz)
        v_cm = np.array([pnetx,pnety,pnetz]) / sum(masses)
        v_cm_arr = np.concatenate([v_cm , [0.0,0.0,0.0]])
        # take away the cm velocity from each velocity 
        Y_new = []
        for y,m in zip(Y,masses):
            y_new = y - v_cm_arr * m 
            Y_new.append(y_new) 
        return np.array(Y_new)
        

# %% CLASS PLANET 
class Planet:
    # class attributes
    
    def __init__(self,y0,m,name):
        # check that y0, m are correct
        if len(y0)!=6: raise ValueError 
        
        # initialize all the object attributes
        self.m = m
        self.y0 = y0
        self.y = y0[:]
        self.name = name
        self.trace = [y0]
    
    # GETTERS
    def get_kinetic_energy(self):
        return np.dot(self.y[:3],self.y[:3])/ (2*self.m)
    
    def get_potential_energy(self,planets):
        potential = 0
        for planet in planets:
            if self != planet:
                potential -= System.G * planet.m * self.m / np.linalg.norm(self.radius(planet))
        return potential
    
    def get_energy(self,planets):
        return self.get_kinetic_energy() + self.get_potential_energy(planets) 
    
    def get_position(self):
        return self.y[3:]
    
    def get_ang_mom(self):
        return np.linalg.cross(self.y[:3],self.y[3:])
    
    def radius(self,planet):
        # returns the vector from this planet pointing to the planet planet
        rad =  planet.y[3:] - self.y[3:] 
        if np.all(rad == 0.0):
            raise ZeroDivisionError("The planets self and planet seem to be super-imposed")
        return rad 
        
    # return arr of len 6
    def nabla_H(self,planets):
        # nabla_H of the momentum is just the same
        nabla_p = self.y[:3] / self.m # = p / m = v = \dot x
        pot_grad = np.array([0.0,0.0,0.0])
        for planet in planets:
            if self != planet:
                pot_grad = pot_grad - System.G * planet.m * self.m / np.linalg.norm(self.radius(planet))**3 * self.radius(planet)
        return np.concatenate([nabla_p , pot_grad])
    
    # derivative only of q
    def nabla_q(self,planets):
        pot_grad = np.array([0.0,0.0,0.0])
        for planet in planets:
            if self != planet:
                pot_grad = pot_grad - System.G * planet.m * self.m / np.linalg.norm(self.radius(planet))**3 * self.radius(planet)
        return pot_grad
        
    def flow_derivative(self,planets):
        nabla_h = self.nabla_H(planets)
        return np.concatenate([-nabla_h[3:] , nabla_h[:3]])
    
    # returns the angular momentum gradient vectors
    def nabla_lx(self):
        return np.array([0,self.y[5],-self.y[4],0,-self.y[2],self.y[1]])
    def nabla_ly(self):
        return np.array([-self.y[5],0,self.y[3],self.y[2],0,-self.y[0]])
    def nable_lz(self):
        return np.array([self.y[4],-self.y[3],0,-self.y[0],self.y[0],0])
    
    
    # MODIFIERS 
    def update_y(self,y_next): 
        self.y = y_next 
        self.trace.append(y_next) 
        return 
    
    # DISPLAY
    def display_xy_projection(self):
        x,y = np.array(self.trace).T[3:5]
        plt.plot(x,y,'-',label=self.name)
        
    def display_yz_projection(self):
        y,z = np.array(self.trace).T[4:]
        plt.plot(y,z,'-',label=self.name)
        
    def display_zx_projection(self): 
        x = np.array(self.trace).T[3] 
        z = np.array(self.trace).T[5] 
        plt.plot(z,x,'-',label=self.name) 
    
    # UTILITY METHODS 
        
# %%
if __name__=='__main__': 
    # initialize the system with the sun + 3 planets
    sun,msun = np.array([0.0,0.0,0.0,0.0,0.0,0.0]),100.0 
    jupyter,mjup = np.array([7.0,-0.0,0.0,0.0,2.0,0.0]),1.0 
    saturn,msat = np.array([-7.2,0.0,0.0,-0.0,-4.2,0.0]),1.4
    uranus,mura = np.array([0.0,3.1,0.01,12.0,0.0,-0.05]),1.1
    
    Y0 = np.array([sun,jupyter,saturn,uranus]) # list of arrays 
    masses = [msun,mjup,msat,mura]
    
    # put into cm frame 
    Y0 = System.cm_frame(Y0,masses)
    
    # initialize the solar system object 
    solar = System(Y0,masses,names=['Sun','Jupiter','Saturn','Uranus']) 
    
    # time evolve the system a bunch then display 2d projections of trajectories
    for i in range(System.STEPS): 
        solar.stromer_verlet_timestep() 
        # solar.naive_first_integral_projection(first_integrals=[(get_energy,get_nabla_H),
        #                                                        (get_angular_momentum,nabla_l))]) 
        solar.update_energies()
        solar.update_planets() 
        
    print("\nFinished integrating, now computing trace and invariants.") 
    
    solar.make_trace() 
    
    # display the trajectories projected onto 2d 
    solar.display_projections_and_energy() 
    
    
    



    