#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 12:04:45 2020

@author: steve
"""

import numpy as np
from kep_derivative_functions import nabla_lin_y,nabla_lin_x,nabla_l,nabla_H
from kep_util_globvar import Y0
from kep_util_globvar import supervec_dot,supervec_norm,to_supervec
from kep_util_globvar import get_energy,get_total_angular_momentum,get_lin_mom_x,get_lin_mom_y



# %% PROJECTIONS


# naive should be same as other if manifold tanjent spaces are perpendicular 
# this method can do with some optimization (/ refactoring) 
def naive_projection(y,first_integrals=[(get_energy,nabla_H),
                                        (get_total_angular_momentum,nabla_l)]):
    y_proj=y[:]
    # apply each projection one after the other 
    for i,j in first_integrals:
        fint,nabla_fint = i,j(y_proj) 
        lambda_i = fint(Y0) - fint(y_proj) 
        lambda_i /= supervec_norm(nabla_fint)
        y_proj = y_proj + lambda_i*nabla_fint 
        
    return y_proj # since y is an array you could have written this so that you're directly operating on it, it's a pointer remember this stuff...
        
# same as naive but does it in k steps (is also k times slower!), also it's only effective if 
# there are multiple first integrals, no point if there is just one 
def iterated_projection(y,k=5,first_integrals=[(get_energy,nabla_H)]):
    y_proj = y[:]
    # apply k times 
    for l in range(k,0,-1):
        # apply each projection one after the other 
        for i,j in first_integrals: 
            fint,nabla_fint = i,j(y_proj) 
            lambda_i = fint(Y0) - fint(y_proj) 
            lambda_i /= supervec_norm(nabla_fint) 
            y_proj = y_proj + (lambda_i / l) * nabla_fint 
    
    return y_proj


def parallel_projection(y,first_integrals=[(get_energy,nabla_H),
                                                (get_lin_mom_x,nabla_lin_x),
                                                (get_lin_mom_y,nabla_lin_y)]):
    global overflow_count
    
    # first compute the gradients
    fivag = [(i(y),i(Y0),j(y)) for i,j in first_integrals]
    finty,fint0,beta = [j for i,j,k in fivag],[i for i,j,k in fivag],[k for i,j,k in fivag]
    beta_prime = [j[:] for j in beta] # copy of the gradients to be modified
    
    
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
    for e0,ey,b_i_p,b_i in zip(fint0,finty,beta_prime,beta):
        try:
            lambda_i = -(e0 - ey) / supervec_dot(b_i_p,b_i)
            y_proj += lambda_i * b_i_p 
        except OverflowError("overflow in computing lambda_i"):
            overflow_count+=1
            
    return y_proj


def standard_projection(y,k=2,first_integrals=[(get_energy,nabla_H),
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
    
