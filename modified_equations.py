#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 13:54:39 2020

@author: steve
"""
# %% IMPORTS
import numpy as np
import itertools



# %%ALGORITHMS FOR FINDING MODIFIED EQUATIONS

# gamma(n) is the sequence representing the coefs of the function g in the n'th total derivative of g wrt time
# g(\tilde y) = \dot {\tilde y} = f(y) + hf_2(y) + h^2f_3(y) + O(h^3)
# formati is [[multiplicity , (tuple representing coefs of derivative of g wrt y tilde)] , [] , [] , ...]
# Examples gamma1 = [[1,np.array([1])]] , gamma2 = [[1,np.array([1,1])]] , gamma3 = [[1,(np.array([2,0,1])],[1,np.array([1,2])]] , etc.
def next_gamma(gamma):
    gamma_next = []
    for mult,term in gamma:
        for idx,element in enumerate(term):
            if element==0: 
                continue
            # calculate the multiplicity of the new term
            new_mult = mult*element
            # calculate the new term array representation
            if idx==(-1)%len(term):
                new_term = np.concatenate([np.copy(term),np.array([0])])
                added_term = np.zeros(len(term)+1,dtype=int)
            else:
                new_term = np.copy(term)
                added_term = np.zeros(len(term),dtype=int)
            added_term[0]+=1
            added_term[idx]-=1
            added_term[idx+1]+=1
            new_term += added_term 
            gamma_next.append([new_mult,new_term])
    # now we have obtained a correct (simplified) form of gamma next we simplify it by adding multiplicities of like arrays
    gamma_next_simpl = []
    while gamma_next:
        mult,term = gamma_next[0]
        gamma_reduced = gamma_next[1:]
        print()
        for m,t in gamma_next[1:]:
            if t.shape==term.shape and all(t==term):
                mult+=m
                gamma_reduced.remove([m,t])
        gamma_next_simpl.append([mult,term]) 
        gamma_next = gamma_reduced # reduce gamma
    return gamma_next_simpl 
        

# we represent g^(n) by an array of strings
# g^{(n)} = [[0,["f1n"]] , [1,["f2n"]] , [2,["f3n"]],...] means f^(n) + hf_2^(n) + h^3f_3^(n) + O(h^4)
# the format foe each element is [power of h , [list of string representations of nth derivative of functions f_i(y tilde)]]
def convolve(arrays):
    convolved = arrays[0]
    arrays = arrays[1:]
    while arrays:
        conv_step = itertools.product(convolved,arrays[0])
        conv_next = []
        for i,j in conv_step:
            # add h order coefs and concatenate the lists
            conv_next.append([i[0] + j[0] , i[1] + j[1]])
        convolved = conv_next 
        arrays = arrays[1:]
    # put it in order so that it's human redable 
    convolved.sort()
    return convolved
        
        
    
            

# %% KEPLER




# %% N BODY



# %% TESTING THE ALGORITHMS

# TEST convolve
if __name__ == "__main__":
    g1 = [[0,["a"]],[1,["b"]]]
    g2 = [[0,["r"]],[2,["s"]]]
    conv = convolve([g1,g2])
    print(conv)


# # TEST next_gamma
# if __name__=="__main__":
#     gamma = [[1,np.array([1])]]
#     gammas = [gamma]
#     for i in range(4):
#         gamma = next_gamma(gamma)
#         gammas.append(gamma)
#     [print(gamma,end="\n\n") for gamma in gammas] 
