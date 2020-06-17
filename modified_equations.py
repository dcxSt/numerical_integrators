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
# Examples gamma1 = [[1,np.array([1])]] , gamma2 = [[1,np.array([1,1])]] , 
# gamma3 = [[1,np.array([2,0,1])],[1,np.array([1,2])]] , etc.
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
# g^{(n)} = [[0,["f1,n"]] , [1,["f2,n"]] , [2,["f3,n"]],...] means f^(n) + hf_2^(n) + h^3f_3^(n) + O(h^4)
# the format foe each element is [power of h , [list of string representations of nth derivative of functions f_i(y tilde)]]
# takes list arrays, contains arrays of pihsr s to be convolved with each other
# returns covolution pishr 
def convolve_pihsr(arrays): 
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

# returns string repr nth derivative of g to kth order 
def get_gn_str(k,n):
    g = []
    for i in range(k+1):
        g.append([i , ["f{},{}".format(i+1,n)]])
    return g


# takes a term (nth gamma list) from taylor exp of modified equation \tilde y
# in terms of g and returns it in terms of f strings 
# pihsr stands for polynomial in h string represent of g and returns it in terms of f strings 
# gamma is in format specified in next_gamma up top
# o is the order
def gamma_to_pihsr(gamma,o=5):
    # first, find which is the highest power of g which you have
    # power = max([len(j) for i,j in gamma])
    
    pihsr = []
    # for each term in gamma
    for coef,term in gamma:
        # compute all the pihsr_i's belonging to that term and add it to the list
        # convolve the pihsr representations of this term with each other
        
        gs_list = []
        for n,power in enumerate(term):
            for i in range(power):
                gs_list.append(get_gn_str(k=o,n=n))
        pihsr_terms = convolve_pihsr(gs_list)
        # now add each of these terms to the list
        for i in pihsr_terms:
            pihsr.append(i)
    
    # drop the order higher than o terms
    pihsr = [i for i in pihsr if i[0]<= o]
    pihsr.sort() 
    return pihsr
    
# takes a long term (written in terms of the f strings lists) and returns all order n terms
# pihsr stands for polynomial in h string representation, it's the format
def get_order_n_terms(pihsr,n): 
    order_n_terms_pihsr = []
    for i,j in pihsr:
        if i==n:
            order_n_terms_pihsr.append([i,j])
    return order_n_terms_pihsr
        
# shows the polynomial in human form
def display_pihsr(pihsr):
    for i in range(max([i for i,j in pihsr])+1):
        print("\nh^{} x (".format(i),end="  ")
        for o,fk in [[l,m] for l,m in pihsr if l==i]:
            mystring = ""
            for f in fk:
                split = str.split(f,",")
                mystring += "f_{}^{}".format(split[0][1:] , split[1])
                mystring += " "
            print(mystring,end="  +  ")
        print(" )")
    # ignore the plus at the end, it's not worth our time
    return
            

# %% KEPLER




# %% N BODY



# %% TESTING THE ALGORITHMS
    
# Test get_gn_str and gamma_to_pihsr
if __name__=="__main__":
    gamma = [[1,np.array([1])]]
    for i in range(5):
        gamma=next_gamma(gamma)
    
    pihsr = gamma_to_pihsr(gamma,o=3)
    # print("gamma : ",gamma)
    # print("pihsr : ",pihsr)
    display_pihsr(pihsr) 

# # TEST convolve
# if __name__ == "__main__":
#     g1 = [[0,["a"]],[1,["b"]]]
#     g2 = [[0,["r"]],[2,["s"]]]
#     conv = convolve_pihsr([g1,g2])
#     conv = convolve_pihsr([conv,g1])
#     print(conv)


# # TEST next_gamma
# if __name__=="__main__":
#     gamma = [[1,np.array([1])]]
#     gammas = [gamma]
#     for i in range(4):
#         gamma = next_gamma(gamma)
#         gammas.append(gamma)
#     [print(gamma,end="\n\n") for gamma in gammas] 
