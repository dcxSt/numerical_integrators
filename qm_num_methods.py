#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 12:44:42 2020

@author: steve
"""
# NUMERICAL SOLUTION TO FINITE DIFFERENCE EQUATION FOR EIGNENVALUE EQUATION
# for reference see 8.1 and 8.2 in Marchildon QM book
"""
a particle's eigenvalue equation is given by 

\frac{\hat p}{2m} \ket\phi + \hat V\ket\phi = E \ket\phi

written in position basis (left multiply by the position bra / duel) 
and chosing a invers square potential with coeff -a^2 we obtain 
(rem: E should be negative becuase we are in bound state)

\frac{-\hbar^2}{2m}\frac{d^2}{dx^2}\phi(x) - (\frac{a^2}{x^2} + E)\phi(x) = 0 
"""

# define the size of delta
DELTA = 0.01
# the upper and lower bounds in terms of delta
NG,ND = -50,50

# define the function xi which returns descrete thing...
def get_xi(n):
    # n is an integer
    return n*DELTA

# the potential
def get_v(x):
    if x!=0:
        return 1/x**2
    raise Exception("Every time you divide by zero a puppy dies")

# return the non free particle term, called h
def get_h(x):
    return 

# second order approx
def next_2nd_order(F1,F2,n):
    hn = get_h(DELTA*N)
    F3 = (2 - DELTA**2 hn)*F2 - F1
    return F3


