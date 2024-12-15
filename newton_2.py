# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 17:58:53 2023

@author: andre
"""
import math
import numpy as np
from numpy import linalg as la

t = 10
L = 64

def f0(x, t, L):
    return x[0] * (math.exp(-x[1] * t) + math.exp(-x[1] * (L - t))) + x[2] * (-1)**t * (math.exp(-x[3] * t) + math.exp(-x[3] * (L - t)))

def f(x):
    y = np.zeros((4, 1))
    y[0] = f0(x, t + 0, L) - 0.2047 * 10**3
    y[1] = f0(x, t + 1, L) - 0.1473 * 10**3
    y[2] = f0(x, t + 2, L) - 0.1059 * 10**3
    y[3] = f0(x, t + 3, L) - 0.7634 * 10**2
    return y

def df(x):
    y = np.zeros((4,4))
    for i in range (4):  
        y[i][0] = math.exp(-x[1] * (t + i)) + math.exp(-x[1] * (L - (t + i)))
        y[i][1] = - x[0] * ((t+i) * math.exp(-x[1] * (t + i)) + (L - (t + i)) * math.exp(-x[1] * (L - (t + i))))
        y[i][2] = (-1)**(t + i) * (math.exp(-x[3] * (t + i)) + math.exp(-x[3] * (L - (t + i))))
        y[i][3] = - x[2] * (-1)**(t+i) * ((t + i) * math.exp(-x[3] * (t + i)) + (L - (t + i)) * math.exp(-x[3] * (L - (t + i))))
    return y

def newtons_method(
    x0,               # The initial guess
    f,                # The function whose root we are trying to find
    df,               # The Jacobian of the function
    tolerance,        # Stop when iterations change by less than this
    a1,               # Right side of interval
    a2,               # Left side of interval
    max_iterations,   # The maximum number of iterations to compute
    ):
    ff = np.zeros(4)
    x1 = x0
    for i in range(max_iterations):
        
        if la.det(df(x0))==0:                    # Avoiding division by zero
            return "error"
        
        ff = np.matmul(la.inv(df(x0)),f(x0))     # Computing part of newton's method
        
        for i in range (4):                      # Do Newton's computation
            x1[i] = x0[i] - ff[i]

            
        if la.norm(f(x0)) <= tolerance :         # Stop when the result is within the desired tolerance
            q = True
            for i in range(4):
                q = q and (a1[i] < x1[i] < a2[i]) # Checking interval 
            if q:
                return x1                         # x1 is a solution within tolerance
            

        for i in range (4):                     # Update x0 to start the process again
            x0[i] = x1[i]
        
        
    return None



a1 = np.array([0.5 * 10**4, 0.1, 0.1 * 10**8, 0.1])     # Set initial parametrs
a2 = np.array([0.6 * 10**4, 0.5, 0.5 * 10**8, 5])
x0 = np.array([0.52 * 10**4, 0.18, 0.18 * 10**8, 1.05])
x1 = np.array([0,0,0,0])

print(df(x0))

print("Root is ", newtons_method(x0, f, df, 0.4, a1, a2, 10))   # Print result

# n = 10

# for i in range(n):
#     #print(1)
#     print(x0)
#     newtons_method(x0, x1, f, df, 10**(-8), a1, a2, 100000)
#     for j in range(4):
#         x0[j] = a1[j] + i * (a2[j] - a1[j])/n
#         #print((a2[3] - a1[3])/n)
        
