#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 14:08:55 2022

@author: Rob Klock (adapted from Pat Simen)
"""

"""
Spiker Generator
simulates a two unit neural network that can generate a
Poisson-like sequence of spikes using Euler-Maruyama simulation
"""
import numpy as np
import math 
import matplotlib.pyplot as plt

num_steps = 100000

W = np.asarray([[2, -0.75], [1,2]])
V = np.zeros([2,1])
lambda_ = np.asarray([[4],[4]])
beta = np.asarray([[0.8],[1.3]])
noise = np.asarray([[0.1], [0.1]])
dt = 0.01

V_hist = np.full([2, num_steps], V)

time_history = np.empty([1, num_steps])
time_history[0] = 0

external_input = 0.2 # or 0, or -.1

for step in range(1, num_steps):
    net_input = np.matmul(W,V)
    net_input[0] = net_input[0] + external_input
   
    dV = -V + 1.0/(1.0 + np.exp(-lambda_ * (net_input - beta)))
    
    V = V + dV * dt+ noise * math.sqrt(dt) * np.random.normal(0, 1, 1)
   
    V_hist[0][step] = V[0]
    V_hist[1][step] = V[1]

    
time_history = np.linspace(0, num_steps * dt, num_steps)

plt.plot(time_history, V_hist[0])
plt.plot(time_history, V_hist[1])
    