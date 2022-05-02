#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 22:41:09 2022

@author: Rob Klock
Euler-method heuristic algorithm simulation
"""
import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import random
from timer_module import TimerModule as TM
from labellines import labelLine, labelLines
from scipy.stats import invgauss   
import matplotlib.colors as mcolors


def activationAtIntervalEnd(timer, ramp_index, interval_length, c):
    # Simulate DDM process for activation amount
    act = timer.timers[ramp_index] * interval_length
    for i in range (0, len(act)):
        act[i] = act[i] + c * np.sqrt(act[i]) * np.random.normal(0, 1) * math.sqrt(interval_length)
    return act

def generate_hit_time(weight, threshold, noise, dt, plot=False):   
    # Alternative method for hitting time
    T = int(((threshold/weight)+50)/dt)
    arr = np.random.normal(0,1,T) * noise * np.sqrt(dt)
    
    drift_arr = np.ones(T) * weight * dt
    act_arr = drift_arr + arr
    cum_act_arr = np.cumsum(act_arr)
   
    hit_time = np.argmax(cum_act_arr>threshold) 
    x = np.arange(0, T*dt, dt)
    
    # plot many trajectories over each other
    if plot:
       plt.figure()
       plt.hlines(threshold,0,T)
       plt.plot(x, cum_act_arr, color="grey")
       plt.xlim([0,hit_time + (hit_time//2)])
       plt.ylim([0, threshold + (threshold/2)])
       
    if hit_time > 0:    
        return [hit_time, cum_act_arr[:hit_time]]

def start_threshold_time(act_at_interval_end, interval_length):
    # Time of ramp hitting start threshold
    angle = np.arctan(act_at_interval_end/interval_length)
    beta = 3.14159 - (1.5708 + angle)
    return START_THRESHOLD * np.tan(3.14159 - (1.5708 + angle))

def stop_threshold_time(act_at_interval_end, interval_length):
    # Time of ramp hitting stop threshold
    angle = np.arctan(act_at_interval_end/interval_length)
    beta = 3.14159 - (1.5708 + angle)
    return STOP_THRESHOLD * np.tan(3.14159 - (1.5708 + angle))
    
def generate_responses(interval_length):
    num_samples = int(interval_length / dt)
    responses = np.random.exponential(4, num_samples)
    return responses
    
    

def lateUpdateRule(vt, timer_weight, learning_rate, v0=1.0, z = 1, bias = 1):
    """
    Parameters
    ----------
    v0: activation of IN unit
    z: desired activation, threshold
    Vt: timer unit activation
    bias: bias of timer unit
    timer_weight: the weight of the timer

    Returns 
    -------
    The corrected timer weight for the associated event

    """

    drift = (timer_weight * v0)
    d_A = drift * ((1-vt)/vt)
    ret_weight = timer_weight + (learning_rate * d_A)
    return ret_weight
    
def earlyUpdateRule(vt, timer_weight, learning_rate, v0=1.0, z = 1, bias = 1):
    """
    Parameters
    ----------
    v0: activation of IN unit
    z: desired activation, threshold
    Vt: timer unit activation
    bias: bias of timer unit
    timer_weight: the weight of the timer

    Returns 
    -------
    The corrected timer weight for the associated event

    """
    drift = (timer_weight * v0)
    d_A = drift * ((vt-z)/vt)
    ret_weight = timer_weight - (learning_rate * d_A)
    return ret_weight

def update_rule(timer_values, timer, timer_indices, start_time, end_time, event_type, v0=1.0, z = 1, bias = 1, plot = False):
    # Update rule without reassignment
    for idx, value in zip(timer_indices, timer_values):
        # Frozen timers arent updated
        if idx in timer.frozen_ramps:
            continue
        
        if value > 1:
            ''' Early Update Rule '''
            timer_weight = earlyUpdateRule(value, timer.timerWeight(idx), timer.learningRate(idx))
            plt.grid('on')

            if plot:
                plot_early_update_rule(start_time, end_time, timer_weight, T, event_type, value)
                    
            timer.setTimerWeight(timer_weight, idx)
            
        else:
            ''' Late Update Rule '''
            timer_weight = lateUpdateRule(value, timer.timerWeight(idx), timer.learningRate(idx))
            timer.setTimerWeight(timer_weight, idx)

def coin_flip_update_rule(timer_values, timer, timer_indices, start_time, end_time, stimulus_type, event_type, next_stimulus_type, v0=1.0, z = 1, bias = 1, plot = False):
    # Update rule with coin flip
    for idx, value in zip(timer_indices, timer_values):
        # Frozen timers arent updated
        if idx in timer.frozen_ramps:
            continue
        flip = random.random()
       
        if flip >=0:
            if int(stimulus_type) in timer.stimulusDict().keys() and (idx in timer.stimulusDict()[int(stimulus_type)]):
                if value > 1:
                    ''' Early Update Rule '''
                    #plot_early_update_rule(start_time, end_time, timer_weight, T, event_type, value)
                    timer_weight = earlyUpdateRule(value, timer.timerWeight(idx), timer.learningRate(idx))
                    plt.grid('on')
        
                    if plot:
                        plot_early_update_rule(start_time, end_time, timer_weight, T, event_type, value)
                            
                    timer.setTimerWeight(timer_weight, idx)
                    
                else:
                    ''' Late Update Rule '''
                    timer_weight = lateUpdateRule(value, timer.timerWeight(idx), timer.learningRate(idx))
                    timer.setTimerWeight(timer_weight, idx)

def respond(timer_value, event_time, next_event, ax1, idx):
    # Given all ramp vaues, respond when K are between start and stop range
    
    # Find start threshold times for each ramp
    start_threshold_times = start_threshold_time(timer_value, next_event-event_time)
    start_threshold_times += event_time
    start_threshold_times.sort()
    start_threshold_times = np.vstack((start_threshold_times, np.ones(len(start_threshold_times)))).T
    
    # Find stop threshold times for each ramp
    stop_threshold_times = stop_threshold_time(timer_value, next_event-event_time)
    stop_threshold_times += event_time
    stop_threshold_times.sort()
    stop_threshold_times = np.vstack((stop_threshold_times, (-1* np.ones(len(stop_threshold_times))))).T
    
    # Zip start and stop times
    start_stop_pairs = np.vstack((start_threshold_times, stop_threshold_times))
    start_stop_pairs = start_stop_pairs[start_stop_pairs[:, 0].argsort()]

    responses = []
    response_periods = []
    k = 0
    k_o = 0 # Old value of k
    
    # Form list of start and stop events, sorted by time (a1, sig1, a2, a3, sig2, sig3 etc)
    # Loop through all, if start event, k++, else, k--
    # Identify all periods of k > K
    # Fill with Poisson seq (samples then add the start time to all of them)
    # once theyre greater than the boundary where they stop, throw them out
    for jdx, time in enumerate(start_stop_pairs):
        k+=time[1]
        # print(f'k: {k} \t time: {time[0]}')
        # We're entering a response period
        if k_o < K and k >= K:
            response_period_start = time[0]
        if k_o >=K and k < K:
            response_period_end = time[0]
            response_periods.append([response_period_start, response_period_end])
        k_o+=time[1]
    r = list(generate_responses(next_event-event_time))
    r.insert(0, event_time)
    r=list(np.cumsum(r))
    
    for response_period in response_periods:
        responses.extend([i for i in r if (i>response_period[0] and i<response_period[1] and i<next_event and i>event_time)])
        # Debugging plots
        # ax1.vlines(response_period[0], 0,Y_LIM, color="green")
        # ax1.vlines(response_period[1], 0,Y_LIM, color="red")
        # ax1.text(response_period[0],1.5,str(idx))
        # ax1.text(response_period[1],1.5,str(idx))
    
    ax1.plot(responses, np.ones(len(responses)), 'x') 
    responses and ax1.text(responses[0],1.2,str(idx))
    return responses

def update_and_reassign_ramps(timer, timer_values, timer_indices, next_stimulus_type, stimulus_type, sequence_code = '', v0=1.0, z = 1, bias = 1, plot = False):
    # Frozen timers arent updated
    for idx, value in zip(timer_indices, timer_values):
        if idx in timer.frozen_ramps:
            continue
        
        # Generate coin flip for random update
        flip = random.random()
       
        """ 
        From all the ramps not idle who have either:
            - S2=e_i and start<act<stop
            - S2 = NA
        Pick N randomly and update them for the interval s1->s2=e_i
        """
        # If a timer is unassigned
        if timer.terminating_events[idx] == -1:
            if flip >=.75:
                # if the timer has the appropriate terminating event, update the weight
                if value > 1:
                    ''' Early Update Rule '''
                    #plot_early_update_rule(start_time, end_time, timer_weight, T, event_type, value)
                    timer_weight = earlyUpdateRule(value, timer.timerWeight(idx), timer.learningRate(idx))
                    plt.grid('on')
                    
                else:
                    ''' Late Update Rule '''
                    timer_weight = lateUpdateRule(value, timer.timerWeight(idx), timer.learningRate(idx))
                   
                if timer.initiating_events[idx] == stimulus_type and timer.terminating_events[idx] == next_stimulus_type:
                    if flip>=.9:
                        timer.setTimerWeight(timer_weight, idx)
                        timer.terminating_events[idx]= next_stimulus_type
               
                if idx in timer.free_ramps:
                    timer.setTimerWeight(timer_weight, idx)
                    timer.free_ramps = np.delete(timer.free_ramps, np.where(timer.free_ramps == idx))
                    timer.initiating_events[idx] = stimulus_type
                    timer.terminating_events[idx] = next_stimulus_type
        
        if timer.terminating_events[idx] == next_stimulus_type and timer.initiating_events[idx] == stimulus_type:
            if value > 1:
                    ''' Early Update Rule '''
                    #plot_early_update_rule(start_time, end_time, timer_weight, T, event_type, value)
                    timer_weight = earlyUpdateRule(value, timer.timerWeight(idx), timer.learningRate(idx))
                    plt.grid('on')
                    # timer.setTimerWeight(timer_weight, idx)
                    
            else:
                ''' Late Update Rule '''
                timer_weight = lateUpdateRule(value, timer.timerWeight(idx), timer.learningRate(idx))
            
def relative_to_absolute_event_time(relative_time_events):
   absolute_time_events = relative_time_events   
   for i in range (1,NUM_EVENTS):
       absolute_time_events[i][0] = relative_time_events[i-1][0] + relative_time_events[i][0]
   return absolute_time_events  

def sigmoid(l, total_input, b):
    f = 1/ (1 + np.exp(-l * (total_input - b)))
    return f

def piecewise_linear(v, bias):
    if ((v - (bias) + .5) <= 0):
        return 0
    elif (((v - (bias) + .5)> 0) and ((v - (bias) + .5) < 1)):
        return ((v - (bias) + .5))
    else:
        return 1
            
''' Global variables '''
dt = 0.1
N_EVENT_TYPES= 2 # Number of event types (think, stimulus A, stimulus B, ...)
# NUM_EVENTS=17#  Total amount of events
Y_LIM=2 # Vertical plotting limit
NOISE=0.0009 # Internal noise - timer activation
LEARNING_RATE=.8 # Default learning rate for timers
STANDARD_INTERVAL=20 # Standard interval duration 
K = 5 # Amount of timers that must be active to respond
START_THRESHOLD=.5 # Response start threshold
STOP_THRESHOLD=1.2 # Response stop threshold
PLOT_FREE_TIMERS=False
ERROR_ANALYSIS_RESPONSES=[]
colors = list(mcolors.TABLEAU_COLORS) # Color support for events
ALPHABET_ARR = ['A','B','C','D','E','F','G'] # For converting event types into letters 

# [Event Time, Event Type, Stimulus Type]
event_data = np.asarray([[0,1,1], [50,0,0], [25,1,1], [50,0,0], [25,1,1], [50,0,0], [25,1,1], [50,0,0], [25,1,1], [50,0,0], [25,1,1], [50,0,0], [25,1,1], [50,0,0], [25,1,1], [50,0,0], [25,1,1],
                               [50,0,0], [25,1,1], [50,0,0], [25,1,1], [50,0,0], [25,1,1], [50,0,0], [25,1,1]])
total_duration = round(300 / 1)

data1 = np.zeros((1,round(total_duration/dt)))
data1 = np.zeros((1,round(total_duration/dt)))
data1[0][0:round(4/dt)] = 1


NUM_EVENTS = len(event_data) 
HOUSE_LIGHT_ON= [*range(0, 2, 1)] + [*range(4,6,1)] + [*range(8,10,1)] + [*range(12,15, 1)] + [*range(16,19,1)] # [*range(6, 8, 1)] + [*range(9,11,1)] + [*range(13,16,1)] + [*range(17, 20, 1)] + [*range(21, 24, 1)]#  + [*range(12, 16, 1)] + [*range(18, 22, 1)]

error_arr = np.zeros(NUM_EVENTS)
event_data = relative_to_absolute_event_time(event_data)

# Last event, time axis for plotting        
T = event_data[-1][0]

# Timer with 100 (or however many you want) ramps, all initialized to be very highly weighted (n=1)
timer=TM(1,100)

ax1 = plt.subplot(211) # Subplot for timer activations and events
ax2 = plt.subplot(212) # Subplot for error (not yet calculated)

activation_plot_xvals = np.arange(0, T, dt)

weights = np.array([[2,     0,  0,   -.5,      0,  0,  0,  0],     # 1->1, 2->1, 3->1 4->1
                    [0.55,  1,  0,   -.5,       0,  0,  0,  0],      # 1->2, 2->2, 3->2
                    [0,     .5,  2,   -.5,      0,  0,  0,  0],     # 1->3, 2->3, 3->3
                    [0,     0,  1,    2,      0,  0,  0,  0],
                     
                    [0,     0,  1.1,    0,      2,  0,  0,-.5],
                    [0,     0,  0,    0,     .55, 1,  0,-.5],
                    [0,     0,  0,    0,      0,  .5,  2, -.5],
                    [0,     0,  0,    0,      0,  0,  1, 2]]) 
                             
stretch = .8
beta = 1.2
inhibition_unit_bias = 1.5
ramp_bias = 1.5
lmbd = 4
v_hist = np.array([np.zeros(weights.shape[0])]).T 
noise = NOISE
tau = 1
l = np.array([[lmbd, lmbd, lmbd, lmbd, lmbd, lmbd, lmbd, lmbd]]).T   
#l = np.full([ weights.shape[0] ], lmbd).T  
bias = np.array([[beta, ramp_bias, beta, inhibition_unit_bias, beta, ramp_bias, beta, inhibition_unit_bias]]).T 
 
v = np.array([np.zeros(weights.shape[0])]).T 
net_in = np.zeros(weights.shape[0])
timer_learn_1 = True  
timer_learn_2 = True  
early_1 = False
early_2 = False
stretched = False

for i in range (0, data1.size):   
    net_in = weights @ v

    # Transfer functions
    net_in[0] = sigmoid(l[0], data1[0][i] + net_in[0], bias[0])    
    net_in[1] = piecewise_linear(net_in[1], bias[1])
    net_in[2:5] = sigmoid(l[2:5], net_in[2:5], bias[2:5])
    net_in[5] = piecewise_linear(net_in[5], bias[5])
    net_in[6:8] = sigmoid(l[6:8], net_in[6:8], bias[6:8])

    dv = (1/tau) * ((-v + net_in) * dt) + (noise * np.sqrt(dt) * np.random.normal(0, 1, (weights.shape[0],1)))  # Add noise using np.random
    v = v + dv            
    v_hist = np.concatenate((v_hist,v), axis=1)
    
    z = .99
    
    """=== Early Timer Update Rules ==="""
    if (v[1] >= z) and timer_learn_1 == True:
        early_1 = True
        if i < round(events["pBA"]/dt):
            if not stretched:
                # We're still in the interval, so we keep updating
                # Drift for PL assuming a slope of 1
                drift = ((weights[1][0]) - bias[1] + .5) 
                d_A = (- (drift ** 2)/z) * dt
                weights[1][0] = weights[1][0] + d_A  
            else:
                drift = ((ramp_bias + stretch * (stretch_weights[1][0] - ramp_bias)) - bias[1] + .5)
                d_A = (- (drift ** 2)/z) * dt
                stretch_weights[1][0] = stretch_weights[1][0] + d_A  
        else:
            print("early")
            timer_learn_1 = False
            print(weights[1][0])
    """=== Late Timer Update Rules ==="""                
    if (v[1] > 0) and (timer_learn_1 == True) and (not early_1):
        #         If we hit our target late
        #         Do the late update
        if not stretched:
            timer_learn_1 = False
            z = .99
            Vt = net_in[1][-1]
            v[2] = 1
            # drift = (weights[1][0] - bias[1] + .5)
            """=== LOOK HERE! Timer Update Rules ==="""
            drift = ((weights[1][0] * v[0]) - bias[1] + .5)
            d_A = drift * ((z-Vt)/Vt)
            weights[1][0] = weights[1][0] + d_A
            print("Timer 1 was late, interval 2 starting...")
        #                print(weights[1][0])
        #                print("late")
        #                print("new weights", weights[1][0])
        else: 
            timer_learn_1 = False
            z = .99
            Vt = net_in[1][-1]
            v[2] = 1
            # drift = (weights[1][0] - bias[1] + .5)
            """=== LOOK HERE! Timer Update Rules ==="""
            drift = ((ramp_bias + stretch * (stretch_weights[1][0] - ramp_bias)) * v[0]) - bias[1] + .5
            d_A = drift * ((z-Vt)/Vt)
            stretch_weights[1][0] = stretch_weights[1][0] + d_A
#                print(weights[1][0])
#                print("late")
#                print("new weights", stretch_weights[1][0])
ax1.set_ylim([0,Y_LIM])
ax1.set_xlim([0,T])
ax1.set_ylabel("Activation")
ax1.set_xlabel("Time")
ax1.grid('on')
ax1.hlines(START_THRESHOLD,0,T, color="green", alpha=0.3)
ax1.hlines(STOP_THRESHOLD,0,T, color="red", alpha=0.3)

# For plotting error on ax2# events = event_data[:-1,0]
# MSE = np.square(np.subtract(events,recorded_responses)).mean()
# ax2.plot(np.arange(0,NUM_EVENTS,1), MSE)
# ax2.set_xlim([0,NUM_EVENTS])
# ax2.set_ylabel("Sq Error")
# ax2.set_xlabel("Event #")
# ax2.grid('on')
                