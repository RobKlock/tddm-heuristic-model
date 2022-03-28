#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 14:38:18 2022

@author: rob klock
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
    act = timer.timers[ramp_index] * interval_length
    #print("ramp_index: ", ramp_index)
    #print(act)
    for i in range (0, len(act)):
        act[i] = act[i] + c * np.sqrt(act[i]) * np.random.normal(0, 1) * math.sqrt(interval_length)
    return act

def generate_hit_time(weight, threshold, noise, dt, plot=False):   
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
    angle = np.arctan(act_at_interval_end/interval_length)
    beta = 3.14159 - (1.5708 + angle)
    return START_THRESHOLD * np.tan(3.14159 - (1.5708 + angle))

def stop_threshold_time(act_at_interval_end, interval_length):
    angle = np.arctan(act_at_interval_end/interval_length)
    beta = 3.14159 - (1.5708 + angle)
    return STOP_THRESHOLD * np.tan(3.14159 - (1.5708 + angle))
    
def generate_responses(interval_length):
    num_samples = int(interval_length / dt)
    responses = np.random.exponential(1, num_samples)
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
    
    #drift = ((timer_weight * v0) - bias + .5)
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
    #drift = ((timer_weight * v0) - bias + .5)
    drift = (timer_weight * v0)
    d_A = drift * ((vt-z)/vt)
    ret_weight = timer_weight - (learning_rate * d_A)
    return ret_weight

def update_rule(timer_values, timer, timer_indices, start_time, end_time, event_type, v0=1.0, z = 1, bias = 1, plot = False):
    # Frozen timers arent updated
    for idx, value in zip(timer_indices, timer_values):
        if idx in timer.frozen_ramps:
            continue
        
        if value > 1:
            ''' Early Update Rule '''
            #lot_early_update_rule(start_time, end_time, timer_weight, T, event_type, value)
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
    # Frozen timers arent updated
    for idx, value in zip(timer_indices, timer_values):
        if idx in timer.frozen_ramps:
            continue
        flip = random.random()
       
        if flip >=.5:
            # only update timer for those that keep track of next stim type
           # print("event dict timer:", timer.eventDict())
            # event_dict=timer.eventDict()
            if int(next_stimulus_type) in timer.stimulusDict().keys() and (idx in timer.stimulusDict()[int(next_stimulus_type)]):
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

    
# A ramp has a start-event s_1 (last reset), weight w (inf)
# A timer has start-event s_1, weight w, and stop event s_2, can be off
# Universal start threshold = 0.9, stop = 1.1. Timer threshold is 1

dt = 0.1
N_EVENT_TYPES= 2 # Number of event types (think, stimulus A, stimulus B, ...)
NUM_EVENTS=20 # Total amount of events
Y_LIM=2 # Vertical plotting limit
NOISE=0.04 # Internal noise - timer activation
LEARNING_RATE=.99 # Default learning rate for timers
STANDARD_INTERVAL=20 # Standard interval duration 
K = 100 # Amount of timers that must be active to respond
START_THRESHOLD=.9
STOP_THRESHOLD=.95
TIMER_THRESHOLD=1 
PLOT_FREE_TIMERS=False

colors = list(mcolors.TABLEAU_COLORS) # Color support for events

ALPHABET_ARR = ['A','B','C','D','E','F','G']
# HOUSE_LIGHT_ON= [*range(0, 5, 1)] + [*range(14, 25, 1)] + [*range(30, 40, 1)] + [*range(42, NUM_EVENTS, 1)]
HOUSE_LIGHT_ON = [*range(0,NUM_EVENTS+1,1)]
#events_with_type = TM.getSamples(NUM_EVENTS, num_normal = N_EVENT_TYPES)
events_with_type = TM.getSamples(NUM_EVENTS, num_normal = N_EVENT_TYPES, scale_beg = 20, scale_end = 30)
events_with_type = np.insert(events_with_type, 0, [0,0,0], axis=0)
NUM_EVENTS = NUM_EVENTS+1

event_occurances = (list(zip(*events_with_type))[0]) # Relative occurance of event

events = np.zeros(NUM_EVENTS)
error_arr = np.zeros(NUM_EVENTS)

# Make event_w_t in terms of absolute time
for i in range (1,NUM_EVENTS):
     events_with_type[i][0] = events_with_type[i-1][0] + events_with_type[i][0]

# Time axis for plotting        
T = events_with_type[-1][0]

# Timer with x ramps, all initialized to be very highly weighted (n=1)
timer=TM(2,200)

ax1 = plt.subplot(211)
ax2 = plt.subplot(212)

timer.eventDict()[0] = np.arange(0,10).tolist() # Initialize ten ramps to each event type
free_indices = np.arange(10,200) # Establish free ramps

# Timers are allocated to an event type based on the timer's eventDict object
# eventDict takes in an event type as a key and gives an array of timer indices 
# for that object as the value

first_event = True

# At each event e_i

# From all not idle ramps
# Ramps with terminating event s_2 = e_i or s_2 is unassigned 
# Random collection of N ramps are updated for interval s1 -> s2 = e_i

# Active ramps (act>stop) are reset and s_1 is set to e_i
# K idle ramps are started

# If R ramps are between start and stop thresholds, respond
for idx, event in enumerate(events_with_type):
    house_light = True
    prev_event = 0
    event_time = event[0]
    event_type = int(event[1])
    stimulus_type = int(event[2])
    
    try:
        next_stimulus_type = int(events_with_type[idx+1][2])
    except IndexError:
        next_stimulus_type = -1
    
    if stimulus_type not in timer.stimulusDict():
        # Allocate a new timer for this event type 
        # need protection if we run out of timers 
        # stimulus type is A, B, C 
        timer.stimulusDict()[stimulus_type] = free_indices[:10].tolist()
        free_indices = free_indices[11:]
    
    if event_type not in timer.eventDict():
        # event type is really interval type (0-8) or A->B, B->A, etc
        timer.eventDict()[event_type] = free_indices[:10].tolist()
        free_indices = free_indices[11:]
    
    ramps_stim_index = timer.stimulusDict()[stimulus_type]
    
    # first event, e_0, activates < 200 ramps with s_1 = e_0 
    if first_event:
        first_event=False   
        event_time = events_with_type[idx][0]
        next_event = events_with_type[idx+1][0]
        # plot house light indicator
        ax1.plot([0, events_with_type[idx][0]], [1.9, 1.9], 'k-', lw=4)
        ramps_stim_index = np.arange(0,len(timer.timers))
        
        timer_value = activationAtIntervalEnd(timer, ramps_stim_index, next_event, NOISE)
        # plot the full step by step process
        # use np.where to find where it goes above and below threshold
        # keep track of how many are in response range
        
        start_threshold_times = start_threshold_time(timer_value, next_event)
        start_threshold_times.sort()
        stop_threshold_times = stop_threshold_time(timer_value, next_event)
        stop_threshold_times.sort()
        start_stop_pairs = np.vstack((start_threshold_times, stop_threshold_times)).T
        r = list(generate_responses(100))
        c = np.cumsum(r)
        responses = []
        k_count = 0
        start_ramp_pointer_idx = K
        stop_ramp_pointer_idx = 0
        response_start_pointer=0
        response_end_pointer=0
        
        # This has to be in terms of dt/aboslute time, not start stop pairs
        for jdx, time in enumerate(start_stop_pairs):
            # Need to see if this is equivalent to the original algorithm
            # do we need to sample at every timestep?
            if time[0] < start_stop_pairs[stop_ramp_pointer_idx][1]:
                k_count+=1
                response_end_pointer = stop_ramp_pointer_idx
            else:
                k_count=max(0,k_count-1)
                stop_ramp_pointer_idx+=1
                response_end_pointer = stop_ramp_pointer_idx
            
            # Generate all and cumsum instead of this
            s =  time[0] + .01 # np.random.exponential(1, 1) * dt
                
            (k_count >= K and r and s<stop_threshold_times[-1]) and responses.append(s)
        
        ax1.plot(responses, np.ones(len(responses)), '.')
        
        # for t in range(start_stop_pairs[0][0], start_stop_pairs[-1][1], dt):
            
        #print(np.argwhere(start_stop_pairs > start_stop_pairs[0][1]))
            
                
        
        free_timers_vals = activationAtIntervalEnd(timer, free_indices, next_event, NOISE)
        
        response_time = generate_hit_time(timer.timerWeight(ramps_stim_index[0]), TIMER_THRESHOLD, NOISE, dt)
        
        coin_flip_update_rule(timer_value, timer, ramps_stim_index, prev_event, event_time, stimulus_type, event_type, next_stimulus_type, plot= False)    
       
        # Do we want to score the first event which we know is bad?
        #timer.setScore(ramps_stim_index, timer.getScore(ramps_stim_index[0]) + score_decay(response_time, event_time))
        ax1.vlines(event_time, 0,Y_LIM, label="v", color=colors[4 + int(event[2])])
        #ax1.text(event_time,2.1,ALPHABET_ARR[int(events_with_type[idx][2])])
        ax1.text(event[0],2.1,ALPHABET_ARR[int(events_with_type[idx+1][2])])
      
        
        
        for i, value in enumerate(timer_value):
           ax1.plot([start_threshold_times[i]], [START_THRESHOLD], marker='x', alpha=0.8) 
           ax1.plot([stop_threshold_times[i]], [STOP_THRESHOLD], marker='o', alpha=0.8) 
          # ax1.plot([0,next_event], [0, value], linestyle = "dashed", c=colors[stimulus_type], alpha=0.1)
           #plt.plot([event_time], [i], marker='o',c=colors[event_type],  alpha=0.2) 
           # ax1.plot([response_time], [RESPONSE_THRESHOLD], marker='o', c=colors[stimulus_type], alpha=0.8) 
           
        if PLOT_FREE_TIMERS:
            for i in free_timers_vals:
                ax1.plot([0,event_time], [0, i], linestyle = "dashed", c='grey', alpha=0.5)
                
ax1.plot([0,2],[START_THRESHOLD, START_THRESHOLD], '0.8', lw=1)
ax1.plot([0,2],[STOP_THRESHOLD, STOP_THRESHOLD], '0.8', lw=1)

#ax2.hist(r, bins=200)
ax2.plot(c, np.ones(1000), '.')

    

    
    