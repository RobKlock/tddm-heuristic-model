#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 21:02:41 2021

@author: Rob Klock

Heuristic Algorithm Exploration
"""
import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import random
from timer_module import TimerModule as TM
from labellines import labelLine, labelLines
from scipy.stats import invgauss   

#print(TM.getSamples())

def getSampleFromParameters(params):
    """
    Parameters
    ----------
    params : TYPE array
        Loc, Scale, Dist TYPE (0 normal, 1 exp) in that order.

    Returns 
    -------
    a sample from provided distribution

    """
    if params[2] == 0:
        return np.random.normal(params[0], params[1], 1)
    else:
        return np.exponential(params[1], 1)

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
    ret_weight = timer_weight + d_A
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

def activationAtIntervalEnd(timer, ramp_index, interval_length, c):
    act = timer.timers[ramp_index] * interval_length
    print("ramp_index: ", ramp_index)
    print(act)
    for i in range (0, len(act)):
        act[i] = act[i] + c * np.sqrt(act[i]) * np.random.normal(0, 1) * math.sqrt(interval_length)
    return act
    

def responseTime(weight, threshold, noise):
    #print(invgauss.rvs(interval_end, 1))
    shape = (threshold/noise)**2
    return invgauss.rvs(threshold/weight, shape)
   # return threshold/weight

def reward(activation, margin=.025):
    # Squared duration of error from event
    if 1 - activation <= margin:
        return 1
    else:
        return 0
    
def score_decay(response_time, event_time):
    #print(f'response_time: {response_time}, event_time: {event_time}')
    #print("response time: ", response_time)
    #print("event time: ", event_time)
    diff = event_time - response_time
    #print(f'diff: {diff}')
    if diff < 0:
        return 0  
    else:
        #return 0.02**(1.0-diff)
        #print("diff: ", diff)
        return 1.2**(-diff)
    
def frange(start, stop, step):
     i = start
     while i < stop:
         yield i
         i += step

def piecewise_linear(v, bias):
    if ((v - (bias) + .5) <= 0):
        return 0
    elif (((v - (bias) + .5)> 0) and ((v - (bias) + .5) < 1)):
        return ((v - (bias) + .5))
    else:
        return 1
    

def plot_early_update_rule(start_time, end_time, timer_weight, T, event_type, value, v0=1, z=.99, bias=1, tau=1, noise=0.002, dt=.2):
    # Alternatively - save every frame of this into a folder and render a movie after the fact. Automate this in Python
    # Eliminate the previous ramp so all we see is a single falling ramp
    v = 0
    v_hist = [0]    
    activation_plot_xvals = np.arange(start_time, end_time, dt) 
    etr = False
    net_in=0
    for i in frange (start_time, end_time, dt):
        net_in = timer_weight * v0 # + (net_in)
        net_in = piecewise_linear(net_in, bias)
        dv = (1/tau) * ((-v + net_in) * dt) + (noise * np.sqrt(dt) * np.random.normal(0, 1, 1))  # Add noise using np.random
        v = v + dv
        v_hist.append(v[0])
        # if i == start_time + z/timer_weight:
        if (i >= start_time + (1/timer_weight)) or (i == 1):
            etr = True
        if etr:
             drift = (timer_weight - 1 + .5) 
             d_A = (- (drift ** 2)/z) * dt
             timer_weight = timer_weight + d_A  
        
             #plt.plot(i, timer_weight,  marker='.', c = colors[event_type], alpha=0.2) 
             # Delete old element
             plt.plot([start_time, i], [0, 1], c=colors[event_type], alpha=0.1)
    
             plt.ylim([0,1.5])
             plt.xlim([start_time,end_time])
             plt.pause(0.0000001)
     
    #plt.plot(activation_plot_xvals, v_hist[0:-1], color=colors[event_type], dashes = [2,2]) 

def plot_hitting_times(weight,threshold,noise):
    print("weight: ", weight)
    print("threshold: ", threshold)
    print("noise: ", noise)
    print("mu: ", mu)

    mu = threshold/weight
    lmbda = (threshold/noise)**2
    
    r = invgauss.rvs(mu/lmbda, scale=lmbda, size=1000)
    plt.hist(r, density=True, histtype='stepfilled', alpha=0.2, bins = 200)

    plt.legend(loc='best', frameon=False)

    plt.show()
    
def compare_random_walk(weight, threshold, noise, dt):
    plt.subplot(211)
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212, sharex=ax1)
    
    mu = threshold/weight
    lmbda = (threshold/noise)**2
    # Set noise coefficient to be 2 * sqrt(drift)) and look at the first passage times 
    
    
    plt.suptitle("mu = {mu}, noise = {noise}, n = 1000".format(mu=mu, noise=noise))
    r = invgauss.rvs(mu/lmbda, scale=lmbda, size=1000)
    ax1.hist(r, density=True, histtype='stepfilled', alpha=0.5, bins = 200)
    ax1.set_title("Inverse Gaussian Variates")
        
    ht = list(map(lambda idx: generate_hit_time(weight, threshold, noise, 0.01), range(1000)))
    ax2.hist(ht, density=True, histtype='stepfilled', alpha=0.5, bins = 200)
    ax2.set_title("Simulated Hitting Times")
    plt.show()
    

def generate_hit_time(weight, threshold, noise, dt, plot=False):   
    T = int(((threshold/weight)+50)/dt)
    arr = np.random.normal(0,1,T) * noise * np.sqrt(dt)
    
    drift_arr = np.ones(T) * weight * dt
    act_arr = drift_arr + arr
    cum_act_arr = np.cumsum(act_arr)
   
    hit_time = np.argmax(cum_act_arr>threshold) * dt
    x = np.arange(0, T*dt, dt)
    
    # plot many trajectories over each other
    if plot:
       plt.figure()
       plt.hlines(threshold,0,T)
       plt.plot(x, cum_act_arr, color="grey")
       plt.xlim([0,hit_time + (hit_time//2)])
       plt.ylim([0, threshold + (threshold/2)])
       
    if hit_time > 0:    
        return hit_time


    
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
        
dt = 0.1
N_EVENT_TYPES= 2 # Number of event types (think, stimulus A, stimulus B, ...)
NUM_EVENTS=10# Total amount of total events
Y_LIM=2 # Vertical plotting limit
NOISE=0.005 # Internal noise - timer activation
LEARNING_RATE=.9 # Default learning rate for timers
STANDARD_INTERVAL=20 # Standard interval duration 
RESPONSE_THRESHOLD=1 
PLOT_FREE_TIMERS=False

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'o', 'k'] # Color support for 8 kinds of events

MAX_SCORE = NUM_EVENTS # Max score over all events is just num_events since max score on a single event is 1
REALLOCATION_THRESHOLD = .7 # If average performance of a timer is below .7 it is reallocated (frozen)

#events_with_type = TM.getSamples(NUM_EVENTS, num_normal = N_EVENT_TYPES)
events_with_type = TM.getSamples(NUM_EVENTS, num_normal = N_EVENT_TYPES, standard_interval = 20)
#events_with_type = TM.getSamples(NUM_EVENTS, num_normal = N_EVENT_TYPES, standard_interval = 20) # All event occurances during this trial

event_occurances = (list(zip(*events_with_type))[0]) # Relative occurance of event
# plt.hist(event_occurances, bins=80, color='black')

events = np.zeros(NUM_EVENTS)
events_with_type[0][0] = event_occurances[0]

# Make event_w_t in terms of absolute time
for i in range (1,NUM_EVENTS):
     events_with_type[i][0] = events_with_type[i-1][0] + events_with_type[i][0]

# Time axis for plotting        
T = events_with_type[-1][0] + 100

# Timer with x ramps, all initialized to be very highly weighted (n=1)
timer=TM(1,40)

plt.figure()

timer.eventDict()[0] = np.arange(0,10).tolist() # Initialize ten ramps to each event type
timer.eventDict()[1] = np.arange(10,20).tolist()
free_indices = np.arange(20,40) # Establish free ramps
timer.time_until[3]=2

# Timers are allocated to an event type based on the timer's eventDict object
# eventDict takes in an event type as a key and gives an array of timer indices 
# for that object as the value

first_event = True

for idx, event in enumerate(events_with_type):
    event_time = event[0]
    event_type = int(event[1])
    
    if event_type not in timer.eventDict():
        # Allocate a new timer for this event type 
        # need protection if we run out of timers 
        timer.eventDict()[event_type] = free_indices[:5].tolist()
        free_indices = free_indices[6:]
        
    event_timer_index = timer.eventDict()[event_type]
    prev_event = 0

    if first_event:
        first_event=False                   
        timer_value = activationAtIntervalEnd(timer, event_timer_index, event_time, NOISE)
        free_timers_vals = activationAtIntervalEnd(timer, free_indices, event_time, NOISE)
        
        response_time = generate_hit_time(timer.timerWeight(event_timer_index[0]), RESPONSE_THRESHOLD, NOISE, dt)
        
        # variable for each ramp about if its falling or not and the event that triggered it
        # A has ramps that are frozen and not frozen, and it times durations to different kinds of events
        
        # Do we want to score the first event which we know is bad?
        timer.setScore(event_timer_index, timer.getScore(event_timer_index[0]) + score_decay(response_time, event_time))
        
        for i in timer_value:
            plt.plot([0,event_time], [0, i], linestyle = "dashed", c=colors[event_type], alpha=0.8)
            #plt.plot([event_time], [i], marker='o',c=colors[event_type],  alpha=0.2) 
            plt.plot([response_time], [RESPONSE_THRESHOLD], marker='x', c=colors[event_type], alpha=0.8) 
        
        if PLOT_FREE_TIMERS:
            for i in free_timers_vals:
                plt.plot([0,event_time], [0, i], linestyle = "dashed", c='grey', alpha=0.5)
                #plt.plot([event_time], [i], marker='o',c=colors[event_type],  alpha=0.2) 

    else:
        prev_event = events_with_type[idx-1][0]
        print("event type: ", event_type, "event_timer_index: ", event_timer_index)
        timer_value = activationAtIntervalEnd(timer, event_timer_index, event_time - events_with_type[idx-1][0], NOISE)
        
        
        timer.active_ramps = free_indices[0]
        
        # if idx>2:
        #     active_ramp_event = events_with_type[idx-2][0]
        #     active_r_t = active_ramp_event + generate_hit_time(timer.timerWeight(event_timer_index[0]), RESPONSE_THRESHOLD, NOISE, dt)
            
        #     plt.plot([active_ramp_event,event_time], [0, i], linestyle = "solid",  c='pink', alpha=0.5)
        #     plt.plot([event_time], [i], marker='o',c='pink', alpha=0.5) 
        #     plt.plot([active_r_t], [RESPONSE_THRESHOLD], marker='x', c='pink', alpha=0.8) 
        print("===Free Timers===")
        free_timers_vals = activationAtIntervalEnd(timer, free_indices, event_time, NOISE)

        response_time = prev_event + generate_hit_time(timer.timerWeight(event_timer_index[0]), RESPONSE_THRESHOLD, NOISE, dt)
        timer.setScore(event_timer_index, timer.getScore(event_timer_index[0]) + score_decay(response_time, event_time))
        
        
        learning_rate = timer.learningRate(event_timer_index[0])
        # Learning rate drops as score increases
        new_learning_rate = math.exp(-0.1 * timer.getScore(event_timer_index[0]))
        print("learning rate: ", new_learning_rate)
        timer.setLearningRate(event_timer_index[0], new_learning_rate)
        
        
        # print("Timer: ", event_timer_index[0], "learning rate: ", timer.learningRate(event_timer_index[0]))
        for i in timer_value:
            plt.plot([prev_event,event_time], [0, i], linestyle = "dashed",  c=colors[event_type], alpha=0.5)
            plt.plot([event_time], [i], marker='o',c=colors[event_type], alpha=0.2) 
            plt.plot([response_time], [RESPONSE_THRESHOLD], marker='x', c=colors[event_type], alpha=0.8) 
        
        if PLOT_FREE_TIMERS:
            for i in free_timers_vals:
               plt.plot([prev_event,event_time], [0, i], linestyle = "dashed", c='grey', alpha=0.5)
               #plt.plot([event_time], [i], marker='o',c=colors[event_type],  alpha=0.2) 
        
    #plot_early_update_rule(prev_event, event_time, timer.timerWeight(), T)
    #print("timer value: ", timer_value)
    update_rule(timer_value, timer, event_timer_index, prev_event, event_time, event_type, plot= False)    
    # TODO: Rest of the heuristic (scores, reallocation, etc)
     
    plt.vlines(event[0], 0,Y_LIM, label="v", color=colors[event_type], alpha=0.5)
    print("event:", event)
    print("\n")
    if Y_LIM>1:
        plt.hlines(1, 0, T, alpha=0.2, color='black')
  
    plt.ylim([0,Y_LIM])
    plt.xlim([0,event_time])
    plt.ylabel("activation")
    plt.xlabel("Time")
    plt.grid('on')
    #plt.pause(0.2)
print(timer.eventDict())
plt.xlim([0,event_time + (.1 * event_time)])
plt.show()
   