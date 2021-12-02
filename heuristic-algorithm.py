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
    if diff <= 0:
        return 0  
    else:
        #return 0.02**(1.0-diff)
        #print("diff: ", diff)
        return 2**(-diff/5)
    
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
        net_in = timer_weight * v0 + (net_in)
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
    ax2 = plt.subplot(212)
    
    mu = threshold/weight
    lmbda = (threshold/noise)**2
    
    r = invgauss.rvs(mu/lmbda, scale=lmbda, size=1000)
    ax1.hist(r, density=True, histtype='stepfilled', alpha=0.2, bins = 200)

    ax1.legend(loc='best', frameon=False)
    
    ht = list(map(lambda idx: generate_hit_time(weight, threshold, noise, 0.05), range(1000)))
    ax2.hist(ht, density=True, histtype='stepfilled', alpha=0.2, bins = 200)
    ax2.legend(loc='best', frameon=False)
    plt.show()

def generate_hit_time(weight, threshold, noise, dt):
    arr = np.random.normal(0,1,int(((threshold/weight)+5)/dt)) * np.sqrt(dt) * noise
    for i in range (1, arr.shape[0]):
        arr[i] = arr[i-1] + (weight * dt) + (np.random.normal(0, 1, 1) * noise * np.sqrt(dt)) 
    
    hit_time = np.argmax(arr>threshold) * dt
    return hit_time
    
def update_rule(timer_values, timer, timer_indices, start_time, end_time, event_type, v0=1.0, z = 1, bias = 1, plot = False):
    for idx, value in zip(timer_indices, timer_values):
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
N_EVENT_TYPES=2 # Number of event types (think, stimulus A, stimulus B, ...)
NUM_EVENTS=4# Total amount of events across all types
Y_LIM=2 # Plotting limit
NOISE=0.002 # Internal noise - timer activation
LEARNING_RATE=.9
STANDARD_INTERVAL=20
RESPONSE_THRESHOLD=0.99
ALPHA = 0.8
colors = ['b', 'g', 'r', 'c', 'm', 'y']
ANIMATE_FIRST_EARLY_RULE = True

events_with_type = TM.getSamples(NUM_EVENTS, num_normal = 3, num_exp = 1)
events_with_type = TM.getSamples(NUM_EVENTS, num_normal = 2, num_exp = 0, standard_interval = 15)
event_occurances = (list(zip(*events_with_type))[0])
plt.hist(event_occurances, bins=80, color='black')

events = np.zeros(NUM_EVENTS)
events_with_type[0][0] = event_occurances[0]

for i in range (1,NUM_EVENTS):
     events_with_type[i][0] = events_with_type[i-1][0] + events_with_type[i][0]

# Time axis for plotting        
T = events_with_type[-1][0] + 100

# Timer with 20 ramps, all initialized to be very highly weighted
timer=TM(1,20)

plt.figure()

timer.eventDict()[0] = [0,1,2]
timer.eventDict()[1] = [3,4,5]
free_indices = np.arange(5,20)

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
        timer.eventDict()[event_type] = free_indices[:3]
        free_indices = free_indices[3:]
        
    event_timer_index = timer.eventDict()[event_type]
    prev_event = 0

    if first_event:
        first_event=False                   
        timer_value = activationAtIntervalEnd(timer, event_timer_index, event_time, NOISE)
        # TODO: set up response times correctly. for now its the first of the timers
        # TODO: set up response time with noise. centered at event time, deviation proportional to noise and interval
    
        response_time = responseTime(timer.timerWeight(event_timer_index[0]), RESPONSE_THRESHOLD, NOISE)

        # TODO: set up scores
        # Do we want to score the first event which we know is bad?
        #timer.setScore(event_timer_index, timer.getScore(event_timer_index[0]) + score_decay(response_time, event_time))
        
        for i in timer_value:
            plt.plot([0,event_time], [0, i], linestyle = "dashed", c=colors[event_type], alpha=0.5)
            plt.plot([event_time], [i], marker='o',c=colors[event_type],  alpha=0.2) 

    else:
        prev_event = events_with_type[idx-1][0]
        timer_value = activationAtIntervalEnd(timer, event_timer_index, event_time - events_with_type[idx-1][0], NOISE)
        response_time = prev_event + responseTime(timer.timerWeight(event_timer_index[0]), RESPONSE_THRESHOLD, NOISE)
        timer.setScore(event_timer_index, timer.getScore(event_timer_index[0]) + score_decay(response_time, event_time))
        learning_rate = timer.learningRate(event_timer_index[0])
        new_learning_rate = 1 - math.exp(-0.1 * timer.getScore(event_timer_index[0]))
        timer.setLearningRate(event_timer_index[0], new_learning_rate)
        print("Timer: ", event_timer_index[0], "learning rate: ", timer.learningRate(event_timer_index[0]))
        for i in timer_value:
            plt.plot([prev_event,event_time], [0, i], linestyle = "dashed",  c=colors[event_type], alpha=0.5)
            plt.plot([event_time], [i], marker='o',c=colors[event_type], alpha=0.2) 
            plt.plot([response_time], [RESPONSE_THRESHOLD], marker='x', c=colors[event_type], alpha=0.8) 
    # plot_early_update_rule(prev_event, event_time, timer.timerWeight(), T)
    update_rule(timer_value, timer, event_timer_index, prev_event, event_time, event_type, plot= False)    
    # TODO: Rest of the heuristic (scores, reallocation, etc)
     
    plt.vlines(event, 0,Y_LIM, label="v", color=colors[event_type], alpha=0.5)
    if Y_LIM>1:
        plt.hlines(1, 0, T, alpha=0.2, color='black')
  
    plt.ylim([0,Y_LIM])
    plt.xlim([0,T])
    plt.ylabel("activation")
    plt.xlabel("Time")
    plt.grid('on')
    plt.pause(0.4)
plt.show()
   