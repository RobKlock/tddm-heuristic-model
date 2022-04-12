#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 23:32:38 2022

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
       
        if flip >=0:
            # only update timer for those that keep track of next stim type
           # print("event dict timer:", timer.eventDict())
            # event_dict=timer.eventDict()
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
    start_threshold_times = start_threshold_time(timer_value, next_event-event_time)
    start_threshold_times += event_time
    start_threshold_times.sort()
    start_threshold_times = np.vstack((start_threshold_times, np.ones(len(start_threshold_times)))).T
    
    stop_threshold_times = stop_threshold_time(timer_value, next_event-event_time)
    stop_threshold_times += event_time
    stop_threshold_times.sort()
    stop_threshold_times = np.vstack((stop_threshold_times, (-1* np.ones(len(stop_threshold_times))))).T
    
    start_stop_pairs = np.vstack((start_threshold_times, stop_threshold_times))
    start_stop_pairs = start_stop_pairs[start_stop_pairs[:, 0].argsort()]

    responses = []
    response_periods = []
    k = 0
    k_o = 0 # old value of k
    # print(event_time)
    # print(start_stop_pairs)
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
    # print(response_periods)
    # print(r)
    
    for response_period in response_periods:
        responses.extend([i for i in r if (i>response_period[0] and i<response_period[1] and i<next_event and i>event_time)])
        # ax1.vlines(response_period[0], 0,Y_LIM, color="green")
        # ax1.vlines(response_period[1], 0,Y_LIM, color="red")
        # ax1.text(response_period[0],1.5,str(idx))
        # ax1.text(response_period[1],1.5,str(idx))
    
    ax1.plot(responses, np.ones(len(responses)), 'x') 
    responses and ax1.text(responses[0],1.2,str(idx))
    return responses

def multi_stim_update_rule(timer_values, timer, timer_indices, next_stimulus_type, sequence_code = '', v0=1.0, z = 1, bias = 1, plot = False):
    # Frozen timers arent updated
    for idx, value in zip(timer_indices, timer_values):
        if idx in timer.frozen_ramps:
            continue
        if i in timer.free_ramps:
            continue
        
        flip = random.random()
        
        # coin flip update
        if flip >=.2:
            if next_stimulus_type not in timer.terminatingDict():
                timer.terminatingDict()[next_stimulus_type] = [idx]
            else:
                timer.terminatingDict()[next_stimulus_type].append(idx)
            
            if value > 1:
                ''' Early Update Rule '''
                #plot_early_update_rule(start_time, end_time, timer_weight, T, event_type, value)
                
                timer_weight = earlyUpdateRule(value, timer.timerWeight(idx), timer.learningRate(idx))
                plt.grid('on')
                
                
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
# NUM_EVENTS=17#  Total amount of events
Y_LIM=2 # Vertical plotting limit
NOISE=0.00 # Internal noise - timer activation
LEARNING_RATE=.8 # Default learning rate for timers
STANDARD_INTERVAL=20 # Standard interval duration 
K = 5 # Amount of timers that must be active to respond
START_THRESHOLD=.5
STOP_THRESHOLD=1.5
TIMER_THRESHOLD=1 
PLOT_FREE_TIMERS=False
recorded_responses=[]
colors = list(mcolors.TABLEAU_COLORS) # Color support for events
NEW_TIMERS=20

ALPHABET_ARR = ['A','B','C','D','E','F','G']

#HOUSE_LIGHT_ON = [*range(0,NUM_EVENTS+1,1)]
#events_with_type = TM.getSamples(NUM_EVENTS, num_normal = N_EVENT_TYPES)
#events_with_type = TM.getSamples(NUM_EVENTS, num_normal = N_EVENT_TYPES, scale_beg = 20, scale_end = 30)
events_with_type = np.asarray([[0,1,1], [50,0,0], [25,1,1], [50,0,0], [25,1,1], [50,0,0], [25,1,1], [50,0,0], [25,1,1], [50,0,0], [25,1,1], [50,0,0], [25,1,1], [50,0,0], [25,1,1], [50,0,0], [25,1,1],
                               [50,0,0], [25,1,1], [50,0,0], [25,1,1], [50,0,0], [25,1,1], [50,0,0], [25,1,1]])
NUM_EVENTS = len(events_with_type)
HOUSE_LIGHT_ON= [*range(0, 4, 1)] + [*range(6, 10, 1)] #  + [*range(12, 16, 1)] + [*range(18, 22, 1)]

observed_stim_type =  [False] * N_EVENT_TYPES
#events_with_type = np.insert(events_with_type, 0, [0,1,1], axis=0)
# NUM_EVENTS += 1

events = np.zeros(NUM_EVENTS)
error_arr = np.zeros(NUM_EVENTS)

# Make event_w_t in terms of absolute time
for i in range (1,NUM_EVENTS):
     events_with_type[i][0] = events_with_type[i-1][0] + events_with_type[i][0]

# Time axis for plotting        
T = events_with_type[-1][0]

# Timer with N ramps, all initialized to be very highly weighted (n=1)
timer=TM(1,100)

ax1 = plt.subplot(211)
ax2 = plt.subplot(212)

#timer.eventDict()[0] = np.arange(0,10).tolist() # Initialize ten ramps to first event type

# Timers are allocated to an event type based on the timer's eventDict object
# eventDict takes in an event type as a key and gives an array of timer indices 
# for that object as the value

# Ramps with terminating event s_2 = e_i or s_2 is unassigned 
# Random collection of N ramps are updated for interval s1 -> s2 = e_i

# At each event e_i
first_event = True
for idx, event in enumerate(events_with_type[:-1]):    
    # Two cases: 
        # First Event
            # Start at zero and look at the next event
        # Middle Event
            # Start at prior event and look at next event   
   
    house_light = idx in HOUSE_LIGHT_ON
    
    event_time = event[0]
    event_type = int(event[1])
    stimulus_type = int(event[2])
    next_event = events_with_type[idx+1][0]
    next_stimulus_type=int(events_with_type[idx+1][2])
    

    if stimulus_type not in timer.stimulusDict():
        print("stim type not found")
        # Allocate a new timer for this event type 
        # TODO: need protection if we run out of timers 
        # stimulus type is A, B, C 
        timer.stimulusDict()[stimulus_type] = timer.free_ramps[:NEW_TIMERS].tolist()
        timer.free_ramps = timer.free_ramps[NEW_TIMERS+1:]
    
    ramps_stim_index = np.append(timer.stimulusDict()[stimulus_type], timer.free_ramps)
    
    ax1.vlines(event_time, 0,Y_LIM, label="v", color=colors[next_stimulus_type])
    if house_light:
        ax1.plot([event_time, next_event], [1.9, 1.9], 'k-', lw=4)
        #not observed_stim_type[stimulus_type] and coin_flip_update_rule(timer_value, timer, ramps_stim_index, event_time, next_event, stimulus_type, event_type, next_stimulus_type, plot= False)
      
        # Look forward to all other intervals before house light turns off and start updating weights
        curr_interval_idx = HOUSE_LIGHT_ON.index(idx)
        next_house_light_idx = idx + 1
        
        house_light_interval = True
        sequence_code=''
        while house_light_interval:
            # If the next interval is in the house light period
            if next_house_light_idx-1 in HOUSE_LIGHT_ON: 
                next_event_o_time = events_with_type[next_house_light_idx][0]
                next_stimulus_o_type = events_with_type[next_house_light_idx][2]
                print(f'INNER timing from {event_time} to {next_event_o_time}...')
                # Can use sequence code if we want to handle A->B->C as a distinct event type
                # sequence_code = sequence_code + str(int(stimulus_type)) + str(int(next_house_light_stimulus))
                
                hl_timer_value = activationAtIntervalEnd(timer, ramps_stim_index, next_event_o_time - event_time, NOISE)
                
                for i in hl_timer_value:
                    if i in timer.free_ramps:
                        ax1.plot([event_time,next_event_o_time], [0, i], linestyle = "dashed",  c='g', alpha=0.3)
                        ax1.plot([next_event_o_time], [i], marker='o',c='g', alpha=0.2) 
                    else:
                        ax1.plot([event_time,next_event_o_time], [0, i], linestyle = "dashed",  c=colors[next_stimulus_type], alpha=0.5)
                        ax1.plot([next_event_o_time], [i], marker='o',c=colors[next_stimulus_type], alpha=0.2) 
                
                # responses = respond(hl_timer_value, event_time, next_event_o_time, ax1, idx)
                
                multi_stim_update_rule(hl_timer_value, timer, ramps_stim_index, next_stimulus_o_type)
                
                next_house_light_idx+=1
            
                if not observed_stim_type[stimulus_type]:
                    observed_stim_type[stimulus_type] = True
                    break
            
            else:
                house_light_interval=False
        
        first_event=False
        
    
        
    
    if idx < NUM_EVENTS - 1:
            ax1.text(event[0],2.1,ALPHABET_ARR[int(events_with_type[idx+1][2])])
    else:
        ax1.text(event[0],2.1,'End')
        
        
events = events_with_type[:-1,0]
#MSE = np.square(np.subtract(events,recorded_responses)).mean()

ax1.set_ylim([0,Y_LIM])
ax1.set_xlim([0,T])
ax1.set_ylabel("Activation")
ax1.set_xlabel("Time")
ax1.grid('on')
ax1.hlines(START_THRESHOLD,0,T, color="green", alpha=0.3)
ax1.hlines(STOP_THRESHOLD,0,T, color="red", alpha=0.3)

# ax2.plot(np.arange(0,NUM_EVENTS,1), MSE)
# ax2.set_xlim([0,NUM_EVENTS])
# ax2.set_ylabel("Sq Error")
# ax2.set_xlabel("Event #")
# ax2.grid('on')
                