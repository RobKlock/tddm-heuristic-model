#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 23:32:38 2022
@author: rob klock
This model stimulates a timer that times events via ramps. 
Ramps are basically weight values that adjust the rate of accumulation in a 
drift-diffusion process. The weights are learned with one-shot rules to time
the most recently observed event. 
Each ramp has:
- A start-event s_1 (last reset)
- A weight w (initally very high, 1)
- A stop event s_2 (can be NA, in which case the ramp is "unassigned")
We refer to s_1 as "initiating event" and s_2 as "terminating event"
We use stimulus and event interchangably. There is a difference, which is 
why both are used, but not in this version.
There is a pool of "free timers" which are ramps that retain their initiated
weight and are able to learn new s_1,s_2 pairs. When a new event sequence is 
observed, the model grabs free timers from the pool and updates them with 
proper weight, s_1, and s_2. They are chosen randomly
"""
import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import random # TODO: Just use np.random instead of random.random
# TODO: Specify the seed so random things are generated again

from timer_module import TimerModule as TM
from labellines import labelLine, labelLines
from scipy.stats import invgauss   
import matplotlib.colors as mcolors
import networkx as nx
import time


''' code review notes
# Define all the parameters
# PEP 8 Standard 

Include datatypes in method comments
specifically look out for numpy: single values, arrays, etc

Class for DDM 
Update rules and activations would be in that class
Step function that does all the updates in the class
'''
def activationAtIntervalEnd(timer, ramp_index, interval_length, c):
    # Simulate DDM process for activation amount
    # Change act to activation
    act = timer.timers[ramp_index] * interval_length
    for i in range (0, len(act)):
        act[i] = act[i] + c * np.sqrt(act[i]) * np.random.normal(0, 1) * math.sqrt(interval_length)
    return act


# Generate when it will cross threshold
# Paramters:
#   weight: scalar,
#   threshold: scalar,
#   noise: scalar,
def generate_hit_time(weight, threshold, noise, dt, plot=False):   
    # Alternative method for hitting time
    T = int(((threshold/weight)+50)/dt)
   
    # TODO: Change to np.random.normal((loc=0, scale=1, size=T))
    # Rename to be clearer
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
    # TODO: get rid of magic numbers here
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
    responses = np.random.exponential(1, num_samples)
    return responses
    
    

def lateUpdateRule(vt, timer_weight, learning_rate, v0=1.0, z = 1, bias = 1):
    """
    TODO: these are out of order
    Include data types in each of these
    
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

    # TODO: Preallocate memory instead of initializing an empty list
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

# Use timer indices in Timer object instead of passing them in parameters
def update_and_reassign_ramps(timer, timer_values, timer_indices, next_stimulus_type, stimulus_type, ramp_graph, ax2, external_idx, sequence_code = '', v0=1.0, z = 1, bias = 1, plot = False):
    # Frozen timers arent updated
    for idx, value in zip(timer_indices, timer_values):
        if idx in timer.frozen_ramps:
            continue
        
        # Generate coin flip for random update
        flip = random.random()

        
    
        if idx in timer.free_ramps:
            stim_type_y_plot_val = (NUM_RAMPS/2) - (NUM_RAMPS/4)
            next_stim_type_y_plot_val = (NUM_RAMPS/2) - (NUM_RAMPS/4)
            # if external_idx == 0:
            #     point_1 = ax2.plot([0], [stim_type_y_plot_val], marker='o',c='green', alpha=0.2) 
            #     point_2 = ax2.plot([4], [next_stim_type_y_plot_val], marker='o',c='green', alpha=0.2) 
                
            #     free_line = ax2.plot([0,2], [stim_type_y_plot_val, idx],c='green', alpha=0.5)
            #     free_line_2 = ax2.plot([2,4], [idx, next_stim_type_y_plot_val],c='green', alpha=0.5)
                
            # else:
            #     stim_type_y_plot_val = (NUM_RAMPS/2) + (stimulus_type * (NUM_RAMPS/4))                 
            #     next_stim_type_y_plot_val = (NUM_RAMPS/2) + (next_stimulus_type * (NUM_RAMPS/4))
            #     ax2.plot([0], [stim_type_y_plot_val], marker='o',c=colors[stimulus_type], alpha=0.2) 
            #     ax2.plot([4], [next_stim_type_y_plot_val], marker='o',c=colors[next_stimulus_type], alpha=0.2) 
                
            #     free_line = ax2.plot([0,2], [stim_type_y_plot_val, idx],c=colors[stimulus_type], alpha=0.5)
            #     free_line_2 = ax2.plot([2,4], [idx, next_stim_type_y_plot_val],c=colors[next_stimulus_type], alpha=0.5)
            
            # ax2.plot([2], [idx], marker='o',c=colors[next_stimulus_type], alpha=0.2) 
            
            # if SAVE_RAMP_NETWORK_ANIMATION_FRAMES:
            #     filename=f'ramp-graph-frames/ramps-{time.time()}.png'
            #     fig.savefig(filename)     
            
            # line_1 = free_line.pop(-1)
            
            # line_1.remove()
            
            
            # line_2 = free_line_2.pop(-1)
            
            # line_2.remove()
        
        
        
        
        """ 
        From all the ramps not idle who have either:
            - S2=e_i and start<act<stop
            - S2 = NA
        Pick N randomly and update them for the interval s1->s2=e_i
        """
        # If a timer is unassigned
        if timer.terminating_events[idx] == -1:
            if flip >=.85: # Update this to be a var, not a magic number
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
                    if flip>=.5:
                        timer.setTimerWeight(timer_weight, idx)
                        
                timer.terminating_events[idx] = next_stimulus_type
                if idx in timer.free_ramps:
                    timer.setTimerWeight(timer_weight, idx)
                    timer.free_ramps = np.delete(timer.free_ramps, np.where(timer.free_ramps == idx))
                    timer.initiating_events[idx] = stimulus_type
                    
                    # ramp_graph.add_edge(idx+N_EVENT_TYPES, stimulus_type)
                    # ramp_graph.add_edge(idx+N_EVENT_TYPES, next_stimulus_type + 200)
        
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
            
def beat_the_clock_reward(event_time, response_time):
    print(response_time[0])
    print(event_time)
    return math.exp(0.4 * (event_time - response_time[0]))

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
BEAT_THE_CLOCK = True
colors = list(mcolors.TABLEAU_COLORS) # Color support for events
ALPHABET_ARR = ['A','B','C','D','E','F','G'] # For converting event types into letters 
ramp_graph=nx.Graph()
SAVE_RAMP_NETWORK_ANIMATION_FRAMES = False
#event_data = TM.getSamples(NUM_EVENTS, num_normal = N_EVENT_TYPES)
#event_data = TM.getSamples(NUM_EVENTS, num_normal = N_EVENT_TYPES, scale_beg = 20, scale_end = 30)
# [Event Time, Event Type, Stimulus Type]
event_data = np.asarray([[0,1,1], [50,0,0], [25,1,1],
                         [50,0,0], [25,1,1], [50,0,0], 
                         [25,1,1], [50,0,0], [25,1,1], 
                         [50,0,0], [25,1,1], [50,0,0], 
                         [25,1,1], [50,0,0], [25,1,1], 
                         [50,0,0], [25,1,1], [50,0,0],
                         [25,1,1], [50,0,0], [25,1,1],
                         [50,0,0], [25,1,1], [50,0,0],
                         [25,1,1], [50,0,0], [25,1,1],
                         [50,0,0], [25,1,1], [50,0,0],
                         [25,1,1], [50,0,0], [25,1,1]])

# event_data = TM.getEvents(25, 2)
NUM_EVENTS = len(event_data) 
HOUSE_LIGHT_ON= [*range(0, 2, 1)] + [*range(4,6,1)] + [*range(8,10,1)] + [*range(12,14,1)] # + [*range(14,19, 1)] + [*range(20,25,1)] + [*range(26,31,1)] # [*range(6, 8, 1)] + [*range(9,11,1)] + [*range(13,16,1)] + [*range(17, 20, 1)] + [*range(21, 24, 1)]#  + [*range(12, 16, 1)] + [*range(18, 22, 1)]
BTC_REWARD=np.empty(NUM_EVENTS)

error_arr = np.zeros(NUM_EVENTS)
event_data = relative_to_absolute_event_time(event_data)

# Last event, time axis for plotting        
T = event_data[-1][0]
NUM_RAMPS = 40
# Timer with 100 (or however many you want) ramps, all initialized to be very highly weighted (n=1)
timer=TM(1,NUM_RAMPS)
fig = plt.figure()
ax1 = fig.add_subplot(211) # Subplot for timer activations and events
ax2 = fig.add_subplot(212) # Subplot for error (not yet calculated)
ax1.set_ylim([0,Y_LIM])
ax1.set_xlim([0,T])
# At each event e_i
for idx, event in enumerate(event_data[:-1]):    
    house_light = idx in HOUSE_LIGHT_ON
    event_time = event[0]
    event_type = int(event[1])
    stimulus_type = int(event[2])
    next_event = event_data[idx+1][0]
    next_stimulus_type=int(event_data[idx+1][2])
    # Plot event times and labels
    if idx < NUM_EVENTS - 1:
            ax1.text(event[0],2.1,ALPHABET_ARR[int(event_data[idx+1][2])])
            ax1.vlines(event_time, 0,Y_LIM, label="v", color=colors[next_stimulus_type])
    else:
        ax1.text(event[0],2.1,'End')
    if house_light:
        # Plot house light bar
        ax1.plot([event_time, next_event], [1.9, 1.9], 'k-', lw=4)  
        # Look forward to all other intervals before house light turns off and start updating weights
        house_light_idx = idx + 1
        house_light_interval = True
        
        while house_light_interval:
            # If the next interval is in the house light period
            if house_light_idx-1 in HOUSE_LIGHT_ON: 
                # Get next event time and stimulus type
                next_house_light_event_time = event_data[house_light_idx][0]
                next_house_light_stimulus_type = event_data[house_light_idx][2]
                
                # All indices of ramps active by initiating event
                initiating_active_indices = np.where(timer.initiating_events == stimulus_type)
                
                # All initiating and free ramp indices
                active_ramp_indices = np.append(initiating_active_indices, timer.free_ramps)
                
                house_light_timer_value = activationAtIntervalEnd(timer, active_ramp_indices, next_house_light_event_time - event_time, NOISE)
                
                # Poisson sequence responses (not fully working yet)
                # responses = respond(house_light_timer_value, event_time, next_house_light_event_time, ax1, idx)
                
                if BEAT_THE_CLOCK:
                    if not (event_time==0):
                        response_time = event_time + start_threshold_time(house_light_timer_value, next_house_light_event_time-event_time)
                        reward = beat_the_clock_reward(event_time, response_time)
                        print(reward)
                        START_THRESHOLD = math.exp(-.2 * reward)
                        ax1.hlines(START_THRESHOLD,0,event_time, color="green", alpha=0.3)
                        BTC_REWARD[idx]=reward
                    
                update_and_reassign_ramps(timer, house_light_timer_value, active_ramp_indices, next_house_light_stimulus_type, stimulus_type, ramp_graph, ax2, idx)
                for i, val in zip(active_ramp_indices, house_light_timer_value):
                    if timer.terminating_events[i] == next_house_light_stimulus_type and timer.initiating_events[i] == stimulus_type or i in timer.free_ramps:
                        if (val<STOP_THRESHOLD and val>START_THRESHOLD) or i in timer.free_ramps:
                            ax1.plot([event_time,next_house_light_event_time], [0, val], linestyle = "dashed",  c=colors[next_stimulus_type], alpha=0.5)
                            ax1.plot([next_house_light_event_time], [val], marker='o',c=colors[next_stimulus_type], alpha=0.2) 
        
                # Contiue to the next event in the house light interval
                house_light_idx+=1
            else:
                house_light_interval=False
       
    
    
        


ax1.set_ylim([0,Y_LIM])
ax1.set_xlim([0,T])
ax1.set_ylabel("Activation")
ax1.set_xlabel("Time")
ax1.grid('on')
#ax1.hlines(START_THRESHOLD,0,T, color="green", alpha=0.3)
ax1.hlines(STOP_THRESHOLD,0,T, color="red", alpha=0.3)

# nx.draw(ramp_graph, with_labels = True)
# plt.savefig("ramp_graph_spectral.png")

# For plotting error on ax2
# events = event_data[:-1,0]
# MSE = np.square(np.subtract(events,recorded_responses)).mean()
# ax2.plot(np.arange(0,NUM_EVENTS,1), MSE)
ax2.plot(BTC_REWARD)
ax2.set_xlim([0,NUM_EVENTS])
# ax2.set_ylabel("Sq Error")
# ax2.set_xlabel("Event #")
# ax2.grid('on')
                