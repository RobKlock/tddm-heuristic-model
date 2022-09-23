#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 13:45:56 2022

@author: Robert Klock
"""
import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import random # TODO: Just use np.random instead of random.random
# TODO: Specify the seed so random things are generated again

from timer_module import TimerModule as TM
import matplotlib.colors as mcolors
import time

# Global constants
ALPHABET_ARR = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','AA','BB','CC'] # For converting event types into letters 
colors = [[1,0,0], [0,1,0], [0,0,1], [1,1,0], [0,1,1], [1,0,1],[.46,.03,0], [.1,.3,.2], [.2,.7,.2], [.5,.3,.6], [.7,.3,.4]]# list(mcolors.CSS4_COLORS) # Color support for events
NUM_RAMPS = 100
Y_LIM=2
NOISE=0.0
dt = 0.1
RAMPS_PER_EVENT = 1

START_THRESHOLD=.5 # Response start threshold
STOP_THRESHOLD=1.2 # Response stop threshold

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


# Use timer indices in Timer object instead of passing them in parameters
def update_and_reassign_ramps(timer, timer_values, timer_indices, next_stimulus_type, stimulus_type, ax2, external_idx, sequence_code = '', v0=1.0, z = 1, bias = 1, plot = False):
    # Frozen timers arent updated
    for idx, value in zip(timer_indices, timer_values):
        if idx in timer.frozen_ramps:
            continue
        
        # Generate coin flip for random update
        flip = random.random()

        if idx in timer.free_ramps:
            stim_type_y_plot_val = (NUM_RAMPS/2) - (NUM_RAMPS/4)
            next_stim_type_y_plot_val = (NUM_RAMPS/2) - (NUM_RAMPS/4)
          
        """ 
        From all the ramps not idle who have either:
            - S2=e_i and start<act<stop
            - S2 = NA
        Pick N randomly and update them for the interval s1->s2=e_i
        """
        # If a timer is unassigned
        if len(
                np.where(
                    timer.terminating_events[
                        np.where(timer.initiating_events == stimulus_type)] == next_stimulus_type)[0])>RAMPS_PER_EVENT:
            continue
        
        if timer.terminating_events[idx] == -1:
            if flip >=.75: # Update this to be a var, not a magic number
                # if the timer has the appropriate terminating event, update the weight
                if value > 1:
                    ''' Early Update Rule '''
                    #plot_early_update_rule(start_time, end_time, timer_weight, T, event_type, value)
                    timer_weight = earlyUpdateRule(value, timer.timerWeight(idx), timer.learningRate(idx))
                    plt.grid('on')
                    
                else:
                    ''' Late Update Rule '''
                    timer_weight = lateUpdateRule(value, timer.timerWeight(idx), timer.learningRate(idx))
                        
                timer.terminating_events[idx] = next_stimulus_type
                if idx in timer.free_ramps:
                    timer.setTimerWeight(timer_weight, idx)
                    timer.free_ramps = np.delete(timer.free_ramps, np.where(timer.free_ramps == idx))
                    timer.initiating_events[idx] = stimulus_type
        
            continue
        
        if timer.terminating_events[idx] == next_stimulus_type and timer.initiating_events[idx] == stimulus_type:
            if flip>=.9:
                if value > 1:
                    ''' Early Update Rule '''
                    timer_weight = earlyUpdateRule(value, timer.timerWeight(idx), timer.learningRate(idx))
                        
                else:
                    ''' Late Update Rule '''
                    timer_weight = lateUpdateRule(value, timer.timerWeight(idx), timer.learningRate(idx))
                timer.setTimerWeight(timer_weight, idx)   

def activationAtIntervalEnd(timer, ramp_index, interval_length, c):
    # Simulate DDM process for activation amount
    # Change act to activation
    act = timer.timers[ramp_index] * interval_length
    for i in range (0, len(act)):
        act[i] = act[i] + c * np.sqrt(act[i]) * np.random.normal(0, 1) * math.sqrt(interval_length)
    
    return act

def relative_to_absolute_event_time(relative_time_events):
     # TODO: Change this to a simple cumsum on the first column
    absolute_time_events = relative_time_events   
    for i in range (1,NUM_EVENTS):
        absolute_time_events[i][0] = relative_time_events[i-1][0] + relative_time_events[i][0]
     
    return absolute_time_events  

def activationAtIntervalEndHierarchical(timer, ramp_index, interval_length, next_stimulus_type, c, ax2):
    # Simulate DDM process for activation amount
    # Change act to activation
    delta = 0.5
    '''
    assigned_ramps = []
    for timer_idx in ramp_index:
        if timer.terminating_events[ramp_index] == next_stimulus_type:
            assigned_ramps.append(timer_idx)
    '''       
    act = timer.timers[ramp_index] * interval_length
    
    for i in range (0, len(act)):
        act[i] = act[i] + c * np.sqrt(act[i]) * np.random.normal(0, 1) * math.sqrt(interval_length)
        f_act = act * act
    
    for timer_act in act:
        hier_act = timer_act + 1/((act.size)-1) * delta * f_act[act != timer_act].sum()
    
    return act

event_data = np.asarray([[0,0,0], [50,0,0], [25,2,1],
                          [50,0,2], [25,1,3], [50,0,4], 
                          [25,1,5], [50,0,0], [25,1,1], 
                          [50,0,2], [25,1,3], [50,0,4], 
                          [25,1,5], [50,0,0], [25,1,1], 
                          [50,0,2], [25,1,3], [50,0,4],
                          [25,1,5], [50,0,0], [25,1,1],
                          [50,0,2], [25,1,3], [50,0,4],
                          [25,1,5], [50,0,0], [25,1,1],
                          [50,0,2], [25,1,3], [50,0,4],
                          [25,1,5], [50,0,0], [25,1,1]])

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

NUM_EVENTS = len(event_data) 
event_data = relative_to_absolute_event_time(event_data)
HOUSE_LIGHT_ON = [*range(0,1,1)] + [*range(2,3,1)] + [*range(5,6,1)] + [*range(7,8,1)] # + [*range(12,17,1)]
T = event_data[HOUSE_LIGHT_ON[-1]+1][0] + 20
LAST_HOUSE_LIGHT = HOUSE_LIGHT_ON[-1]
# Timer with 100 (or however many you want) ramps, all initialized to be very highly weighted (n=1)
timer=TM(1,NUM_RAMPS)

fig = plt.figure()
ax3 = fig.add_subplot(211) # Subplot for timer activations and events
ax2 = fig.add_subplot(212, sharex=ax3) # Subplot for error (not yet calculated)


simulation_behavior = plt.figure()
# simulation_behavior.suptitle('Event Sequence', fontsize=16)
simulation_subplot = simulation_behavior.add_subplot(111)
simulation_subplot.set_ylim([0,Y_LIM])
simulation_subplot.set_xlim([0,T])
simulation_subplot.set_ylabel("Activation")
simulation_subplot.set_xlabel("Time")
simulation_subplot_legend = {}

'''
reproduced_sequence_plot = plt.figure()
reproduced_sequence_plot.suptitle('Reproduced Sequence', fontsize=16)
rsp_lines = reproduced_sequence_plot.add_subplot(212)
rsp = reproduced_sequence_plot.add_subplot(211, sharex=rsp_lines)
# rsp_lines.set_ylim([0,.5])
rsp_lines.set_xlim([0,event_data[seq_len][0]+10])
rsp.set_ylim([0,Y_LIM])
rsp.set_xlim([0,200])
hist_lines = []
# captured_distribution_plot = plt.figure()
# cap_dist = captured_distribution_plot.add_subplot(111)
'''

''' Simulation Start '''
# At each event e_i
for idx, event in enumerate(event_data[:-1]):    
    house_light = idx in HOUSE_LIGHT_ON
    event_time = event[0]
    event_type = int(event[1])
    stimulus_type = int(event[2])
    next_event = event_data[idx+1][0]
    next_stimulus_type=int(event_data[idx+1][2])
    
    # Plot event times and labels
    if idx < (LAST_HOUSE_LIGHT+2):
        simulation_subplot.text(event[0],Y_LIM+.05,ALPHABET_ARR[int(event_data[idx+1][2])])
        simulation_subplot.vlines(event_time, 0,Y_LIM, label="v", color=colors[next_stimulus_type])
        
        # ax2.text(event[0],2.1,ALPHABET_ARR[int(event_data[idx+1][2])])
        # ax2.vlines(event_time, 0,Y_LIM, label="v", color=colors[next_stimulus_type])
    # else:
    #    simulation_subplot.text(event[0],2.1,'End')
            
    if house_light:
        # Plot house light bar
        house_light_bar = simulation_subplot.plot([event_time, next_event], [1.9, 1.9], 'k-', lw=4)  
        # ax2.plot([event_time, next_event], [1.9, 1.9], 'k-', lw=4)  
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
                house_light_hierarchical_value = activationAtIntervalEndHierarchical(timer, initiating_active_indices, next_house_light_stimulus_type, next_house_light_event_time - event_time, NOISE, ax2)
                # active_timer_value = activationAtIntervalEndEulerian(timer, active_ramp_indices, next_house_light_event_time - event_time, NOISE, dt)
                active_timer_value = activationAtIntervalEndHierarchical(timer, active_ramp_indices, next_house_light_stimulus_type, next_house_light_event_time - event_time, NOISE, dt)
                # Poisson sequence responses (not fully working yet)
                # responses = respond(active_timer_value, event_time, next_house_light_event_time, simulation_subplot, idx)
                '''
                for path in active_timer_value:
                    path_length = event_time + path.size * dt
                    if len(path) != 0:
                        simulation_subplot.plot(np.linspace(event_time,path_length, num=len(path)), path, c=colors[next_stimulus_type], alpha=0.3)
                '''
                if False:
                    if not (event_time==0):
                        response_time = beat_the_clock_threshold_time(active_timer_value, event_time, next_house_light_event_time, simulation_subplot, idx)
                       # print(f'response_time: {response_time}')
                       # print(f'start threshold: {START_THRESHOLD}')
                        reward = beat_the_clock_reward(next_house_light_event_time, response_time)
                        simulation_subplot.hlines(START_THRESHOLD,event_time,next_house_light_event_time, color="green", alpha=0.8)
                        
                        # TODO: This is code smell
                        START_THRESHOLD = change_response_threshold(START_THRESHOLD, RESPONSE_THRESHOLD_LEARNING_RATE, btc_reward, reward)
                        
                        btc_reward[idx]=reward
                    
                update_and_reassign_ramps(timer, house_light_timer_value, active_ramp_indices, next_house_light_stimulus_type, stimulus_type, ax2, idx)
                # for value in house_light_hierarchical_value:
                    # ax2.plot([next_house_light_event_time], [value], marker='o',c=colors[next_stimulus_type], alpha=0.2) 
                for i, val in zip(active_ramp_indices, house_light_timer_value):
                    if timer.terminating_events[i] == next_house_light_stimulus_type and timer.initiating_events[i] == stimulus_type or i in timer.free_ramps:
                        # if i in timer.free_ramps:
                        #    timer_plot_legend_free[stimulus_type] = simulation_subplot.plot([event_time,next_house_light_event_time], [0, val], linestyle='--', c=colors[next_stimulus_type])
                        
                        # else:
                        simulation_subplot.plot([event_time,next_house_light_event_time], [0, val], linestyle='--', c=colors[next_stimulus_type])
                        if (val<STOP_THRESHOLD and val>START_THRESHOLD):
                            simulation_subplot_legend[stimulus_type] = simulation_subplot.plot([event_time,next_house_light_event_time], [0, val],   c=colors[next_stimulus_type])
                            simulation_subplot.plot([next_house_light_event_time], [val], marker='o', c=colors[next_stimulus_type], markeredgecolor='black', markeredgewidth=1, alpha=0.2) 
                            
                        '''
                        if val < 5:
                            ax2.plot([next_house_light_event_time], [val], marker='o',c=colors[next_stimulus_type], alpha=0.5) 
                       '''
                # Contiue to the next event in the house light interval
                house_light_idx+=1
            else:
                house_light_interval=False

simulation_subplot.legend(handles=[simulation_subplot_legend[1][0], simulation_subplot_legend[1][0], simulation_subplot_legend[0][0], simulation_subplot_legend[0][0], simulation_subplot_legend[0]], labels=["A-available ramps", "A-assigned ramps", "B-available ramps", "B-assigned ramps", "attention period"], loc='lower left')

    
    