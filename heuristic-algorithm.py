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
# import sys
import random # TODO: Just use np.random instead of random.random
# TODO: Specify the seed so random things are generated again

from timer_module import TimerModule as TM
# from labellines import labelLine, labelLines
# from scipy.stats import invgauss   
# from scipy.stats import trim_mean
# # from statistics import NormalDist
# import matplotlib.colors as mcolors
# import networkx as nx
# import time
# import scipy.signal 



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

def activationAtIntervalEndEulerian(timer, ramp_index, interval_length, c, dt):
    # Simulate DDM process for activation amount
    # Change act to activation
    
    T = int(interval_length/dt)
    """
    acts = np.ones([len(ramp_index), T]) * weight * dt
    
    arr = np.random.normal(0,1,T) * noise * np.sqrt(dt)
    
    drift_arr = np.ones(T) * weight * dt
    act_arr = drift_arr + arr
    cum_act_arr = np.cumsum(act_arr)
   
    hit_time = np.argmax(cum_act_arr>threshold) 
    
    for i in range(0, len(ramp_index)):
        for t in range(1,T):
            acts[i][t] = acts[i][t-1] + c * np.sqrt(acts[i][t-1]) * np.random.normal(0, 1) * math.sqrt(interval_length)
    """
    
    acts = []
    for ramp in ramp_index:
        arr = np.random.normal(0,1,T) * c * np.sqrt(dt)
        
        drift_arr = np.full(T, timer.timers[ramp] * dt)
        act_arr = drift_arr + arr
        
        cum_act_arr = np.cumsum(act_arr)
        
        hit_time = np.argmax(cum_act_arr>1) 
        acts.append(cum_act_arr[:hit_time])
    return acts

# Generate when it will cross threshold
# Paramters:
#   weight: scalar,
#   threshold: scalar,
#   noise: scalar,
def generate_hit_time(weight, threshold, noise, dt, plot=False):   
    # Alternative method for hitting time
    T = int(((threshold/weight)+50)/dt)
   
    # Rename to be clearer
    arr = np.random.normal(loc=0,scale=1,size=T) * noise * np.sqrt(dt)
    
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
    # K is a global declared later on, tyically == 5
    
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
    # print('===')
    # print(f'{event_time} / {next_event}')
    for response_period in response_periods:
        responses.extend([int(i) for i in r if (i>response_period[0] and i<response_period[1] and i<next_event and i>event_time)])
        # print(r)
    
    ax1.plot(responses, np.ones(len(responses)), 'x') 
    # responses and ax1.text(responses[0],1.2,str(idx))
    return responses

# Gives reward for reward array based on the offset from the event occurance
def timing_reward(offset):
    return 1/(2**(.1*offset))
    # \frac{1}{2^{-x}}

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
                    #plot_early_update_rule(start_time, end_time, timer_weight, T, event_type, value)
                    timer_weight = earlyUpdateRule(value, timer.timerWeight(idx), timer.learningRate(idx))
                    # timer.setTimerWeight(timer_weight, idx)
                        
                else:
                    ''' Late Update Rule '''
                    timer_weight = lateUpdateRule(value, timer.timerWeight(idx), timer.learningRate(idx))
                timer.setTimerWeight(timer_weight, idx)   

def relative_to_absolute_event_time(relative_time_events):
    # TODO: Change this to a simple cumsum on the first column
   absolute_time_events = relative_time_events   
   for i in range (1,NUM_EVENTS):
       absolute_time_events[i][0] = relative_time_events[i-1][0] + relative_time_events[i][0]
   return absolute_time_events  
            
def beat_the_clock_reward(event_time, response_time):
    if (response_time - event_time) > 0:
        return 0
    # TODO: use numpy exp
    # TODO: make 0.4 a hyperparameter
    return math.exp(0.4 * (response_time - event_time))

def beat_the_clock_threshold_time(timer_value, event_time, next_event, ax1, idx):
    # Given all ramp vaues, return first time when K active ramps cross start threshold
    # TODO: Comment all of this to give a better sense of whats going on
    # Find start threshold times for each ramp
    start_threshold_times = start_threshold_time(timer_value, next_event-event_time)
    start_threshold_times += event_time
    start_threshold_times.sort()
    start_threshold_times = np.vstack((start_threshold_times, np.ones(len(start_threshold_times)))).T
    
    # Find stop threshold times for each ramp
    stop_threshold_times = stop_threshold_time(timer_value, next_event-event_time)
    stop_threshold_times += event_time
    # TODO: Specify the dimension we're sorting on
    stop_threshold_times.sort()
    # TODO: Do np.fill(-1) here
    stop_threshold_times = np.vstack((stop_threshold_times, (-1* np.ones(len(stop_threshold_times))))).T
    
    # TODO: Comment this out
    # Zip start and stop times
    start_stop_pairs = np.vstack((start_threshold_times, stop_threshold_times))
    
    
    # TODO: use np.sort axis=0
    start_stop_pairs = start_stop_pairs[start_stop_pairs[:, 0].argsort()]
    # print(start_stop_pairs)
    k = 0
    k_o = 0 # Old value of k

    # Form list of start and stop events, sorted by time (a1, sig1, a2, a3, sig2, sig3 etc)
    # Loop through all, if start event, k++, else, k--
    # Identify all periods of k > K
    # Fill with Poisson seq (samples then add the start time to all of them)
    # once theyre greater than the boundary where they stop, throw them out
    response_period_start = 100
    for jdx, time in enumerate(start_stop_pairs):
        k+=time[1]
        # print(f'k: {k} \t time: {time[0]}')
        # We're entering a response period
        if k_o < K and k >= K:
            response_period_start = time[0]
            break
    
    return response_period_start
    

def change_response_threshold(response_threshold, learning_rate, btc_reward, reward):
    # delta = learning_rate*(1-response_threshold)
    # TODO: Try reward now - running average of rewards
    # Want to have a function that makes big jumps when very wrong and little jumps as it starts to be right
    
    # Or just go back to the first stop threshold once you're late
    # Or do hill climbing algorithm with tiny steps
    #return response_threshold + (1/(1+np.exp(1-response_threshold)))
    
    # need to keep track of how early or late we are 
    return response_threshold + learning_rate*(1-response_threshold) # np.tanh(100 * (reward - btc_reward[-1])) * (1-response_threshold)

def reproduce_sequence(timer, events, reproduced_sequence_plot):
    # TODO: Preallocate this memory in numpy arrays 
    # make less hacky
    initiating_terminating_pairs = []
    repro_sequence = []
    # Transform event list into pairs of initiating and terminating events
    for i in range(0, len(events)-2):
        event_1 = events[i]
        event_2 = events[i+1]
        
        stim_1 = event_1[2]
        stim_2 = event_2[2]
        
        # change this to not mutate the array itself
        initiating_terminating_pairs.append([stim_1,stim_2])
        
    # Get all A-initiating ramps and sort them
    weight_idx = np.where(timer.initiating_events == stim_1)
    weights = np.flip(np.sort([timer.timerWeight(weight_idx)]))
    
    # Calculate threshold time
    for weight in weights:
        threshold_time = 1/weight
        # # Add time to reproduced sequence
        repro_sequence.append(threshold_time)
        reproduced_sequence_plot.vlines(threshold_time, 0,1, label="v")
    # Plot reproduced sequence
    return 0


def reproduce_sequence_(timer, events, reproduced_sequence_plot):
    # make less hacky
    initiating_terminating_pairs = []
    repro_sequence = []
   
    # Get the first event weights
    weight_idx = np.where(timer.initiating_events == events[0][2])
    
    # asending sort
    weights = np.flip(np.sort([timer.timerWeight(weight_idx)]))
    
    # weights = np.flip(np.sort([timer.timerWeight(weight_idx)]))
    weights_set=set(timer.terminating_events.flatten())
    
    if -1 in weights_set:
        weights_set.remove(-1)
    
    avg_weights = np.empty(len(weights_set))
    # timer.terminating_events[idx] == next_stimulus_type and timer.initiating_events[idx] == stimulus_type
    
    # plot red bars
    for weight in weights:
          threshold_time = 1/weight
          # # Add time to reproduced sequence
          repro_sequence.append(threshold_time)
          # reproduced_sequence_plot.vlines(threshold_time, 0,1, label="v", color='r')
   
    # print(weights_set)
    for idx, term_event in enumerate(weights_set):
        idxs_term = np.where(timer.terminating_events == term_event)
        idxs_init = np.where(timer.initiating_events == events[0][2])
        idxs = np.intersect1d(idxs_term, idxs_init)
        # idxs = weight_idx[np.in1d(weight_idx, term_idxs)]
        # print(f'weights for {idx}: {timer.timerWeight(idxs)}')
        # print(f'indexes: {idxs}')
        non_1_weights = []
        for weight in timer.timerWeight(idxs)[timer.timerWeight(idxs) != 2]:
            non_1_weights.append(weight)
        # trimmed = trim_mean(timer.timerWeight(idxs), 0.3)
        avg_weight = sum(non_1_weights)/len(non_1_weights) # np.nanmean(timer.timerWeight(idxs))
        avg_weights[idx] = avg_weight
        
    # print(f'avg weights: {avg_weights}')
    # Go through each weight and average the assigned ramps
    for weight in avg_weights:
        #timer.timerWeight(
        threshold_time = 1/weight
        # # Add time to reproduced sequence
        #repro_sequence.append(threshold_time)
        # plot blue bars
        reproduced_sequence_plot.vlines(threshold_time, 0,2, label="v", color='g')
    # Plot reproduced sequence
    return 0

# def expected_gain(target_time, current_time, short_length, long_length, short_prob, long_prob, reward_schedule):
    
#     g_-Ts = -.2 # incorrect short trial
#     g_Ts = 1 # correct short trial
#     g_-Tl = -.2 # incorrect long trial
#     g_Tl = 1.2 # correct long trial
    
#     p-Ts = short_prob
#     p-Tl = long_prob
#     # w = 
#     # dist = NormalDist(mu=target_time,  )
    


''' Global variables '''
dt = 0.1
N_EVENT_TYPES= 10 # Number of event types (think, stimulus A, stimulus B, ...)
# NUM_EVENTS=17#  Total amount of events
Y_LIM=2 # Vertical plotting limit
NOISE=0.0 # Internal noise - timer activation
LEARNING_RATE=.8 # Default learning rate for timers
STANDARD_INTERVAL=20 # Standard interval duration 
K = 5 # Amount of timers that must be active to respond
START_THRESHOLD=.5 # Response start threshold
STOP_THRESHOLD=1.2 # Response stop threshold
PLOT_FREE_TIMERS=False
ERROR_ANALYSIS_RESPONSES=[]
BEAT_THE_CLOCK = False
colors = [[1,0,0], [0,1,0], [0,0,1], [1,1,0], [0,1,1], [1,0,1],[.46,.03,0], [.1,.3,.2], [.2,.7,.2], [.5,.3,.6], [.7,.3,.4]]# list(mcolors.CSS4_COLORS) # Color support for events
ALPHABET_ARR = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','AA','BB','CC'] # For converting event types into letters 
# ramp_graph=nx.Graph()
SAVE_RAMP_NETWORK_ANIMATION_FRAMES = False
RESPONSE_THRESHOLD_LEARNING_RATE = .6
NUM_RAMPS = 300
RAMPS_PER_EVENT = 10
reward_window_plot = 1600
reward_window_sim = 1600
x_lb = np.linspace(-reward_window_plot * dt,0, reward_window_plot)
exp_weighted_average = np.exp(x_lb * .01)
plt.plot(x_lb, exp_weighted_average)
event_data = []


random_seq = False

# Begin simulation
if random_seq:
    
    seq_length = 3
    random_samples = TM.getSamples(seq_length, num_normal = 3, seed = 12, scale_beg = 20, scale_end = 50)
    event_data = [[0,0,0]]
    for sample in random_samples:
        event_data.append([sample[0], sample[1], sample[2]])
        
    # event_data = event_data + event_data[1:] + event_data[1:] + event_data[1:]
    # event_data = np.asarray(event_data)
    HOUSE_LIGHT_ON = [*range(0,seq_length-1,1)] + [*range(seq_length,(seq_length*2)-1,1)] + [*range(seq_length*2,(seq_length*3)-3,1)] + [*range(seq_length*3,(seq_length*4)-3,1)]
    
    event_data = TM.getEvents(num_samples=seq_length, num_normal = 2, deviation=2, num_exp = 0, repeat = 3, scale_beg = 20, scale_end=30)
    
else:
    # TODO: Change this to arange
    HOUSE_LIGHT_ON = [*range(0,2,1)] + [*range(3,5,1)] + [*range(7,9,1)] + [*range(11,13,1)] + [*range(12,14,1)]
    # HOUSE_LIGHT_ON = [*range(0,2,1)] + [*range(4,6,1)] + [*range(8,10,1)] 
    
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
    
    # HOUSE_LIGHT_ON = [*range(0,10,1)] + [*range(11,21,1)] #+ [*range(24,35,1)]
    # event_data = np.asarray([[0,1,1], [50,0,0], [25,1,2],
    #                       [50,0,3], [25,1,4], [50,0,5],
    #                       [30,1,6], [60,0,7], [10,1,8],
    #                       [100,0,9], [25,1,10], 
                          
    #                       [25,1,1], [50,0,0], [25,1,2], 
    #                       [50,0,3], [25,1,4], [50,0,5],
    #                       [30,1,6], [60,0,7], [10,1,8],
    #                       [100,0,9], [25,1,10], 
                          
    #                       [25,1,1], [50,0,0], [25,1,2], 
    #                       [50,0,3], [25,1,4], [50,0,5],
    #                       [30,1,6], [60,0,7], [10,1,8],
    #                       [100,0,9], [25,1,10], ])
                          
    
    
# TODO: Make start threhsolds an array of values
seq_len =  4
repeat_num = 3
penalty=.02

NUM_EVENTS = len(event_data) 
btc_reward=np.empty(NUM_EVENTS)

error_arr = np.zeros(NUM_EVENTS)
event_data = relative_to_absolute_event_time(event_data)
event_data[0][2] = event_data[seq_len][2]


# Last event, time axis for plotting        
T = event_data[HOUSE_LIGHT_ON[-1]+1][0] + 10

# Timer with 100 (or however many you want) ramps, all initialized to be very highly weighted (n=1)
timer=TM(1,NUM_RAMPS)


simple_learning_fig = plt.figure()
# simple_learning_fig.suptitle('Simple Learning Sequence', fontsize=16)
ax1 = simple_learning_fig.add_subplot(311)
ax2 = simple_learning_fig.add_subplot(312, sharex = ax1)
ax3 = simple_learning_fig.add_subplot(313, sharex = ax1)
ax1.set_ylim([0,Y_LIM])
ax1.set_xlim([0,T])

ax2.set_ylim([0,1])
ax2.set_xlim([0,T])

# ax3.set_ylim([0,1])
ax3.set_xlim([0,T])

ax2.plot()

reward_arr_plot = np.zeros(int(event_data[HOUSE_LIGHT_ON[-1]+1][0] / dt))

timer_plot_legend_free = {}
timer_plot_legend_assigned = {}

# Initialize a reward arr that has a small amount of reward at each time step
reward_arr = np.zeros(int(event_data[HOUSE_LIGHT_ON[-1]+1][0]/dt))

reward_x_axis = np.linspace(0,event_data[HOUSE_LIGHT_ON[-1]+1][0]/dt,reward_arr.shape[0])

# Define hidden states
hidden_states = [175, 325, 475]
for hidden_state in hidden_states:
    reward_arr[hidden_state] = 2
    # for i in range(1,20):
        # reward_arr[hidden_state-i] = 1 - (i*.1)
        # reward_arr[hidden_state-i] = timing_reward(i)
    #    reward_arr[hidden_state-i] = 2
# For event, add a large amount of reward at the event and a little right before it 
for event in event_data:
    # reward_arr[event[0]] = 1
    # for i in range(1,5):
        # reward_arr[event[0]-i] = 1 - (i*.1)
        # reward_arr[event[0]-i] = timing_reward(i)
    reward_arr[event[0] - 1] = 2

reward_arr[0] = 0 
    
ax2.plot(reward_x_axis, reward_arr, label="reward")
for state in hidden_states:
    ax2.plot(state, .9, marker = '*', color='r', label="hidden state")
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
    if idx < (NUM_EVENTS - 1):
        ax1.text(event[0],2.1,ALPHABET_ARR[int(event_data[idx+1][2])])
        ax1.vlines(event_time, 0,Y_LIM, label="v", color=colors[next_stimulus_type])
    
    # else:
    #    ax1.text(event[0],2.1,'End')
            
    if house_light:
        # Plot house light bar
        house_light_bar = ax1.plot([event_time, next_event], [1.9, 1.9], 'k-', lw=4)  
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
                house_light_hierarchical_value = activationAtIntervalEndHierarchical(timer, initiating_active_indices, next_house_light_stimulus_type, next_house_light_event_time - event_time, NOISE, ax1)
                house_light_responding_values = activationAtIntervalEnd(timer, initiating_active_indices, next_house_light_event_time - event_time, NOISE)
                
                active_timer_value = activationAtIntervalEndHierarchical(timer, active_ramp_indices, next_house_light_stimulus_type, next_house_light_event_time - event_time, NOISE, dt)
                
                if BEAT_THE_CLOCK:
                    if not (event_time==0):
                        response_time = beat_the_clock_threshold_time(active_timer_value, event_time, next_house_light_event_time, ax1, idx)
                        reward = beat_the_clock_reward(next_house_light_event_time, response_time)
                        ax1.hlines(START_THRESHOLD,event_time,next_house_light_event_time, color="green", alpha=0.8)
                        
                        # TODO: This is code smell
                        START_THRESHOLD = change_response_threshold(START_THRESHOLD, RESPONSE_THRESHOLD_LEARNING_RATE, btc_reward, reward)
                        
                        btc_reward[idx]=reward
                if idx > 0:
                    responses = respond(house_light_responding_values, event_time, next_house_light_event_time, ax1, idx)
                    reward = reward_arr[responses]
                    pos_reward = np.where(reward > 0)[0]
                
                    # if pos_reward.shape[0] > 0:
                    #     ax2.vlines(responses[pos_reward], reward[pos_reward])
                    
                    # print(reward) 
                    # print(responses)
                    
                    for i,r in enumerate(reward):
                       if int(responses[i]/dt) < reward_arr_plot.shape[0]:
                           reward_arr_plot[int(responses[i]/dt)] = r - penalty
                    
                    reward_penalty = np.full(reward.shape[0], penalty)
                    
                    reward_end = np.sum(reward) + np.sum(reward_penalty)
                    ax2.plot(responses,reward, marker='x', color = 'g')
                    
                    # Look back and get exp moving average to get non-causal reward rate
                    for t in range (int(reward_arr_plot[i]-reward_window_plot/dt), int(reward_arr_plot[i]/dt) ):
                        exp_reward_rate = np.mean(reward_arr_plot[-reward_window_plot:] * exp_weighted_average)
                        # Positiv reward, respond more
                        if exp_reward_rate > 0:
                            START_THRESHOLD -= 0.0001
                        
                        # Negative reward, respond less
                        if exp_reward_rate < 0:
                            START_THRESHOLD += 0.0001
                            
                        
                    ax1.plot(event_time+i,START_THRESHOLD, marker='.', color = 'g')
                            
                    
                        # np.mean(reward_arr_plot[i-window:i] * kern)
                    
                update_and_reassign_ramps(timer, house_light_timer_value, active_ramp_indices, next_house_light_stimulus_type, stimulus_type, ax1, idx)
                
                # response_time = (event_time + next_house_light_event_time // 2) 
                # reward = reward_arr[responses]
                
                '''
                if reward.shape[0] > 0:
                    penalty = np.full(reward.shape[0], penalty)
                
                    reward = np.subtract(reward,penalty)
                '''
                
                # for index, r in enumerate(reward):
                #     reward_arr_plot[int(event_time/dt) + int(index/dt)] = r
                # reward_arr_plot.extend(reward)
                
                # pos_reward = np.where(reward > 0)[0]
                
                # # if pos_reward.shape[0] > 0:
                # #     ax2.vlines(responses[pos_reward], reward[pos_reward])
                
                # # print(reward) 
                # # print(responses)
                
                # for i,r in enumerate(reward):
                #    if int(responses[i]/dt) < reward_arr_plot.shape[0]:
                #        reward_arr_plot[int(responses[i]/dt)] = r - penalty
                
                # reward_penalty = np.full(reward.shape[0], penalty)
                
                # reward_end = np.sum(reward) + np.sum(reward_penalty)
                # # print(reward_end)
                # # print(reward_end)
                
                # # ax1.plot([response_time],[.8], marker='x', color = 'g')
                # # in between responses an first interval, pad with zeros
               
                # # if responses
                # # zeros_reward_arr = np.linspace(0, event_time, (responses[0]-event_time) / dt)
                
                # ax2.plot(responses,reward, marker='x', color = 'g')
                
                for value in house_light_hierarchical_value:
                    ax1.plot([next_house_light_event_time], [value], marker='o',c=colors[next_stimulus_type], alpha=0.2) 
                for i, val in zip(active_ramp_indices, house_light_timer_value):
                    if timer.terminating_events[i] == next_house_light_stimulus_type and timer.initiating_events[i] == stimulus_type or i in timer.free_ramps:
                        if i in timer.free_ramps:
                            timer_plot_legend_free[stimulus_type] = ax1.plot([event_time,next_house_light_event_time], [0, val], linestyle='--', c=colors[next_stimulus_type])
                        
                        else:
                            if (val<STOP_THRESHOLD and val>START_THRESHOLD):
                                timer_plot_legend_assigned[stimulus_type] = ax1.plot([event_time,next_house_light_event_time], [0, val],   c=colors[next_stimulus_type])
                                ax1.plot([next_house_light_event_time], [val], marker='o', c=colors[next_stimulus_type], markeredgecolor='black', markeredgewidth=1, alpha=0.2) 
                            
                            
                # Contiue to the next event in the house light interval
                house_light_idx+=1
            else:
                house_light_interval=False
  
window_size = 100
def gkern(l=window_size, sig=1.):
    """\
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    
    # gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    # kernel = np.outer(gauss, gauss)
    return gauss # kernel / np.sum(kernel)

kernel = gkern()

threshold_times = []

ax1.set_ylim([0,Y_LIM])
ax1.set_xlim([0,400])
ax1.set_ylabel("Activation")
ax1.set_xlabel("Time")


# reward_sliding_windows = np.lib.stride_tricks.sliding_window_view(reward_arr_plot, window_size)

reward_arr_x = np.linspace(0,int(event_data[HOUSE_LIGHT_ON[-1]+1][0]), reward_arr_plot.shape[0])
reward_sliding_windows_vals = np.zeros(reward_arr_plot.shape[0])

for i in range(window_size):
    reward_sliding_windows_vals[i] =  np.sum(reward_arr_plot[:i] * kernel[:i])
    
    if i == 0:  
        reward_sliding_windows_vals[-1] = np.sum(reward_arr_plot[-1:]) # * kernel[:1])
    else:
        reward_sliding_windows_vals[-i] = np.sum(reward_arr_plot[-i:]) # * kernel[:i])

#for i in range(window_size, reward_arr_plot.shape[0] - window_size):
    #reward_sliding_windows_vals[i] =  np.sum(reward_arr_plot[i:i+window_size]) #* kernel)

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='full')
    return y_smooth

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

''' Various plotting for reward rate estimation ''' 

# win_len = 20
# for i in range(0, 5250-win_len):
#     reward_arr_plot[i] = np.mean(reward_arr_plot[i:i+win_len])

#for i in 
# ax3.plot(reward_arr_x, smooth(reward_arr_plot, 40))
# moving_average_reward = moving_average(reward_arr_plot, 8)
moving_avg_1 = []
moving_avg_2 = []
moving_avg_3 = []


window_l = 100
window_r = 100
ax1.plot([0,event_data[HOUSE_LIGHT_ON[-1]+1][0]], [START_THRESHOLD, START_THRESHOLD], color = 'green', alpha=.5)
ax1.plot([0,event_data[HOUSE_LIGHT_ON[-1]+1][0]], [STOP_THRESHOLD, STOP_THRESHOLD], color = 'red', alpha=.5)

for i in range(window_l,reward_arr_plot.shape[0]-window_r):
    moving_avg_1.append(np.sum(reward_arr_plot[i-window_l:i+window_r])/window_r)
    moving_avg_2.append(np.sum(reward_arr_plot[i-(window_l*2):i+(window_r*2)])/(window_r*2))
    moving_avg_3.append(np.sum(reward_arr_plot[i-(window_l*4):i+(window_r*4)])/(window_r*4))
    
#ax3.plot(np.linspace(0,int(event_data[HOUSE_LIGHT_ON[-1]+1][0]), len(moving_avg_1)), moving_avg_1)
#ax3.plot(np.linspace(0,int(event_data[HOUSE_LIGHT_ON[-1]+1][0]), len(moving_avg_2)), moving_avg_2)
#ax3.plot(np.linspace(0,int(event_data[HOUSE_LIGHT_ON[-1]+1][0]), len(moving_avg_3)), moving_avg_3)
# ax3.plot(np.linspace(0,len(moving_avg_1)*dt, len(moving_avg_1)), moving_avg_1)

test_window_size = 500
window_avg_values=[]

import scipy.stats as st
def gkern(kernlen=20, nsig=3):
    """Returns a 2D Gaussian kernel."""

    x = np.linspace(-nsig, nsig, kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d/kern2d.sum()


for i in range(0, reward_arr_plot.shape[0]-test_window_size):
    window_avg = np.sum(reward_arr_plot[i:i+test_window_size])/test_window_size
    window_avg_values.append([i+int(test_window_size/2), window_avg])

avg_vals = []
kern = gkern(reward_window_plot)[0]


# Pad with zeros so we plot our avg centered with the other plots
for i in range(int(reward_window_plot/2 * dt)):
    avg_vals.append(0)
    

for i in range(reward_window_plot,reward_arr_plot.shape[0], int(1/dt)):
    avg = np.mean(reward_arr_plot[i-reward_window_plot:i] * exp_weighted_average)# np.mean(reward_arr_plot[i-window:i] * kern)
    avg_vals.append(avg)

# for i in range(int(reward_window/2 * dt)):
#     avg_vals.insert(0,0)




    
# ax3.plot([i[0]*dt for i in window_avg_values], [i[1] for i in window_avg_values])

# filtered = signal.sosfilt(sos,reward_arr_plot)
# ax3.plot(avg_vals, label = [f"Gaussian Smoothed Reward, w={reward_window_plot}"])

''' Newest reward rate estimation ''' 
reward_estimation = [0]
tau = 200
for i in range(1, reward_arr_plot.shape[0]):
        R_t = reward_estimation[i-1] + ((dt * (-reward_estimation[i-1]/tau) + reward_arr_plot[i]/tau))
        reward_estimation.append(R_t)
        
ax3.plot(reward_arr_x, reward_estimation)
'''
t = 1:100; seq = sin(t);
plot(t,seq)
window_width = 90; 
for window_bottom = 1:100-window_width+1
    window_top = window_bottom + window_width - 1;
    window_vals(window_bottom) = sum(seq(window_bottom:window_top))/window_width;
end

plot(round(window_bottom+window_width/2),window_vals,'k')
x axis values for the plot need to be fixed

'''

# tau * dR/dt = r - R
 

                