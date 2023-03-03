#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 14:10:51 2023

@author: Rob Klock
"""
import numpy as np
import matplotlib.pyplot as plt
import math
# import sys
import random # TODO: Just use np.random instead of random.random
# TODO: Specify the seed so random things are generated again

from timer_module import TimerModule as TM

def relative_to_absolute_event_time(relative_time_events, NUM_EVENTS):
    # TODO: Change this to a simple cumsum on the first column
   absolute_time_events = relative_time_events   
   for i in range (1,NUM_EVENTS):
       absolute_time_events[i][0] = relative_time_events[i-1][0] + relative_time_events[i][0]
   return absolute_time_events  

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
    # print(f'ramp_index: {ramp_index}')
    for i in range (0, len(act)):
        act[i] = act[i] + c * np.sqrt(act[i]) * np.random.normal(0, 1) * math.sqrt(interval_length)
    f_act = act * act
    
    # print(f'act: {act}')
    
    #for timer_act in act:
    #    hier_act = timer_act + 1/((act.size)-1) * delta * f_act[act != timer_act].sum()

    return act

def update_and_reassign_ramps(timer, timer_values, timer_indices, next_stimulus_type, stimulus_type, ax2, external_idx, NUM_RAMPS, RAMPS_PER_EVENT, sequence_code = '', v0=1.0, z = 1, bias = 1, plot = False):
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

def start_threshold_time(act_at_interval_end, interval_length,START_THRESHOLD):
    # Time of ramp hitting start threshold
    # TODO: get rid of magic numbers here
    angle = np.arctan(act_at_interval_end/interval_length)
    beta = 3.14159 - (1.5708 + angle)
    return START_THRESHOLD * np.tan(3.14159 - (1.5708 + angle))

def stop_threshold_time(act_at_interval_end, interval_length,STOP_THRESHOLD):
    # Time of ramp hitting stop threshold
    angle = np.arctan(act_at_interval_end/interval_length)
    beta = 3.14159 - (1.5708 + angle)
    return STOP_THRESHOLD * np.tan(3.14159 - (1.5708 + angle))

def generate_responses(interval_length, dt, num_samples):
    num_samples = int(interval_length / dt)
    responses = np.random.exponential(1, num_samples)
    return responses


def respond(timer_value, event_time, next_event, START_THRESHOLD, STOP_THRESHOLD, dt, num_samples, K, ax1, idx):
    # Given all ramp vaues, respond when K are between start and stop range
    # K is a global declared later on, tyically == 5
    
    # Find start threshold times for each ramp
    start_threshold_times = start_threshold_time(timer_value, next_event-event_time, START_THRESHOLD)
    start_threshold_times += event_time
    start_threshold_times.sort()
    start_threshold_times = np.vstack((start_threshold_times, np.ones(len(start_threshold_times)))).T
    
    # Find stop threshold times for each ramp
    stop_threshold_times = stop_threshold_time(timer_value, next_event-event_time, STOP_THRESHOLD)
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
    
    r = list(generate_responses(next_event-event_time, dt, num_samples))
    r.insert(0, event_time)
    r=list(np.cumsum(r))
    # print('===')
    # print(f'{event_time} / {next_event}')
    for response_period in response_periods:
        responses.extend([i for i in r if (i>response_period[0] and i<response_period[1] and i<next_event and i>event_time)])
        
    
    ax1.plot(responses, np.ones(len(responses)), 'x') 
    # responses and ax1.text(responses[0],1.2,str(idx))
    return responses


def main(parameters):
    dt = parameters["dt"]
    N_EVENT_TYPES= parameters["N_EV_TYPES"] # Number of event types (think, stimulus A, stimulus B, ...)
    Y_LIM=2 # Vertical plotting limit
    NOISE=parameters["NOISE"] # Internal noise - timer activation
    LEARNING_RATE=parameters["lr"] # Default learning rate for timers
    STANDARD_INTERVAL=20 # Standard interval duration 
    K = parameters["k"] # Amount of timers that must be active to respond
    START_THRESHOLD=parameters["start_thresh"]# Response start threshold
    STOP_THRESHOLD=parameters["stop_thresh"] # Response stop threshold
    PLOT_FREE_TIMERS=False
    ERROR_ANALYSIS_RESPONSES=[]
    BEAT_THE_CLOCK = False
    colors = [[1,0,0], [0,1,0], [0,0,1], [1,1,0], [0,1,1], [1,0,1],[.46,.03,0], [.1,.3,.2], [.2,.7,.2], [.5,.3,.6], [.7,.3,.4]]# list(mcolors.CSS4_COLORS) # Color support for events
    ALPHABET_ARR = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','AA','BB','CC'] # For converting event types into letters 
    RESPONSE_THRESHOLD_LEARNING_RATE = .6
    NUM_RAMPS = parameters["num_ramps"]
    RAMPS_PER_EVENT = parameters["ramps_per_event"]
    reward_window_plot = 1600
    reward_window_sim = 1600
    x_lb = np.linspace(-reward_window_plot * dt,0, reward_window_plot)
    exp_weighted_average = np.exp(x_lb * .01)
    # plt.plot(x_lb, exp_weighted_average)
    event_data = []

    random_seq = False
    seq_length = 3

    # Initialize events
    if random_seq:
        random_samples = TM.getSamples(seq_length, num_normal = 3, seed = 12, scale_beg = 20, scale_end = 50)
        event_data = [[0,0,0]]
        for sample in random_samples:
            event_data.append([sample[0], sample[1], sample[2]])
            
        HOUSE_LIGHT_ON = [*range(0,seq_length-1,1)] + [*range(seq_length,(seq_length*2)-1,1)] + [*range(seq_length*2,(seq_length*3)-3,1)] + [*range(seq_length*3,(seq_length*4)-3,1)]
        event_data = TM.getEvents(num_samples=seq_length, num_normal = 2, deviation=2, num_exp = 0, repeat = 3, scale_beg = 20, scale_end=30)
        
    else:
        # TODO: Change this to arange
        HOUSE_LIGHT_ON = [*range(0,2,1)] + [*range(3,5,1)] + [*range(7,9,1)] + [*range(11,13,1)] #+ [*range(12,14,1)]
        
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
        
    # TODO: Make start threhsolds an array of values
    seq_len =  4
    repeat_num = 3
    penalty=.02

    NUM_EVENTS = len(event_data) 
    btc_reward=np.empty(NUM_EVENTS)

    error_arr = np.zeros(NUM_EVENTS)
    event_data = relative_to_absolute_event_time(event_data, NUM_EVENTS)
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
    # ax4 = simple_learning_fig.add_subplot(314)

    ax1.set_ylim([0,Y_LIM])
    ax1.set_xlim([0,T])

    ax2.set_ylim([0,1])
    ax2.set_xlim([0,T])

    # ax3.set_ylim([0,1])
    ax3.set_xlim([0,T])

    # ax2.plot()

    reward_arr_plot = np.zeros(int(event_data[HOUSE_LIGHT_ON[-1]+1][0] / dt))

    timer_plot_legend_free = {}
    timer_plot_legend_assigned = {}

    # Initialize a reward arr that has a small amount of reward at each time step
    reward_arr = np.zeros(int(event_data[HOUSE_LIGHT_ON[-1]+1][0]/dt))

    reward_x_axis = np.linspace(0,event_data[HOUSE_LIGHT_ON[-1]+1][0]/dt,reward_arr.shape[0])

    # Define hidden states
    hidden_states = [175, 325, 475]
    for hidden_state in hidden_states:
        reward_arr[int(hidden_state/dt)] = 1
        
    # For event, add a large amount of reward at the event and a little right before it 
    for index, event in enumerate(event_data):
        # print(f'index: {index}, event: {event}')
        if index in HOUSE_LIGHT_ON or (index-1) in HOUSE_LIGHT_ON:
            if int(event[0]/dt) < reward_arr.shape[0]:
                # print(f'House Light: {index}, event: {event}')
                reward_arr[int(event[0]/dt)] = 1
                # print(f'Adding reward to indicies: {int((event[0] /dt)-30)}->{int(event[0]/dt)}')
                # print(f'Adding reward to indicies: {int((event[0] /dt)-60)}->{int((event[0]/dt)-30)}')
                reward_arr[int((event[0] /dt)-30):int(event[0]/dt)] = .5
                reward_arr[int((event[0] /dt)-60):int((event[0]/dt)-30)] = .25

    reward_arr[0] = 0 
        

    for state in hidden_states:
        ax2.plot(state, .9, marker = '*', color='r', label="hidden state")
        
    ''' Simulation Start '''
    # At each event e_i
    reward_est_vals = np.zeros(5000)
    run_twice = 2

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
                        responses = respond(house_light_responding_values, event_time, next_house_light_event_time, START_THRESHOLD, STOP_THRESHOLD, dt, seq_length, K, ax1, idx)
                        if run_twice>0 and len(responses) > 0:
                            # print(responses)
                            run_twice-=1
                            
                        reward = reward_arr[[int(r/dt) for r in responses]]
                        pos_reward = np.where(reward > 0)[0]
                        
                        for i,r in enumerate(reward):
                           if int(responses[i]/dt) < reward_arr_plot.shape[0]:
                               reward_arr_plot[int(responses[i]/dt)] = r - penalty
                 
                        # if reward.size > 0:
                        #     tau = 200
                        #     print(next_house_light_event_time - event_time)
                        #     # decay rate related to tau? 
                        #     estimation = reward_est_vals[-1] * ((1/(2))**((next_house_light_event_time - event_time )*dt))
                        #     print(reward_est_vals[-1] * ((1/(2))**((next_house_light_event_time - event_time )*dt)))
                            
                        #     live_reward_est = reward_est_vals[-1] + ((dt * (-reward_est_vals[-1]/tau) + max(reward)/tau))
                        #     reward_est_vals.append(live_reward_est)
                        #     print(live_reward_est)
                        #     print()
                            
                        #     ax3.plot([event_time + ((next_house_light_event_time - event_time)/ 2 )], [live_reward_est], marker='o')
                               
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
                        
                        tau = 200
                        
                    
                    tau=200
                    # print(f'event_time: {event_time}, next_time: {next_house_light_event_time}')
                    for i in range(int(event_time/dt)+1, int(next_house_light_event_time/dt)): # int(event_data[idx+1][0]/dt)):  #
                        
                        # if event_data[idx+1][0] == next_house_light_event_time:
                        #     reward_est_vals[i] = reward_est_vals[i-1]
                        # else:
                        R_t = reward_est_vals[i-1] + ((dt * (-reward_est_vals[i-1]/tau) + reward_arr_plot[i]/tau))
                        reward_est_vals[i] = R_t
                        
                        
                    update_and_reassign_ramps(timer, house_light_timer_value, active_ramp_indices, next_house_light_stimulus_type, stimulus_type, ax1, idx, NUM_RAMPS, RAMPS_PER_EVENT)
                    
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
        else:
            # print('hello')
            # print(f'event time {event_time}')
            # print(f'next house light: {event_data[idx+1][0]}')
            
            for i in range(int(event_time/dt), int(event_data[idx+1][0]/dt)):
                if i > reward_est_vals.shape[0] - 1:
                    break
                else:
                    R_t = reward_est_vals[i-1]
                    reward_est_vals[i:] = R_t

    window_size = 100


    threshold_times = []

    ax1.set_ylim([0,Y_LIM])
    ax1.set_xlim([0,400])
    ax1.set_ylabel("Activation")
    ax1.set_xlabel("Time")


    # reward_sliding_windows = np.lib.stride_tricks.sliding_window_view(reward_arr_plot, window_size)

    reward_arr_x = np.linspace(0,int(event_data[HOUSE_LIGHT_ON[-1]+1][0]), reward_arr_plot.shape[0])

    ax2.plot(reward_arr_x, reward_arr, label="reward")



    # reward_sliding_windows_vals = np.zeros(reward_arr_plot.shape[0])

    # for i in range(window_size):
    #     reward_sliding_windows_vals[i] =  np.sum(reward_arr_plot[:i] * kernel[:i])
        
    #     if i == 0:  
    #         reward_sliding_windows_vals[-1] = np.sum(reward_arr_plot[-1:]) # * kernel[:1])
    #     else:
    #         reward_sliding_windows_vals[-i] = np.sum(reward_arr_plot[-i:]) # * kernel[:i])

    def smooth(y, box_pts):
        box = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(y, box, mode='full')
        return y_smooth

    def moving_average(a, n=3) :
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n
    # ax3.plot(reward_arr_x, reward_est_vals, linestyle='--')
    ''' Newest reward rate estimation ''' 
    reward_estimation = [0]
    tau = 200
    for i in range(1, reward_arr_plot.shape[0]):
        R_t = reward_estimation[i-1] + ((dt * (-reward_estimation[i-1]/tau) + reward_arr_plot[i]/tau))
        reward_estimation.append(R_t)
            
    ax3.plot(reward_arr_x, reward_estimation)
    ax3.plot(reward_arr_x, reward_est_vals, linestyle='--')
    average_reward = np.mean(reward_est_vals)
    print(f"Average Reward: {average_reward}")
    return average_reward

params = {"dt":0.1,
              "N_EV_TYPES":10,
              "NOISE":0,
              "lr": 0.8,
              "k":5,
              "start_thresh":.7,
              "stop_thresh":1.2,
              "num_ramps":100,
              "ramps_per_event":10,
              }

reward_from_runs = []
num_repeat=3
for i in np.arange(.5,1,.1):
    trial_runs = []
    for repeat in range(num_repeat):
        params["start_thresh"] = i
        avg_reward = main(params)
        trial_runs.append(avg_reward)
    reward_from_runs.append(np.mean(np.array(trial_runs)))

plt.figure()
plt.plot(reward_from_runs)
plt.ylabel("average reward")
plt.xlabel("Run #")
 