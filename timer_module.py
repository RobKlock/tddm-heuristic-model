#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 13:14:22 2021
@author: Robert Klock
Class defining a timer module and related useful methods
"""
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.stats as stats
from scipy import signal
from scipy.optimize import minimize 
from scipy.stats import norm
from itertools import permutations
"""
Alternatively:
    3 components with different weights
        Flip a coin proportional to those weights 
        and draw a random sample from those individual 
        distributions. This works for just Gaussian mixtures
        
        Weighted sum of their CDFs. Then you can just pull from
        the CDF. If its less than the weight you draw from that sample
        
        N components of exponentials and Gaussians. 
        Normal variable, find out which weight range it falls into
        Then just draw a sample from the distribution that falls in line with
    
    https://pytorch.org/docs/stable/distributions.html?highlight=mixture#torch.distributions.mixture_same_family.MixtureSameFamily
    
    Package would roll the dice for you, give a number, and 
    draw the sample. Since we want to combine exponentials and normals,
    its better to build out from scratch but is still relatively simple
    
Initializing Timers:
    
    Semi-Markov Model
    Get a Stimulus Event 
        Assume that event is the mean of the distribution
        Update the mean with new samples 
    Then get another Stimulus and another Event
        Repeat from previous stimulus
    
    For new stimulus, calculate the likelihood of it belonging
    to each distribution and update the running average of the mean
    
    If its really off, create a new distribution
    Need some bias to avoid overfitting. Bias has 
    two components: one to keep track of distribution and 
    another to keep track of speed
    
    Default timer to capture new events and update to 
    record new events and probabilities 
    
    Keep track of variability to fit standard deviation
    
    First, choose a family
    
    Need to define and assitytggn distribution families
    
    
    Dont want to record everything or else youll just overfit
    
    Neural networks currently tend to just learn every frame of video or all data
    instead of learning a timer interval to expect an event
    
    DDMs have good properties: they are speed adaptable through bias,
    can represent means and standard deviations with them.
    
    Model is:
        As soon as a stimulus occurs, I start multiple potential timers. When the event occurs,
        I store information about which ramp was closest. Or you could allocate 5 timers for each event
        and adjust their fan to represent the event's standard deviation
        
    Rivest will write down a basic rule to use
    Events A and B and their independent to get started
    
"""

class TimerModule:
    
    ZEROS_BLOCK = np.zeros((4,4))
    BIAS_BLOCK = np.array([1.2, 1, 1.2, 1.25])
    def __init__(self,timer_weight = 1,n_timers=1):
        self.timers=np.empty(n_timers)
        self.timers.fill(timer_weight)
        self.timer_weight=timer_weight
        self.active_ramps=[]
        # a list of indices
        self.frozen_ramps=[]
        self.time_until=np.empty(n_timers)
        self.time_until.fill(-1)
        self.scores=np.zeros(n_timers)
        self.learning_rates=np.ones(n_timers)
        block = np.array([[2, 0, 0, -.4],
                          [self.timer_weight, 1, 0, -.4],
                          [0, .55, 2, 0],
                          [0, 0, .9, 2]])
        self.block = block
        self.score = 0.0
        self.learning_rate = 1
        self.event_dict = {}
        # big matrix 
        
        # or just use a  n_timers length list that keeps track of terminating event
        # use 0,1,NaN
        # use np.where
        # 3 columns
        # slope of ramp, assigned or not, initiaing events, terminating event, 
        self.terminating_events = np.full(n_timers, -1)
        self.initiating_events = np.full(n_timers, -1)
        self.stimulus_dict = {} # timers wit s_1 = e
        self.terminating_dict= {} # timers with s_2 = e
        
        self.free_ramps = np.arange(0,n_timers)
    
    def setScore(self, ramp_index, score):
        self.scores[ramp_index]= float (score)
    
    def getScore(self, ramp_index):
        return self.scores[ramp_index]
    
    def learningRate(self, ramp_index):
        return self.learning_rates[ramp_index]
    
    def setLearningRate(self, ramp_index, rate):
        self.learning_rates[ramp_index] = rate
        
    def eventDict(self):
        return self.event_dict
    
    def stimulusDict(self):
       return self.stimulus_dict
   
    def terminatingDict(self):
        return self.terminating_dict
    
    def ramps(self):
        return self.timers
    
    def frozen_ramps(self):
        return self.frozen_ramps
    
    def frozen_ramps(self):
        return self.frozen_ramps
    
    def timerWeight(self, index=0):
        return self.timers[index]
    
    def setTimerWeight(self, weight, index=0):
        self.timer_weight = weight
        self.timers[index] = weight
    
    def timerBlock(self):
        return self.block
    
    def buildWeightMatrixFromWeights(timerModules):
        if not isinstance(timerModules, list):
            raise TypeError('Timer modules should be in a list')
        else:
            module_count = len(timerModules)
            weights = np.kron(np.eye(module_count), np.ones((4,4)))
            idx = (0,1)
            for i in range (0, module_count): 
                # t = np.kron(np.eye(module_count), np.ones((4,4)))
               
                weights[0+(4*i):0+(4*(i+1)), 0+(4*i):0+(4*(i+1))] = timerModules[i] #np.array([[2,2,2,2],[2,2,2,2],[2,2,2,2],[2,2,2,2]])
                print(weights)
                # for j in range (0, len(timerModules)):
                 
    def buildWeightMatrixFromModules(timerModules):
        if not isinstance(timerModules, list):
            raise TypeError('Timer modules should be in a list')
        else:
            module_count = len(timerModules)
            weights = np.kron(np.eye(module_count), np.ones((4,4)))
            idx = (0,1)
            for i in range (0, module_count): 
                # t = np.kron(np.eye(module_count), np.ones((4,4)))
               
                weights[0+(4*i):0+(4*(i+1)), 0+(4*i):0+(4*(i+1))] = timerModules[i].timerBlock() #np.array([[2,2,2,2],[2,2,2,2],[2,2,2,2],[2,2,2,2]])
                # we only add connecting weight between modules if there are more than 1
                # and we only do it n-1 times
                if (module_count - i > 1):
                    weights[4+(4*i), 2+(4*i)] = 1
        return weights
                # for j in range (0, len(timerModules)):
    
    def updateV(v):
        # Expands v for a single additional module
        new_v = np.vstack((v,[[0], [0], [0], [0]]))
        return new_v
    
    def updateVHist(v_hist):
        # Expands v_hist for a single additional module
        history = v_hist[0].size
        prior_activity = np.array([np.zeros(history)])
        
        return np.concatenate((v_hist, prior_activity), axis=0)
    
    def updateL(l, lmbd):
        # Expands l for a single additional module
        add_l = np.full((4,1), lmbd)
        return np.concatenate((l, np.array(add_l)))
    
    def updateBias(b):
        add_b = b[:4]
        return np.concatenate((b, add_b))
     
    def generate_sample_from_parameters(num_samples = 1, normal_locs=[] , normal_scales=[],
                                        exponential_scales=[]):
        """
        To be used with predefined distributions in the world. Pass in the distributions'
        scales and centers, and get back N samples. Uses all matching samples and disregards
        inequal argument length, 0 indexed. 
        """
        
        # Calculate number of total distributions
        num_normal = len(normal_locs)
        num_exp = len(exponential_scales)
        num_dists = num_normal + num_exp
        
        event_types = np.arange(num_dists)
        b = np.arange(num_dists)
        perm = permutations(np.concatenate((event_types, event_types), axis=None), 2)
        event_pairs=list(set(list(perm)))
        # print("event pairs: ", event_pairs)

        
        # Generate equal weights based on below algorithm
        # Roll a dice
        # Find which distribution to pull from
        # Pull the sample 
        
        
        return 1
    
    def getSamples(num_samples = 1, num_normal = 2, num_exp = 0, ret_params = False, standard_interval = -1, scale_beg=20, scale_end=50):
        """
        A function that generates random times from a probability 
        distribution that is the weighted sum of exponentials and Gaussians. Returns the parameters 
        of the distribution in a seperate array if ret_params is True
        """
        num_event_types = num_normal + num_exp
        
        P = np.random.rand(num_event_types,num_event_types)
        #print("P: ", P)
        #print("P sum: ", P.sum(axis=1, keepdims=True))
        P = P/P.sum(axis=1, keepdims=True)
        # D = np.random.randint(2,size=(num_event_types,num_event_types))
        T = np.random.randint(20,50, size=(num_event_types,num_event_types))
        S = np.full((num_event_types, num_event_types), 20)
        D=np.ones((num_event_types,num_event_types))
        NUM_DISTS = num_event_types * num_event_types
        DIST_INDICES = np.arange(NUM_DISTS)
        DIST_INDICES.shape=(num_event_types,num_event_types)
        
        
        state = np.random.randint(num_event_types)
        samples = []
        #print("P: ", P)
        for i in range(num_samples):
            
            # if P[state,:].sum() > 1:
            #     P[state,:] = P[state,:] - (P[state,:].sum()/num_event_types)
            next_state = np.random.multinomial(1,P[state,:])
            next_state = np.where(next_state==1)[0][0]
            # np.random.normal(locs[dist_index], scales[dist_index], 1)[0]
            time = (np.random.normal(T[state,next_state], 10,1) * D[state,next_state]) #+ (np.random.exponential(T[state,next_state]) * 1-D[state,next_state])
            #print(D[state,next_state])
            samples.append((time[0],DIST_INDICES[state,next_state],state))  
            state = next_state
        return samples
        
    # Get all permutations of interval pairs
        a = np.arange(num_dists)
        b = np.arange(num_dists)
        perm = permutations(np.concatenate((a, b), axis=None), 2)
        event_pairs=list(set(list(perm)))
        # print("event pairs: ", event_pairs)
        # TODO: Maybe put this into the matrix notation they talked about
        # A hash that gives the indices of event pairings that start with event X
        keys=np.arange(num_dists)
        valid_event_pairs = { key : [] for key in keys }
        for i in range(0, len(event_pairs)):
            pair = event_pairs[i]
            starting_event_type = pair[0]
        
           # print("adding index ", i, " to hash entry ", starting_event_type)
            valid_event_pairs[starting_event_type].append(i)
           # print("event types ", valid_event_pairs)
           # print("\n")
        
        # print("valid event pairs: ", valid_event_pairs)
        num_event_types = num_dists * num_dists
        
        # To get N random weights that sum to 1, add N-1 random numbers to an array
        weights_probs = np.random.rand(num_event_types - 1) 
        # Add 0 and 1 to that array
        weights_probs = np.append(weights_probs, 0)
        weights_probs = np.append(weights_probs, 1)
        # Sort them
        weights_probs = np.sort(weights_probs)
        weights = np.zeros(num_event_types)
        # After establishing the weight array, iterate through the probabilities 
        # and declare the Nth weight to be the difference between the entries at the N+1 and N-1
        # indices of the probability array
        for i in range (0, weights.size):
            weights[i]=(weights_probs[i + 1] - weights_probs[i])
        
        weights = np.sort(weights)
        # Declare distribution types (1 is exp, 0 is normal)
        if num_normal == 0:
            dist_types = np.ones(num_event_types)
        elif num_exp == 0:
            dist_types = np.zeros(num_event_types)
        else:
            dist_types = np.concatenate((np.ones(num_exp), np.zeros(num_normal)), axis=None)
            dist_types = dist_types.concatenate(dist_types, np.zeros(num_event_types - num_normal - num_exp), axis=None)
        
        # print("dist types: ", dist_types)
        # Declare means and std deviations 
        locs = []
        scales = []
            
        # Establish our distributions
        for i in range (0, num_event_types):
            locs.append(np.random.randint(50,80))
            scales.append(math.sqrt(np.random.randint(scale_beg, scale_end)))
        
        # print("")
        # print("locs: ", locs)
        # print("scales: ", scales)
        # print("")
        samples = [] 
        # First event occurs
        dice_roll = np.random.rand(1)
        # find its event type
        for dist_index in range (0, num_event_types):
            if (dice_roll < weights_probs[dist_index + 1]):
                print("di: ", dist_index)
                    
                if dist_types[dist_index] == 1:
                    sample = np.random.exponential(scales[dist_index], 1)[0]
                    samples.append([sample, dist_index])
                
                else:
                    sample = np.random.normal(locs[dist_index], scales[dist_index], 1)[0]
                    if standard_interval > 0:
                        sample = np.random.normal(standard_interval,1, 1)[0]
                    samples.append([sample, dist_index, event_pairs[dist_index][0]])
                break
                        
        # pull next events (A, B, and C) and use which ever chooses first
        next_event_types = valid_event_pairs[event_pairs[dist_index][0]]
        
        def sample_from_event_type(typ):
            #print("typ: ", typ)
            if dist_types[typ] == 1:
                sample = np.random.exponential(scales[dist_index], 1)[0]
            else:
                sample = np.random.normal(locs[dist_index], scales[dist_index], 1)[0]
             
            return [sample, typ]
        
        # print("next event types: ", next_event_types)
        
        next_possible_events = list(map(sample_from_event_type, next_event_types))
        next_possible_events.sort(key=lambda y: y[0])
        # print("next events: ", next_possible_events)
       
        next_event=next_possible_events[0]
        # print("next event: ", next_event)
        # print("event type: ", event_pairs[next_event[1]][1])
        samples.append([next_event[0], next_event[1], event_pairs[next_event[1]][1]])
        
        # print(samples)
        # then repeat
        # print("=====")
        # print("=====")
        # note: doesnt deal with exponential events
        # then repeat     
        for i in range(1, num_samples-1):
            prev_sample = samples[i]
            # print("prev_sample: ", prev_sample)
            next_event_types = valid_event_pairs[prev_sample[2]]
            # print("next_possible_events", next_event_types)
        
            next_possible_events = list(map(sample_from_event_type, next_event_types))
            
            next_possible_events.sort(key=lambda y: y[0])
            # print("next_possible_events: ", next_possible_events)
            next_event=next_possible_events[0]
            samples.append([next_event[0], next_event[1], event_pairs[next_event[1]][1]])

            # print("roll num: ", i)
        
        # if we pull an exponential event, pull the next normal event to have the model progress
        
        
        # Roll a dice N times
         #np.zeros(num_samples)
                # Roll our dice N times
        # I hate that this is O(N * D)
        
        # first = True
        # for i in range(0, num_samples):
        #     dice_roll = np.random.rand(1)
            
        #     # Find which range it belongs in
        #     for dist_index in range (0, num_event_types):
        #         if (dice_roll < weights_probs[dist_index + 1]):
        #            # if not first && samples[i - 1][1] == dist_index:
        #             # The roll falls into this weight, draw our sample
        #             if dist_types[dist_index] == 1:
        #                 sample = np.random.exponential(scales[dist_index], 1)[0]
        #                 samples.append([sample, dist_index])
                    
        #             else:
        #                 sample = np.random.normal(locs[dist_index], scales[dist_index], 1)[0]
        #                 if standard_interval > 0:
        #                     sample = np.random.normal(standard_interval,1, 1)[0]
        #                 samples.append([sample,dist_index])
        #             # if we pull a sample that doesnt begin where the prior left off, we re-roll
        #             # this is so we yield a chain like A->B->A->C->C composed of 
        #             # (A,B), (B,A), (A,C), (C,C) events
                    
        #         # else
        #         #     while samples[i - 1] != dist_index:
        #         #         dice_roll = np.random.rand(1)
                            
        #             break
                
        return np.asarray(samples)
# Useful visualization, just comment it out 
# plt.hist(TimerModule.getSamples(num_samples=1000, num_normal=0, num_exp = 1, num_dists = 1), bins=40, color='black')
# print(TimerModule.getSamples(num_samples=100, num_normal=3, num_exp = 0, num_dists = 3))
    # [relative_time, event_type, stimulus_type]
    def getEvents(num_samples=1, num_normal = 2, num_exp = 0, ret_params = False, standard_interval = -1, scale_beg=20, scale_end=50):
        num_event_types = num_normal + num_exp
        
        P = np.random.rand(num_event_types,num_event_types)
        P = P/P.sum(axis=1, keepdims=True)
        # D = np.random.randint(2,size=(num_event_types,num_event_types))
        T = np.random.randint(20,50, size=(num_event_types,num_event_types))
        S = np.full((num_event_types, num_event_types), 20)
        D=np.ones((num_event_types,num_event_types))
        NUM_DISTS = num_event_types * num_event_types
        DIST_INDICES = np.arange(NUM_DISTS)
        DIST_INDICES.shape=(num_event_types,num_event_types)
        
        
        state = np.random.randint(num_event_types)
        samples = np.empty([num_samples,3])
        
        for i in range(num_samples):
            next_state = np.random.multinomial(1,P[state,:])
            next_state = np.where(next_state==1)[0][0]
            # np.random.normal(locs[dist_index], scales[dist_index], 1)[0]
            time = (np.random.normal(T[state,next_state], 10,1) * D[state,next_state]) #+ (np.random.exponential(T[state,next_state]) * 1-D[state,next_state])
            
            samples[i] = [time[0],DIST_INDICES[state,next_state],state]
            state = next_state
        
        return samples
'''
if something unusual happens, i release some timers 
if you repeat the stimulus, you can look it up in memory to see when it last happened
once thats implemented, start buildig out some heuristics or biases
sum of squares of error of timers. If they exceed some level you make 
some decisions about garbage collection
each ramp has a weight that gets updated/garbage collected depending on its 
error  
'''
                   