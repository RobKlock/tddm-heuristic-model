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
        self.terminating_events = np.full(n_timers, -1) # TODO: change this to np.nan
        
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
    
    def getSamples(num_samples = 1, num_normal = 2, num_exp = 0,seed=100, ret_params = False, standard_interval = -1, scale_beg=20, scale_end=50):
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
        
    
# plt.hist(TimerModule.getSamples(num_samples=1000, num_normal=0, num_exp = 1, num_dists = 1), bins=40, color='black')
# print(TimerModule.getSamples(num_samples=100, num_normal=3, num_exp = 0, num_dists = 3))
    # [relative_time, event_type, stimulus_type]
    def getEvents(num_samples=1, num_normal = 2, num_exp = 0, repeat = 2, deviation=1, ret_params = False, standard_interval = -1, scale_beg=20, scale_end=50):
        num_event_types = num_normal + num_exp
        
        P = np.random.rand(num_event_types,num_event_types)
        distribution_weights = P/P.sum(axis=1, keepdims=True) # Probabilities of each event type
        
        # D = np.random.randint(2,size=(num_event_types,num_event_types))
        centers = np.random.randint(scale_beg,scale_end, size=(num_event_types,num_event_types)) # locs, or centers of each distribution
        deviations = np.full((num_event_types,num_event_types), deviation) # deviations of each distribution
        
        NUM_DISTS = num_event_types * num_event_types # number of distributions
        DIST_INDICES = np.arange(NUM_DISTS) 
        DIST_INDICES.shape=(num_event_types,num_event_types) # convert indices into square matrix
        break_type = num_samples+1
        
        state = np.random.randint(num_event_types) # current state
        print(f'state: {state}')
        samples = np.empty([(num_samples * repeat) + (repeat-1),3]) # initialize array of samples 
        next_state = np.random.multinomial(1,distribution_weights[state,:])
        next_state = np.where(next_state==1)[0][0]

        samples[0] = [0,DIST_INDICES[state,next_state],state]
        first_state = state
        for i in range(num_samples):
            print(f'state: {state}')
            next_state = np.random.multinomial(1,distribution_weights[state,:]) # get next state according to distribution weights
            next_state = np.where(next_state==1)[0][0] # index location of distribution
            # np.random.normal(locs[dist_index], scales[dist_index], 1)[0]
            time = (np.random.normal(centers[state,next_state], 1,1) * deviations[state,next_state]) # sample from normal dist to get time
            #+ (np.random.exponential(T[state,next_state]) * 1-D[state,next_state])
            for j in range(0,repeat):
                samples[i+(num_samples*j) + 1] = [time[0],DIST_INDICES[state,next_state],state] # add sample in form [relative time, event type, stimulus type]
            
                # samples[i+(j-1)] = [time[0],DIST_INDICES[state,next_state],state] 
            state = next_state # proceed to next state
        
        """   
        # Loop through samples and repeat the sequence by re-sampling fro each distribution type
        for i in range(repeat-1):
            state = first_state
            for j in range(num_samples):
                next_state = int(samples[j+1][2])
                
                time = (np.random.normal(centers[state,next_state], 10,1) * deviations[state,next_state]) 
                samples[j+i] = [time[0],DIST_INDICES[state,next_state],state]
                state = next_state 
        """
       # for rep in range (0, num_repeat):
        samples[0][1] = samples[repeat-2][1]
        samples[0][2] = samples[repeat-2][2]
       
        return samples[:-1]
'''
if something unusual happens, i release some timers 
if you repeat the stimulus, you can look it up in memory to see when it last happened
once thats implemented, start buildig out some heuristics or biases
sum of squares of error of timers. If they exceed some level you make 
some decisions about garbage collection
each ramp has a weight that gets updated/garbage collected depending on its 
error  
'''
                   