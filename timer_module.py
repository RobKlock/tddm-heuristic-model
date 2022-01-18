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
Scikit-Learn has useful Gaussian Mixture for probabilities 
Be careful because they probably have two models: one for fitting
and another for generating samples.

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
        self.terminating_events=np.empty(n_timers)
        self.terminating_events.fill(-1)
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
 
    def generate_sample_from_parameters(n=1, normal_locs=None , normal_scales=None,
                                        exponential_scales=None):
        """
        To be used with predefined distributions in the world. Pass in the distributions'
        scales and centers, and get back N samples. Uses all matching samples and disregards
        inequal argument length, 0 indexed. 
        """
        
        # Calculate number of total distributions
        # Generate equal weights based on below algorithm
        # Roll a dice
        # Find which distribution to pull from
        # Pull the sample 
        
        
        return 1
    
    def getSamples(num_samples = 1, num_normal = 2, num_exp = 0, ret_params = False, standard_interval = -1):
        """
        A function that generates random times from a probability 
        distribution that is the weighted sum of exponentials and Gaussians. Returns the parameters 
        of the distribution in a seperate array if ret_params is True
        """
        num_dists = num_normal + num_exp
        
        # Get all permutations of intervals 
        a = np.arange(num_dists)
        b = np.arange(num_dists)
        perm = permutations(np.concatenate((a, b), axis=None), 2)
        a=set(list(perm))
        #print(a)
        print(len(a))
        num_event_types = len(a)
        
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
        
        # Declare means and std deviations 
        locs = []
        scales = []
            
        # Establish our distributions
        for i in range (0, num_event_types):
            locs.append(np.random.randint(50,100))
            scales.append(math.sqrt(np.random.randint(20, 100)))
       
        
       
        # Roll a dice N times
        samples = [] #np.zeros(num_samples)
                # Roll our dice N times
        # I hate that this is O(N * D)
        for i in range(0, num_samples):
            dice_roll = np.random.rand(1)

            # Find which range it belongs in
            for dist_index in range (0, num_event_types):
                if (dice_roll < weights_probs[dist_index + 1]):
                    # The roll falls into this weight, draw our sample
                    if dist_types[dist_index] == 1:
                        sample = np.random.exponential(scales[dist_index], 1)[0]
                        samples.append([sample, dist_index])
                    
                    else:
                        sample = np.random.normal(locs[dist_index], scales[dist_index], 1)[0]
                        if standard_interval > 0:
                            sample = np.random.normal(standard_interval,1, 1)[0]
                        samples.append([sample,dist_index])
                    break
                
        return np.asarray(samples)
# Useful visualization, just comment it out 
# plt.hist(TimerModule.getSamples(num_samples=1000, num_normal=0, num_exp = 1, num_dists = 1), bins=40, color='black')
# print(TimerModule.getSamples(num_samples=100, num_normal=3, num_exp = 0, num_dists = 3))
      
            
'''
if something unusual happens, i release some timers 
if you repeat the stimulus, you can look it up in memory to see when it last happened
once thats implemented, start buildig out some heuristics or biases
sum of squares of error of timers. If they exceed some level you make 
some decisions about garbage collection

each ramp has a weight that gets updated/garbage collected depending on its 
error  
'''
                