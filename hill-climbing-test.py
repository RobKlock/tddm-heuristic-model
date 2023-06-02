#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 28 15:32:25 2023
Hill climbing test
@author: root
"""
import numpy as np
import matplotlib.pyplot as plt
import random

def hillClimbingTest(num_climbers, num_steps, randomness, step_size):
    # Generate Reward Hills
    # Generate a range of x values
    x = np.linspace(-20, 20, int(100/step_size))
    
    # Calculate the y values
    reward = (3 * np.sin(x)) + x
    
    # Plot the function
    # plt.figure(figsize=(10, 6))
    # plt.plot(x, reward)
    # Start our hill climbers at x = 0
    
        # Define the reward surface function
    def f(x):
        return np.sin(x) + x
    
    # Numerically estimate the derivative of f
    def df(x, h=0.0001):
        return (f(x + h) - f(x)) / h
    
    # Parameters for the gradient ascent
    x = 5 # starting point
    learning_rate = 0.01 # learning rate 

    x_space = np.linspace(-20, 20, int(num_steps/step_size))
   
    # Calculate the y values
    reward = (3 * np.sin(x_space)) + x_space
    # Plot the function
    plt.figure(figsize=(10, 6))
    plt.plot(x_space, reward)
   
    for k in range(num_climbers):
        xs=[]
        x = 5
        for i in range(num_steps):
            # Perform gradient ascent step
            x += learning_rate * df(x)
            
            # Randomly add a bit of noise to the position
            if random.random() < randomness:
                x += random.uniform(-1, 1)
            xs.append(x)
            # Print progress every 100 steps
            if i % 100 == 0:
                print(f"Step {i}: x={x}, f(x)={f(x)}")
        plt.plot(xs, np.linspace(np.min(reward),np.max(reward),num_steps), label='random walk', alpha=0.3)
   
            #     if old_reward==cur_reward:
        #         print('equal')
        #         # take a random step
        #         roll = np.random.rand()
        #         if roll >= .5:
        #             cur_pos = cur_pos+np.random.randint(5)
        #             climber_search[step] = cur_pos
        #             climber_reward[step] = reward[cur_pos]
        #         else:
        #             cur_pos = cur_pos-np.random.randint(5)
        #             climber_search[step] = cur_pos
        #             climber_reward[step] = reward[cur_pos]
                
            
            
        #     # we made the wrong step, go back
        #     elif old_reward>cur_reward:
                
        #         if old_pos > cur_pos:
        #             cur_pos=cur_pos+np.random.randint(3)
        #             climber_search[step] = cur_pos
        #             climber_reward[step] = reward[cur_pos]
                    
        #         elif old_pos == cur_pos:
        #             print('e')
        #             if np.random.rand() >= .5:
        #                 cur_pos = cur_pos+np.random.randint(3)
        #                 climber_search[step] = cur_pos
        #                 climber_reward[step] = reward[cur_pos]
        #             else:
        #                 cur_pos = cur_pos-np.random.randint(3)
        #                 climber_search[step] = cur_pos
        #                 climber_reward[step] = reward[cur_pos]
        #         else:
        #             cur_pos=cur_pos-np.random.randint(3)
        #             climber_search[step] = cur_pos
        #             climber_reward[step] = reward[cur_pos]
                
            
        #     else:
        #         print('l')
        #         if old_pos>cur_pos:
        #             print("o")
        #             cur_pos=cur_pos+np.random.randint(3)
        #             climber_search[step]=cur_pos
        #             climber_reward[step] = reward[cur_pos]
               
        #         elif old_pos == cur_pos:
        #             print('e')
        #             if np.random.rand() >= .5:
        #                 cur_pos = cur_pos+np.random.randint(3)
        #                 climber_search[step] = cur_pos
        #                 climber_reward[step] = reward[cur_pos]
        #             else:
        #                 cur_pos = cur_pos-np.random.randint(3)
        #                 climber_search[step] = cur_pos
        #                 climber_reward[step] = reward[cur_pos]
        #         else:
        #             print("h")
        #             cur_pos=cur_pos-np.random.randint(3)
        #             climber_search[step]=cur_pos
        #             climber_reward[step] = reward[cur_pos]
                
            
        # plt.plot(climber_search, np.linspace(np.min(reward),np.max(reward),num_steps), label='random walk', alpha=0.3)
       #  plt.show()
                    
        
    
   # 
    
def test():
        # Define the reward surface function
    def f(x):
        return np.sin(x) + x
    
    # Numerically estimate the derivative of f
    def df(x, h=0.0001):
        return (f(x + h) - f(x)) / h
    
    # Parameters for the gradient ascent
    x = 5
    
    learning_rate = 0.01
    epochs = 1000
    randomness = 0.1
    
    x_space = np.linspace(-20, 20, 1000)
    
    # Calculate the y values
    reward = (3 * np.sin(x_space)) + x_space
    # Plot the function
    plt.figure(figsize=(10, 6))
    plt.plot(x_space, reward)
    for k in range(10):
        xs=[]
        x = 5
        for i in range(epochs):
            # Perform gradient ascent step
            x += learning_rate * df(x)
            
            # Randomly add a bit of noise to the position
            if random.random() < randomness:
                x += random.uniform(-1, 1)
            xs.append(x)
            # Print progress every 100 steps
            if i % 100 == 0:
                print(f"Step {i}: x={x}, f(x)={f(x)}")
        plt.plot(xs, np.linspace(np.min(reward),np.max(reward),epochs), label='random walk', alpha=0.3)

hillClimbingTest(4,1000,.04,.02)
# test()