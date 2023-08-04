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
import math

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
            
            # use position derivative and reward derivative
            if i<2:
                df_p = 0
            else:
                df_p = (xs[-1] - xs[-2]) / step_size 
            
            x += learning_rate * df(x,step_size) * df_p
            
            x += np.random.normal(0, step_size) # 1/f(x) *  1/df(x,step_size))
            
            # Randomly add a bit of noise to the position
            # if random.random() < randomness:
            
            xs.append(x)
            
        
        plt.plot(xs, np.linspace(np.min(reward),np.max(reward),num_steps), label='random walk', alpha=0.3)
 
def simulated_annealing(initial_solution, initial_temperature, cooling_rate, max_iterations, step_size, num_climbers):
    '''
    initial solution: random, or initated to something consistent
    initial temperature: 1-100
    cooling rate: determines how fast temperature decreases. 0.8-0.99
    max_iterations: max iterations
    step_size: dt
    '''
    
    # current_solution = initial_solution
    # init_cs = current_solution
    # best_solution = current_solution
    # init_bs = best_solution
    # current_temperature = initial_temperature
    # init_ct = current_temperature
    
    x = np.linspace(-20, 20, int(100/step_size))
    reward = (3 * np.sin(x)) + x
    x_space = np.linspace(-20, 20, int(max_iterations/step_size))
    plt.figure(figsize=(10, 6))
    plt.plot(x_space, reward)
    
    current_solution = initial_solution
    best_solution = current_solution
    current_sol_init = current_solution
    
    current_temperature = initial_temperature
    current_temp_init = current_temperature
    for climber in range(num_climbers):
        xs=[]
        
        # perturb parameters
        # initial_temperature -= climber * .05
        current_solution = current_sol_init
        best_solution = current_solution
        current_temperature = current_temp_init
        for iteration in range(max_iterations):
         
            # Generate a new candidate solution by perturbing the current solution
            candidate_solution = perturb_solution(current_solution,step_size)
            
            # Calculate the differences in the objective function (reward) between the current and candidate solutions
            current_reward = calculate_reward(current_solution)
            candidate_reward = calculate_reward(candidate_solution)
            reward_difference = candidate_reward - current_reward
    
            # Decide whether to accept the candidate solution
            if reward_difference > 0:
                current_solution = candidate_solution
            else:
                acceptance_probability = math.exp(reward_difference / current_temperature)
                if random.random() < acceptance_probability:
                    current_solution = candidate_solution
    
            # Update the best solution if necessary
            if calculate_reward(current_solution) > calculate_reward(best_solution):
                best_solution = current_solution
    
            # Decrease the temperature
            current_temperature *= cooling_rate
            
            xs.append(best_solution)
        
        
        plt.plot(xs, np.linspace(np.min(reward),np.max(reward),max_iterations), label='simulated annealing', alpha=0.3)
        
    return best_solution

# Function for perturbing the current solution (customize this according to your problem)
def perturb_solution(solution,step_size):
    # Perform some perturbation to generate a new candidate solution
    # Modify the solution based on your problem's requirements
    return solution + np.random.normal(0, step_size) # 1/f(x) *  1/df(x,step_size))

# Function for calculating the reward (customize this according to your problem)
def calculate_reward(solution):
    # Calculate the reward for the given solution
    # Modify the calculation based on your problem's requirements
    return (3 * np.sin(solution)) + solution


simulated_annealing(4.5, 200, 0.01, 100, .2,4)
# hillClimbingTest(4,1000,.04,.02)
# test()