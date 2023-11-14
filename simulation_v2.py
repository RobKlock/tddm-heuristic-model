
import numpy as np
import matplotlib.pyplot as plt
import math
import numpy.random as random 
from timer_module import TimerModule as TM

timer=TM(4,10)
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
                              [25,1,1], [50,0,0], [25,1,1],
                              [50,0,0], [25,1,1], [50,0,0],
                              [25,1,1], [50,0,0], [25,1,1],
                              [50,0,0], [25,1,1], [50,0,0],
                              [25,1,1], [50,0,0], [25,1,1],
                              [50,0,0], [25,1,1], [50,0,0],
                              [25,1,1], [50,0,0], [25,1,1],
                              [50,0,0], [25,1,1], [50,0,0],
                              [25,1,1], [50,0,0], [25,1,1],
                              [50,0,0], [25,1,1], [50,0,0],
                              [25,1,1], [50,0,0], [25,1,1],
                              [50,0,0], [25,1,1], [50,0,0],
                              [25,1,1], [50,0,0], [25,1,1],
                              [50,0,0], [25,1,1], [50,0,0],
                              [25,1,1], [50,0,0], [25,1,1],
                              [50,0,0], [25,1,1], [50,0,0],
                              [25,1,1], [50,0,0], [25,1,1],
                              [50,0,0], [25,1,1], [50,0,0],
                              [25,1,1], [50,0,0], [25,1,1],
                                [50,0,0], [25,1,1], [50,0,0],
                              [25,1,1], [50,0,0], [25,1,1],
                              [50,0,0], [25,1,1], [50,0,0],
                              [25,1,1], [50,0,0], [25,1,1],
                              [50,0,0], [25,1,1], [50,0,0],
                              [25,1,1], [50,0,0], [25,1,1],
                              [50,0,0], [25,1,1], [50,0,0],
                              [25,1,1], [50,0,0], [25,1,1],
                              [50,0,0], [25,1,1], [50,0,0],
                              [25,1,1], [50,0,0], [25,1,1],
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
def generate_hit_time(weight, threshold, noise, dt, plot=False):   
    # Alternative method for hitting time
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
       plt.hlines(threshold,0,hit_time*dt)
       plt.plot(x[0:hit_time], cum_act_arr[0:hit_time], color="grey")
       # plt.xlim([0,hit_time + (hit_time//2)])
       # plt.ylim([0, threshold + (threshold/2)])
       
    if hit_time > 0:    
        return [hit_time, cum_act_arr[:hit_time]]

def activationAtIntervalEnd(timer, ramp_index, interval_length, c):
    # Simulate DDM process for activation amount
    # Change act to activation
    act = timer.timers[ramp_index] * interval_length
    for i in range (0, len(act)):
        act[i] = act[i] + c * np.sqrt(act[i]) * np.random.normal(0, 1) * math.sqrt(interval_length)
    
    return act

def generate_hit_time(weight, threshold, noise, dt, plot=False):   
    # Alternative method for hitting time
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
       plt.hlines(threshold,0,hit_time*dt)
       plt.plot(x[0:hit_time], cum_act_arr[0:hit_time], color="grey")
       # plt.xlim([0,hit_time + (hit_time//2)])
       # plt.ylim([0, threshold + (threshold/2)])
       
    if hit_time > 0:    
        return [hit_time, cum_act_arr[:hit_time]]
    
def activationAtIntervalEndSim(timer, threshold, ramp_index, interval_length, c, dt):
    # Simulate DDM process for activation amount
    T = int(interval_length/dt)
    num_ramps = len(ramp_index)
    noise_arr = np.random.normal(0,1,(num_ramps,T)) * c * np.sqrt(dt)
    weights = np.array([timer.timers[ramp_index]])
    drift_arr = np.ones((num_ramps,T)) * weights.T * dt
    
    act_arr = drift_arr + noise_arr
    
    cum_act_arr = np.cumsum(act_arr, axis=1)
    hit_times = [np.argmax(row>=threshold) for row in cum_act_arr]
    end_vals = [row[-1] for row in cum_act_arr]
#     for i in range(len(ramp_index)):
#         plt.plot(np.linspace(0,interval_length,cum_act_arr[i].shape[0]),cum_act_arr[i])
#         plt.hlines(threshold,0, T)
#         plt.show()
    return (hit_times, cum_act_arr, end_vals)

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
    for i in range (1, len(act)):
        act[i] = act[i] + c * np.sqrt(act[i]) * np.random.normal(0, 1) * math.sqrt(interval_length)
    
    return act

def activationAtIntervalEndHierarchical(timer, ramp_index, interval_length, next_stimulus_type, c):
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

def update_and_reassign_ramps(timer, timer_values, timer_indices, next_stimulus_type, stimulus_type, external_idx, allocation_prob, NUM_RAMPS, RAMPS_PER_EVENT, sequence_code = '', v0=1.0, z = 1, bias = 1, plot = False):
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
        
        if timer.initiating_events[idx] == stimulus_type and timer.terminating_events[idx] == next_stimulus_type:
            if flip>=0.2:
                old_weight = timer.timerWeight(idx)
                # print(f"updating timer {idx}")
                if value > 1:
                    ''' Early Update Rule '''
                    timer_weight = earlyUpdateRule(value, timer.timerWeight(idx), timer.learningRate(idx))
                        
                else:
                    ''' Late Update Rule '''
                    timer_weight = lateUpdateRule(value, timer.timerWeight(idx), timer.learningRate(idx))
                if (abs(timer_weight - old_weight)>0.1):
                    timer.free_ramps = np.append(timer.free_ramps, idx)
                    timer.initiating_events[idx] = -1
                    timer.terminating_events[idx] = -1
                    continue
                    
                timer.setTimerWeight(timer_weight, idx) 
                
        # If a timer is unassigned
        if len(np.where(timer.terminating_events[np.where(timer.initiating_events == stimulus_type)] == next_stimulus_type)[0])>RAMPS_PER_EVENT:
            continue
        
        
        if timer.terminating_events[idx] == -1:
            if flip <= allocation_prob: # Update this to be a var, not a magic number
                # if the timer has the appropriate terminating event, update the weight
                if value > 1:
                    ''' Early Update Rule '''
                    timer_weight = earlyUpdateRule(value, timer.timerWeight(idx), timer.learningRate(idx))
                   
                    
                else:
                    ''' Late Update Rule '''
                    timer_weight = lateUpdateRule(value, timer.timerWeight(idx), timer.learningRate(idx))
                        
                if idx in timer.free_ramps:
                    # print(f"updating timer {idx} to have start {stimulus_type} and stop {next_stimulus_type}")
                    timer.setTimerWeight(timer_weight, idx)
                    timer.free_ramps = np.delete(timer.free_ramps, np.where(timer.free_ramps == idx))
                    timer.initiating_events[idx] = stimulus_type
                    timer.terminating_events[idx] = next_stimulus_type
                    
            continue
        
        
                
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
    responses = np.random.exponential(1, interval_length)
    return responses

def sigmoid(x, gain=20, ceiling=1):
    return ceiling / (1 + np.exp(-gain * x))

def respond(timer_value, event_time, next_event, START_THRESHOLD, STOP_THRESHOLD, dt, num_samples, K, idx, paths, initiating_active_indices):
    # Given all ramp values, respond when K are between start and stop range
    
    # Find start threshold times for each ramp
    start_threshold_times = start_threshold_time(timer_value, next_event-event_time, START_THRESHOLD)
    stt = np.argwhere(paths >= START_THRESHOLD)
    stt = stt * dt
    stt = stt[:,1]
    stt = stt[np.where(stt >= start_threshold_times[0])]
    
    
    start_threshold_times += event_time
    stt += event_time 
    stt.sort()
    stt = np.vstack((stt, np.ones(stt.shape[0]))).T
    
    start_threshold_times.sort()
    start_threshold_times = np.vstack((start_threshold_times, np.ones(len(start_threshold_times)))).T
    
    # Find stop threshold times for each ramp
    stop_threshold_times = stop_threshold_time(timer_value, next_event-event_time, STOP_THRESHOLD)
    
    stop_tt = np.argwhere(paths >= STOP_THRESHOLD)
    stop_tt = stop_tt * dt
    stop_tt = stop_tt[:,1]
    stop_tt = stop_tt[np.where(stop_tt >= stop_threshold_times[0])]
    stop_tt += event_time 
    stop_tt.sort()
    stop_tt = np.vstack((stop_tt, np.ones(stop_tt.shape[0]))).T
    
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
    # print(f"rp {response_periods}")
    r = list(generate_responses(next_event-event_time, dt, num_samples))
    r.insert(0, event_time)
    r=list(np.cumsum(r))
    
    for response_period in response_periods:
        responses.extend([i for i in r if (i>=response_period[0] and i<=response_period[1] and i<next_event and i>event_time + 5)])
        
    
    # responses and ax1.text(responses[0],1.2,str(idx))
    return responses

def respond2(paths, event_time, next_event, START_THRESHOLD, STOP_THRESHOLD, dt, K, idx, initiating_active_indices):
    # Given all ramp values, respond when K are between start and stop range
    # Simulate the DDM process to get cumulative activation array
    
    # paths = paths[:initiating_active_indices[0].shape[0], :]
    
    # Perform threshold analysis to find when activation is between start and stop thresholds
    responding_periods = threshold_analysis(paths, START_THRESHOLD, STOP_THRESHOLD, K)
    # print(f"responding periods: {responding_periods}")
    # Generate responses within the periods where K paths are between thresholds
    responses = []
    for i, period in enumerate(responding_periods):
        # Generate responses for this period
        responses_ = generate_responses2(event_time + period[0]*dt,next_event + period[1]*dt,dt)
        responses.extend(responses_)
    
    return responses

# Additional function to generate responses
def generate_responses2(start_time, end_time, dt):
    # Generate responses between start_time and end_time
    responses = []
    # while start_time < end_time:
    num_samples = int((end_time-start_time) / dt)
    response_times = np.random.exponential(1, num_samples) + start_time
    
    responses = response_times[np.where(response_times < end_time)]
#         if response_time < end_time:
#             responses.append(response_time)
#         start_time += dt
    return responses

def timer_response_analysis(between_thresholds, K):
    # Analyze the timer responses
    sum_between_thresholds = np.sum(between_thresholds, axis=0)
    # print(f"Sum Between Thresholds: {sum_between_thresholds}")
    timer_responses = np.sum(between_thresholds, axis=0) >= K
    return timer_responses

def threshold_analysis(cum_act_arr, start_threshold, stop_threshold, k):
    # Analyze the paths for threshold crossings
    # between_thresholds = (cum_act_arr >= start_threshold) & (cum_act_arr =< stop_threshold)
    out_of_range = True
    responding_periods = []
    starting_period, stopping_period = 0, 0
    
    for col_idx in range(15,cum_act_arr.shape[1]):
        col = cum_act_arr[:,col_idx]
        
        count_between_thresholds = np.where((col>=start_threshold) & (col<stop_threshold))[0].shape[0] 
   
        if count_between_thresholds >= k and out_of_range:
            starting_period = col_idx
            out_of_range = False
        if count_between_thresholds < k and not out_of_range:
            stopping_period = col_idx
            out_of_range=True
            responding_periods.append((starting_period, stopping_period))
    
    return responding_periods

def flag_slope_variation(old_slope, new_slope,dev=.1):
    return abs(1-(old_slope/(new_slope+10**-10))) > dev

def main(parameters, plot=True):
    dt = parameters["dt"]
    N_EVENT_TYPES= parameters["N_EV_TYPES"] # Number of event types (think, stimulus A, stimulus B, ...)
    Y_LIM=2 # Vertical plotting limit
    NOISE=parameters["NOISE"] # Internal noise - timer activation
    LEARNING_RATE=parameters["lr"] # Default learning rate for timers
    ALLOCATION_PROB = parameters["allocation_prob"]
    STANDARD_INTERVAL=20 # Standard interval duration 
    K = parameters["k"] # Amount of timers that must be active to respond
    
    RR_est_tau = 200
    
    # hill climbing params
    START_THRESHOLD=parameters["start_thresh"]# Response start threshold
    STOP_THRESHOLD=parameters["stop_thresh"] # Response stop threshold
    c = 0.1 # noise for iteration, separate from internal neural noise
    baseline_noise = parameters['baseline_noise']
    noise_scale = parameters['noise_scale']
    scale = 100
    reward_vals = [0]
    R_hat = R_delay = 1.0
    R_hat_vals = [R_hat]
    R_delay_vals = [R_delay]
    
    start_hat = start_delay = START_THRESHOLD
    stop_hat = stop_delay = STOP_THRESHOLD
    
    start_hat_vals = []
    start_delay_vals = []
    
    stop_hat_vals = []
    stop_delay_vals = []
    
    R_hat_dot_delay = 0
    R_hat_dot_dot_delay = 0
    R_hat_dot_delay_vals = []
    start_hat_dot_vals = []
    stop_hat_dot_vals = []
    R_hat_dot_vals = []
    R_hat_dot_dot_vals = []
    noise_vals = [baseline_noise]
    
    path = [] # threshold vals and the timestep
    tau_iteration = parameters['tau']
    
    start_thresh_vals = []
    stop_thresh_vals = []
    # End iteration params
    PLOT_FREE_TIMERS=False
    ERROR_ANALYSIS_RESPONSES=[]
    BEAT_THE_CLOCK = False
    colors = [[1,0,0], [0,1,0], [0,0,1], [1,1,0], [0,1,1], [1,0,1],[.46,.03,0], [.1,.3,.2], [.2,.7,.2], [.5,.3,.6], [.7,.3,.4]]# list(mcolors.CSS4_COLORS) # Color support for events
    ALPHABET_ARR = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','AA','BB','CC'] # For converting event types into letters 
    colors = ['red','blue','green','orange','purple']
    RESPONSE_THRESHOLD_LEARNING_RATE = .6
    NUM_RAMPS = parameters["num_ramps"]
    RAMPS_PER_EVENT = parameters["ramps_per_event"]
    reward_window_plot = 1600
    reward_window_sim = 1600
    x_lb = np.linspace(-reward_window_plot * dt,0, reward_window_plot)
    exp_weighted_average = np.exp(x_lb * .01)
    
    event_data = []
    cur_RR = 0
    old_RR = 0
    expand = True 
    contract = False
    random_seq = False
    seq_length = 3


    HOUSE_LIGHT_ON =  [*range(0,16,1)]

    event_data = parameters['event_data']

    event_data = np.asarray([[0,0,0], [50,1,1], 
                              [10,0,0],[50,1,1],
                              [10,0,0], [25,2,2],
                              [10,0,0], [25,2,2],
                              [10,0,0], [25,2,2], [25,1,1],
                              [10,0,0], [50,2,2], [50,1,1],
                              [10,0,0], [50,2,2], [50,1,1],
                              [10,0,0], [20,2,2], [15,1,1],
                              [10,0,0], [20,2,2], [15,1,1],
                              [10,0,0], [20,2,2], [15,1,1],

                              [10,1,1], [10,1,1], [10,1,1]])
       
    # TODO: Make start threhsolds an array of values
    penalty=parameters['penalty']

    NUM_EVENTS = len(event_data) 
    btc_reward=np.empty(NUM_EVENTS)

    error_arr = np.zeros(NUM_EVENTS)
    event_data = relative_to_absolute_event_time(event_data, NUM_EVENTS)
    
    # Last event, time axis for plotting        
    T = event_data[-1][0]
    
    # Timer with 100 (or however many you want) ramps, all initialized to be very highly weighted (n=1)
    timer=TM(1,NUM_RAMPS)
    if plot:
        poster_fig1 = plt.figure(figsize=(8,3))
        poster_fig2 = plt.figure(figsize=(8,3))
        poster_fig3 = plt.figure(figsize=(8,3))
        
        
        pax1 = poster_fig1.add_subplot(111)
        pax2 = poster_fig2.add_subplot(111)
        pax3 = poster_fig3.add_subplot(111)
        pax1.set_ylim([0,Y_LIM])
        pax2.set_ylim([0,Y_LIM])
        pax3.set_ylim([0,Y_LIM])
        
        pax1.set_xlim([0,190])
        pax2.set_xlim([185,465])
        pax3.set_xlim([460,600])
        
        simple_learning_fig = plt.figure(figsize=(8,4))
        # simple_learning_fig.suptitle('Simple Learning Sequence', fontsize=16)
        ax1 = simple_learning_fig.add_subplot(411)
        ax2 = simple_learning_fig.add_subplot(412, sharex = ax1)
        ax3 = simple_learning_fig.add_subplot(413, sharex = ax1)
        ax4 = simple_learning_fig.add_subplot(414, sharex = ax1)
    
        ax1.set_ylim([0,Y_LIM])
        ax1.set_xlim([0,T])
    
        ax2.set_ylim([0,1])
        ax2.set_xlim([0,T])
        ax3.set_xlim([0,T])
        
        ax4.set_xlim([0,event_data.shape[0]])
        ax4.set_ylim([0,2])
        
    reward_arr_plot = np.zeros(int(event_data[-1][0] / dt))

    timer_plot_legend_free = {}
    timer_plot_legend_assigned = {}

    # Initialize a reward arr that has a small amount of reward at each time step
    reward_arr = np.zeros(int(event_data[-1][0]/dt))
    if plot:
        reward_x_axis = np.linspace(0,event_data[-1][0]/dt,reward_arr.shape[0])
        
    # For event, add a large amount of reward at the event and a little right before it 
    for index, event in enumerate(event_data[1:]):
        if index in HOUSE_LIGHT_ON or (index) in HOUSE_LIGHT_ON:
            if int(event[0]/dt) < reward_arr.shape[0]:
                reward_arr[int(event[0]/dt)] = 1
                exp_arr = np.exp(-.5 * np.arange(0, 20, dt))[::-1]
                reward_arr[int(event[0]/dt) - exp_arr.shape[0]:int(event[0]/dt)] = exp_arr

    reward_arr[0] = 0 
    reward_arr_x = np.linspace(0,int(event_data[-1][0]), reward_arr_plot.shape[0])
    reward_est_vals = np.zeros(reward_arr_x.shape)
    reward_estimation = [0]    
    
    ''' Simulation Start '''
    # At each event e_i
    for idx, event in enumerate(event_data[:HOUSE_LIGHT_ON[-1]]):
        house_light = idx in HOUSE_LIGHT_ON
        event_time = event[0]
        event_type = int(event[1])
        stimulus_type = int(event[2])
        next_event = event_data[idx+1][0]
        next_stimulus_type=int(event_data[idx+1][2])
        
        path.append((START_THRESHOLD, STOP_THRESHOLD, event_time))
        
        # Plot event times and labels
        if idx < (NUM_EVENTS - 1):
            if plot:
                ax1.text(event[0],2.1,ALPHABET_ARR[stimulus_type])
                ax1.vlines(event_time, 0,Y_LIM, label="v", color=colors[stimulus_type])
                
                if idx < 8:
                    pax1.text(event[0],2.1,ALPHABET_ARR[stimulus_type])
                    pax1.vlines(event_time, 0,Y_LIM, label="v", color=colors[stimulus_type])
                if idx >= 8 and idx <= 16:    
                    pax2.text(event[0],2.1,ALPHABET_ARR[stimulus_type])
                    pax2.vlines(event_time, 0,Y_LIM, label="v", color=colors[stimulus_type])
                if idx > 16 and idx <= 25:
                    pax3.text(event[0],2.1,ALPHABET_ARR[stimulus_type])
                    pax3.vlines(event_time, 0,Y_LIM, label="v", color=colors[stimulus_type])
                
        if house_light:
            # Plot house light bar
            if plot:
                house_light_bar = ax1.plot([event_time, next_event], [1.9, 1.9], 'k-', lw=4)  
                ax1.plot([event_time, next_event], [1.9, 1.9], 'k-', lw=4)  
                
                if idx < 8:
                    pax1.plot([event_time, next_event], [1.9, 1.9], 'k-', lw=4)  
                if idx >= 8 and idx <= 16:    
                    pax2.plot([event_time, next_event], [1.9, 1.9], 'k-', lw=4)  
                if idx > 16 and idx <= 25:
                    pax3.plot([event_time, next_event], [1.9, 1.9], 'k-', lw=4)  
            
            # Look forward to all other intervals before house light turns off and start updating weights
            house_light_idx = idx + 1
            house_light_interval = True
            tau=200
            while house_light_interval:
                # If the next interval is in the house light period
                if house_light_idx in HOUSE_LIGHT_ON: 
                    # Get next event time and stimulus type
                    next_house_light_event_time = event_data[house_light_idx][0]
                    next_house_light_stimulus_type = event_data[house_light_idx][2]
                    
                    # All indices of ramps active by initiating event
                    initiating_active_indices = np.where(timer.initiating_events == stimulus_type)
                    
                    # All initiating and free ramp indices
                    active_ramp_indices = np.append(initiating_active_indices, timer.free_ramps)
                    
                    house_light_timer_value2, paths, end_vals= activationAtIntervalEndSim(timer, START_THRESHOLD, active_ramp_indices, next_house_light_event_time - event_time, NOISE, dt)
                    
                    # house_light_timer_value = activationAtIntervalEnd(timer, active_ramp_indices, next_house_light_event_time - event_time, NOISE)
                    
                    house_light_timer_value = end_vals
                    house_light_hierarchical_value = activationAtIntervalEndHierarchical(timer, initiating_active_indices, next_house_light_stimulus_type, next_house_light_event_time - event_time, NOISE)
                    
                    
                    # house_light_responding_values = activationAtIntervalEnd(timer, initiating_active_indices, next_house_light_event_time - event_time, NOISE)
                    house_light_responding_values = np.array([paths[i][j] for i, j in enumerate(house_light_timer_value2)])
                    
                    if idx >= 0:
                        if idx>0:
                            responses = respond2(paths, event_time, next_event, START_THRESHOLD, STOP_THRESHOLD, dt, K, idx, initiating_active_indices)
                            responses = respond(end_vals, event_time, next_house_light_event_time, START_THRESHOLD, STOP_THRESHOLD, dt, seq_length, K, idx, paths, initiating_active_indices)
                       
                            reward = reward_arr[[int(r/dt) for r in responses]]
                        
                        else:
                            responses = np.array([])
                            reward = np.array([])
                            
                        if plot:
                            ax1.plot(responses, np.ones(len(responses)), 'x', color=colors[next_stimulus_type])  
                            ax1.plot([event_time, next_house_light_event_time], [START_THRESHOLD, START_THRESHOLD], color='green')
                            ax1.plot([event_time, next_house_light_event_time], [STOP_THRESHOLD, STOP_THRESHOLD], color='red') 
                            
                            if idx < 8:
                                pax1.plot(responses, np.ones(len(responses)), 'x', color=colors[next_stimulus_type])  
                                pax1.plot([event_time, next_house_light_event_time], [START_THRESHOLD, START_THRESHOLD], color='green')
                                pax1.plot([event_time, next_house_light_event_time], [STOP_THRESHOLD, STOP_THRESHOLD], color='red') 
                            if idx >= 8 and idx <= 16:    
                                pax2.plot(responses, np.ones(len(responses)), 'x', color=colors[next_stimulus_type])  
                                pax2.plot([event_time, next_house_light_event_time], [START_THRESHOLD, START_THRESHOLD], color='green')
                                pax2.plot([event_time, next_house_light_event_time], [STOP_THRESHOLD, STOP_THRESHOLD], color='red') 
                            if idx > 16 and idx <= 25:
                                pax3.plot(responses, np.ones(len(responses)), 'x', color=colors[next_stimulus_type])  
                                pax3.plot([event_time, next_house_light_event_time], [START_THRESHOLD, START_THRESHOLD], color='green')
                                pax3.plot([event_time, next_house_light_event_time], [STOP_THRESHOLD, STOP_THRESHOLD], color='red') 
                            
                            ax2.plot(responses,reward, marker='x', color = 'g')
                        
                        reward_vals.append(reward)
                        
                        # Iteration start
                        if all(np.isnan(x) for x in reward):
                            R = 0
                        else:
                            R = R_hat
                            
                        # QUESTION: Reward is an array, for now i average its vals, should we do something diff?
                        # switch this to the reward rate estimator 
                        dt2 = 0.0001
                        start_hat_dot = (1/tau_iteration) * (START_THRESHOLD - start_hat)
                        start_hat_delay = (1/tau_iteration) * (start_hat - start_delay)
                        
                        stop_hat_dot = (1/tau_iteration) * (STOP_THRESHOLD - stop_hat)
                        stop_hat_delay = (1/tau_iteration) * (stop_hat - stop_delay)
                        
                        start_hat += (start_hat_dot * dt2)
                        start_delay += (start_hat_delay*dt2)
                        
                        stop_hat += (stop_hat_dot * dt2)
                        stop_delay += (stop_hat_delay*dt2)
                        
                        start_hat_vals.append(start_hat)
                        start_delay_vals.append(start_delay)
                        
                        stop_hat_vals.append(stop_hat)
                        stop_delay_vals.append(stop_delay)
                        
                        start_hat_dot = start_hat - start_delay
                        start_hat_dot_vals.append(start_hat_dot)
                        
                        stop_hat_dot = stop_hat - stop_delay
                        stop_hat_dot_vals.append(stop_hat_dot)
                        # print(reward)
                        R_hat_dot = (1/tau) * (R - R_hat)
                        R_dot_delay = (1/tau) * (R_hat - R_delay)
                        
                        R_hat += (R_hat_dot*dt2)
                        R_delay += (R_dot_delay*dt2)
                        
                        R_hat_vals.append(R_hat)
                        R_delay_vals.append(R_delay)
                        
                        R_hat_dot = R_hat - R_delay
                        R_hat_dot_vals.append(R_hat_dot)
                        
                        R_hat_dot_dot_delay = (1/(.25*tau)) * (R_hat_dot - R_hat_dot_delay)
                        R_hat_dot_delay += R_hat_dot_dot_delay * dt2
                        
                        R_hat_dot_dot = R_hat_dot - R_hat_dot_delay
                        
                        ceiling = 0.1
                        
                        start_dot_dt = sigmoid(start_hat_dot * R_hat_dot, gain=10, ceiling=ceiling) - (-.5 * ceiling)
                        stop_dot_dt = sigmoid(stop_hat_dot * R_hat_dot, gain=10, ceiling=ceiling) - (-.5 * ceiling)

                        c = baseline_noise + max(0,(1-abs(R_hat_dot))) * R_hat_dot_dot * noise_scale
                        # if params['surface_demo'] == False:
                        #    START_THRESHOLD += (start_dot_dt*dt2) + c * np.random.normal(0,np.sqrt(dt2))
                        #    STOP_THRESHOLD += (stop_dot_dt*dt2) + c * np.random.normal(0,np.sqrt(dt2))
                        
                        # Iteration end
                        pos_reward = np.where(reward > 0)[0]
                        
                        for i,r in enumerate(reward):
                            if int(responses[i]/dt) < reward_arr_plot.shape[0]:
                                reward_arr_plot[int(responses[i]/dt)] = r - penalty
                               
                        for i in range(1, reward_arr_plot.shape[0]):
                            R_t = reward_estimation[i-1] + ((dt * (-reward_estimation[i-1]/tau) + reward_arr_plot[i]/tau))
                            reward_estimation.append(R_t)
                        
                        if idx == 1:
                            old_RR = np.mean(reward_estimation[:-50])
                        
                        # hill-climbing for reward/responding boundaries
                        if idx > 1:
                            cur_RR = np.mean(reward_estimation[:-50])
                            old_RR = cur_RR
                        
                        update_and_reassign_ramps(timer, house_light_timer_value, active_ramp_indices, next_house_light_stimulus_type, stimulus_type, idx, ALLOCATION_PROB, NUM_RAMPS, RAMPS_PER_EVENT)
                   
                    for i in range(int(event_time/dt)+1, int(next_house_light_event_time/dt)): # int(event_data[idx+1][0]/dt)):  #
                        R_t = reward_est_vals[i-1] + ((dt * (-reward_est_vals[i-1]/tau) + reward_arr_plot[i]/tau))
                        reward_est_vals[i] = R_t
                        
                    for index, tup in enumerate(zip(active_ramp_indices, house_light_timer_value)):
                        i = tup[0]
                        val = tup[1]
                       
                        if timer.initiating_events[i] == stimulus_type: # timer.terminating_events[i] == next_house_light_stimulus_type and  
                            p = paths[index]
                            # if p[-1] < 4:
                            xs = np.linspace(event_time, next_house_light_event_time, paths[index].shape[0])
                            ax1.plot(xs, p, color=colors[timer.terminating_events[i]], alpha=0.5)
                            pxs = np.linspace(event_time, next_house_light_event_time, paths[index][::5].shape[0])
                            
                            if idx < 8:
                                pax1.plot(pxs, p[::5], color=colors[timer.terminating_events[i]], alpha=0.5)
                            elif index>=8 and index<=16:
                                pax2.plot(pxs, p[::5], color=colors[timer.terminating_events[i]], alpha=0.5)
                            else:
                                pax3.plot(pxs, p[::5], color=colors[timer.terminating_events[i]], alpha=0.5)

                    # Contiue to the next event in the house light interval
                    house_light_idx+=1
                else:
                    house_light_interval=False
        else:
            for i in range(int(event_time/dt), int(event_data[idx+1][0]/dt)):
                if i > reward_est_vals.shape[0] - 1:
                    break
                else:
                    R_t = reward_est_vals[i-1]
                    reward_est_vals[i:] = R_t

    window_size = 100


    threshold_times = []
    if plot:
        ax1.set_ylim([0,Y_LIM])
        ax1.set_xlim([0,400])
        ax1.set_ylabel("Activation")
        ax1.set_xlabel("Time")

    reward_arr_x = np.linspace(0,int(event_data[HOUSE_LIGHT_ON[-1]+1][0]), reward_arr_plot.shape[0])
    if plot:
        ax2.plot(reward_arr_x, reward_arr, label="reward")

    def smooth(y, box_pts):
        box = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(y, box, mode='full')
        return y_smooth

    def moving_average(a, n=3) :
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n
    ''' Newest reward rate estimation ''' 
    
    reward_estimation = [0]
    tau = 200
 
    reward_estimation = reward_est_vals
    if plot:  
        # ax3.plot(reward_arr_x, reward_estimation)
        ax3.plot(reward_arr_x, reward_est_vals, linestyle='--')
        start_thresh_vals = [step[0] for step in path]
        stop_thresh_vals = [step[1] for step in path]
        event_times = [step[2] for step in path]
        ax4.plot(event_times, start_thresh_vals, color = 'green', label = 'Start Threshold')
        ax4.plot(event_times, stop_thresh_vals, color = 'red', label = 'Stop Threshold')
        
        reward_fig = plt.figure(figsize=(8,4))
        ax = reward_fig.add_subplot(1, 1, 1)
        
        for idx, event in enumerate(event_data):
            color = colors[stimulus_type]
            
            ax.vlines(event[0],-.2,.2, color=color)
            if event[0] > event_times[-1]:
                break
        ax.plot(event_times, start_thresh_vals, color = 'green', label = 'Start Threshold')
        ax.plot(event_times, stop_thresh_vals, color = 'red', label = 'Stop Threshold')
        ax.plot(reward_arr_x, reward_est_vals*100, alpha = 0.3, label='Reward Est (x100)')
        ax.legend()
    plt.xlim(0,T)
    plt.show()
    average_reward = np.mean(reward_est_vals)
    
    return average_reward, path


params = {"dt":0.1,
              "N_EV_TYPES":3,
              "NOISE":0.02,
              "lr": 0.8,
              "k":5,
              "start_thresh":.6,
              "stop_thresh":1.2,
              "num_ramps":100,
              "ramps_per_event":10,
              "allocation_prob":.5,
              "surface_demo": False,
              "tau": 8,
              "baseline_noise": 2,
              "penalty": 0.05,
              "noise_scale":0,
              "event_data":event_data
              }

main(params)