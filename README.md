# tddm-heuristic-model
By: Robert Klock

A constructive interval timing algorithm.

<h1> Abstract </h2>
We attempt to develop a high-level probabilistic and constructive timing learning algorithm. 

<h2> Suggested Reading </h2>
<ul>
  <item>A Model of Interval Timing by Neural Integration: https://www.jneurosci.org/content/31/25/9238 </item>
  <item>Adaptive Drift-Diffusion Process to Learn Time Intervals: https://arxiv.org/abs/1103.2382 </item>
 </ul> 

<h1> Getting Started </h1>
It is recommended to run the code in Spyder. If you don't have Spyder installed, you can install it through Anaconda [here](https://www.anaconda.com/).  

<h2> Running the simulation </h2>
You can adjust most of the important features of the model by changing the global variables marked in the code. This includes dt (the granularity of time simulated), N_EVENT_TYPES (the number of event types present in the simulation, think: A, B, C, D ...), and NOISE (the amount of noise inherent across the simulation. No noise makes the model deterministic).

<h3> Generating Events </h3>
There are two ways to generate events. You can call the getSamples method from the Timer module or you can create your own sequence of events as an array of array triples in the form [time from last event, event type, stimulus type]. As of now (4/20/22), event type should be equal to stimulus type. 
You can also control when the house light is on. Set HOUSE_LIGHT_ON to be an array of event indices. Example: [0,1,2,5,6,7,8] means the house light will be on for the first three events, off, and back on for events 6-8. The house light determines when the timing model should stop timing intra-event durations (for example, A,B,C). 

<h3> Building a timer </h4>
Building a timer is as simple as calling the Timer Module object: <code>timer=TM(1,100)</code>. This generates a timer with 100 ramps, all initialized with weight=1. Weight is equivalent to the rate of accumulation in the Drift Diffusion Model

<h3> Responding </h4>

If you want responses similar to the beat the clock task, call the <code>respond</code> method.
