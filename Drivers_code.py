#This is the drivers code, depicting the advantage of the Q_Learning algorithm over a random policy

#OpenAI's gym
import gym
import time
import numpy as np
import matplotlib.pyplot as plt             #used to plot the graphs

from Q_Learning import Q_Learning            #imports the Q_Learning class

#setting up the environment
env=gym.make('CartPole-v1')
(state,_)= env.reset()

#defining the parameters for state discretization (can be changed accordingly)
upper_bounds= env.observation_space.high
lower_bounds=env.observation_space.low
cart_velocity_min=-3
cart_velocity_max=3
pole_angle_velocity_min=-10
pole_angle_velocity_max=10
upper_bounds[1]=cart_velocity_max
upper_bounds[3]=pole_angle_velocity_max
lower_bounds[1]=cart_velocity_min
lower_bounds[3]=pole_angle_velocity_min

number_of_bins_position=30
number_of_bins_velocity=30
number_of_bins_angle=30
number_of_bins_angle_velocity=30
number_of_bins=[number_of_bins_position,number_of_bins_velocity,number_of_bins_angle,number_of_bins_angle_velocity]

#defining various constant parameters (can be changed accordingly)
alpha=0.1
gamma=1
epsilon=0.2
number_episodes=15000

#creating an object Q1 based on the Q_Learning class
Q1=Q_Learning(env,alpha,gamma,epsilon,number_episodes,number_of_bins,lower_bounds,upper_bounds)
#runs the Q learning algorithm
Q1.simulate_episodes()

#simulates the optimal learned strategy, 100 times (can be changed accordingly)
for i in range(100):
    (obtained_rewards_optimal,env1)= Q1.simulate_learned_strategy()

#plotting the figure for the optimal learned strategy
plt.figure(figsize=(12,5))
plt.plot(Q1.sum_rewards_episode,color='blue',linewidth=1)
plt.xlabel('Episode')
plt.ylabel('Sum of Rewards in Episode')     
plt.yscale('log')
plt.savefig('convergence.png')
plt.show()
#close the environment
env1.close()
#get the sum of rewards
np.sum(obtained_rewards_optimal)


#simulating a random strategy
(obtained_rewards_random,env2)= Q1.simulate_random_strategy()
#plots a histogram for the random strategy
plt.hist(obtained_rewards_random)
plt.xlabel('Sum of Rewards')
plt.ylabel('Percentage')
plt.savefig('histogram.png')
plt.show()









