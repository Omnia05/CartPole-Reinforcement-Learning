# CartPole Reinforcement Learning

Reinforcement learning approach to OpenAI Gym's CartPole environment.

## Description

The cart pole control environment consists of a cart that can move linearly and a rotating pole attached to it.

### Objective

Keep the pole vertical by applying horizontal forces (actions) to the cart.

### Action Space

- Push the cart left – denoted by 0
- Push the cart right – denoted by 1

### States

1. Cart Position: [-4.8, 4.8]
2. Cart Velocity: (-∞, ∞)
3. Pole Angle: [-0.418, 0.418] radians
4. Pole Angular Velocity: (-∞, ∞)

### Initialization

All observations are assigned a uniformly random value in the range (-0.05, 0.05).

### Episode Termination Conditions

- Pole Angle exceeds |0.2095| radians
- Cart Position exceeds |2.4|
- Number of steps in an episode exceeds a set limit 

### Reward

A reward of +1 is obtained for every step taken within an episode.

## Prerequisites

Make sure to have gym, numpy, time, matplotlib and pygame installed on your system.  
Instructions for installation:  
Run the following commands on your terminal.
- pip install gym
- pip install Matplotlib
- pip install pygame

## Note

This is my first project related to reinforcement learning. I am completely new to these topics and this is my initial attempt to understand the proceedings involved in reinforcement learning. As a start, I have implemented the Q Learning Algorithm. I hope to learn further and implement a DQN version of the same. 

  
  




  

         
