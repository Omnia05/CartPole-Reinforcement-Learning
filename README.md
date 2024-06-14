# CartPole Reinforcement Learning

Reinforcement learning approach to OpenAI Gym's CartPole environment

## Description

The cart pole control environment consists of a cart that can move linearly and a rotating pole attatched to it.  
Objective: Keep the pole vertical by applying horizontal forces(actions) to the cart  
Action Space:  
Push the cart left – denoted by 0  
Push the cart right – denoted by 1  
States:  
1. Cart Position: [-4.8,4.8]  
2. Cart Velocity: [-∞,∞]  
3. Pole Angle: [-0.418,0.418] radian
4. Pole Angular Velocity: [-∞,∞]
All observations are assigned a uniformly random value in (-0.05, 0.05)
Episode Termination Conditions:
1. Pole Angle is greater than |0.2095| radian  
2. Cart position is greater than |2.4|
3. Number of steps in an episode is over
Reward: A reward of +1 is obtained every time a step is taken within an episode.         
