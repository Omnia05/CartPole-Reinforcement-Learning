#This class contains various functions inorder to develop the Q_Learning algorithm
import numpy as np

class Q_Learning:

    #initialisation function, which initialises the variables used throughout this class
    def __init__(self,env,alpha,gamma,epsilon,number_episodes, number_of_bins,lower_bounds,upper_bounds):
        import numpy as np

        self.env=env                           #cart-pole environment 
        self.alpha=alpha                       #step size
        self.gamma=gamma                       #discount rate
        self.epsilon=epsilon                   #epsilon, in the epsilon-greedy approach
        self.action_number= env.action_space.n #here, we have 2 actions, left(0) and right(1)
        self.number_episodes= number_episodes  #total number of simulation episodes
        self.number_of_bins= number_of_bins    #4-dimensional list that defines number of grid-points for each state's discretization
        self.lower_bounds=lower_bounds         #list with 4 entries for lower limits of discretization 
        self.upper_bounds=upper_bounds         #list with 4 entries for lower limits of discretization

        self.sum_rewards_episode=[]            #stores sum of rewards in each learning episode

        #action-value function matrix
        self.Qmatrix= np.random.uniform(low=0,high=1, size= (number_of_bins[0], number_of_bins[1], number_of_bins[2],number_of_bins[3], self.action_number))


    #returns the index tuple(4-dimensional) used to index entries of the Qvalue matrix
    def return_index_state(self,state):

        #state: array/list containing 4 entries i.e. cart position, cart velocity, pole angle, pole angular velocity
        position = state[0]
        velocity=state[1]
        angle= state[2]
        angular_velocity=state[3]

        #discretizing the 4 states
        cart_position_bin= np.linspace(self.lower_bounds[0],self.upper_bounds[0],self.number_of_bins[0])
        cart_velocity_bin= np.linspace(self.lower_bounds[1],self.upper_bounds[1],self.number_of_bins[1])
        pole_angle_bin= np.linspace(self.lower_bounds[2],self.upper_bounds[2],self.number_of_bins[2])
        pole_angle_velocity_bin= np.linspace(self.lower_bounds[3],self.upper_bounds[3],self.number_of_bins[3])

        #obtaining the indices of the bins of each state
        index_position= np.maximum(np.digitize(state[0], cart_position_bin)-1,0)
        index_velocity= np.maximum(np.digitize(state[1], cart_velocity_bin)-1,0)
        index_angle= np.maximum(np.digitize(state[2], pole_angle_bin)-1,0)
        index_angular_velocity= np.maximum(np.digitize(state[3], pole_angle_velocity_bin)-1,0)

        #returns a tuple containing the indices of the four states
        return tuple([index_position,index_velocity,index_angle,index_angular_velocity])
    

    #Epsilon-Greedy approach, for selecting an action
    def select_action(self,state,index): 
        #state: state for which to compute the action
        #index: index of the current episode                       

        #random actions to have enough exploration
        if index<500:
            return np.random.choice(self.action_number)
        
        #returns a random real number in the interval [0.0, 1.0)
        random_number = np.random.random()

        #decresing epsilon, to facilitate less exploration
        if index>7000:
            self.epsilon= 0.999 * self.epsilon


        if random_number < self.epsilon:  #exploration
            return np.random.choice(self.action_number)
        else:                             #greedy approach
            #selects first entry having maximum value
            return np.random.choice(np.where(self.Qmatrix[self.return_index_state(state)]==np.max(self.Qmatrix[self.return_index_state(state)]))[0])


    #function to simulate learning episodes
    def simulate_episodes(self):
        import numpy as np

        for index_episode in range(self.number_episodes):

            reward_episode=[]                                            #list that stores reward per episode

            (stateS,_)= self.env.reset()                                 #resets the environment at the beginning of every episode
            stateS = list(stateS)

            print("Simulating episode {}".format(index_episode))

            terminal_state= False                                        #loop terminates when a terminal state is reached 
            while not terminal_state:

                #discretized index of the state
                stateS_index= self.return_index_state(stateS)
                #select an action based on the current state 
                actionA= self.select_action(stateS,index_episode)

                #returns the next state, reward and a boolean
                (stateS_prime, reward, terminal_state,_,_)= self.env.step(actionA)
                reward_episode.append(reward)

                stateS_prime= list(stateS_prime)
                stateS_prime_index= self.return_index_state(stateS_prime)
                #returns the max value (we don't need actionA prime)
                Qmax_prime = np.max(self.Qmatrix[stateS_prime_index])

                if not terminal_state:
                    #append the tuples
                    error = reward + self.gamma*Qmax_prime - self.Qmatrix[stateS_index+ (actionA,)]
                    self.Qmatrix[stateS_index+ (actionA,)]= self.Qmatrix[stateS_index+ (actionA,)] + self.alpha*error
                else:
                    #here, we have Qmatrix[stateS_prime, actionA_prime]=0
                    error = reward- self.Qmatrix[stateS_index+ (actionA,)]
                    self.Qmatrix[stateS_index+ (actionA,)]= self.Qmatrix[stateS_index+ (actionA,)] + self.alpha*error

                #set current state to the next state
                stateS = stateS_prime

            print("Sum of rewards {}".format(np.sum(reward_episode)))
            self.sum_rewards_episode.append(np.sum(reward_episode))

            
    #function to simulate the learned optimal policy
    def simulate_learned_strategy(self):
        import gym
        import time

        #makes the Cart-Pole environment
        env1= gym.make('CartPole-v1', render_mode= 'human')
        (current_state,_)= env1.reset()    #resets the environment
        env1.render()                      #renders the environment
        time_steps=1000
        obtained_rewards=[]

        for time_index in range(time_steps):
            print(time_index)

            action_in_stateS= np.random.choice(np.where(self.Qmatrix[self.return_index_state(current_state)]==np.max(self.Qmatrix[self.return_index_state(current_state)]))[0])
            current_state, reward, terminated, truncated, info = env1.step(action_in_stateS)
            obtained_rewards.append(reward)
            time.sleep(0.05)              #inorder to properly render the environment   
            if(terminated):
                time.sleep(1)
                break

        return obtained_rewards,env1


    #function to simulate random actions many times 
    #this is implemented to evaluate the optimal learned policy in comparision to a random policy
    def simulate_random_strategy(self):
        import gym
        import time
        import numpy as np

        env2= gym.make('CartPole-v1') 
        (current_state,_)= env2.reset()
        env2.render()

        # number of simulation episodes
        episode_number=100
        # time steps in every episode
        time_steps=1000
        # sum of rewards in each episode
        sum_rewards_episode=[]

        for episode_index in range(episode_number):
            rewards_single_episode=[]
            initial_state= env2.reset()
            print(episode_index)

            for time_index in range(time_steps):
                random_action= env2.action_space.sample()
                observation, reward, terminated, truncated, info = env2.step(random_action)
                rewards_single_episode.append(reward)
                if(terminated):
                    break
            sum_rewards_episode.append(np.sum(rewards_single_episode))

            
                













