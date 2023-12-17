import numpy as np
import matplotlib.pyplot as plt
from random import uniform, sample
from tqdm import tqdm


class Agent:
    def __init__(self, k=10, epsilon=0):
        self.est_values = np.zeros(k)
        self.epsilon = epsilon
        self.k = k 
    
    def choose_action(self):
        choose_greedy = uniform(0, 1)
        if choose_greedy <= 1 - self.epsilon:
            max_est_value = np.max(self.est_values)
            max_actions = np.where(self.est_values == max_est_value)[0]

            return sample(list(max_actions), 1)[0]
        else:
            return sample(range(self.k), 1)[0]
    
    def update_est_value(self, reward, action, step_size=None):
        self.est_values[action] += (reward - self.est_values[action]) * step_size

class SampleAverageAgent(Agent):
    def __init__(self, k, epsilon):
        super().__init__(k, epsilon)
        self.N = np.zeros(k)

    def update_est_value(self, reward, action):
        self.N[action] += 1
        super().update_est_value(reward, action, 1/self.N[action])

class ExponentialRecencyWeightedAverageAgent(Agent):
    def __init__(self, k, epsilon, step_size):
        super().__init__(k, epsilon)
        self.step_size = step_size
    
    def update_est_value(self, reward, action):
        super().update_est_value(reward, action, self.step_size)


def run_agents(num_of_agents, time_steps, k=10, agent_selection=0, epsilon=0, const_step_size=0.1):
    avg_rewards = np.zeros(time_steps) # Ri
    optimal_actions_perc = np.zeros(time_steps)
    

    for j in tqdm(range(num_of_agents)):
        agent = SampleAverageAgent(k, epsilon) if not agent_selection else ExponentialRecencyWeightedAverageAgent(k, epsilon, const_step_size)
        true_values = np.zeros(k) #q*
        opt = 0
        
        for i in range(time_steps):
            action = agent.choose_action()
            reward = np.random.normal(true_values[action], 1)
            avg_rewards[i] += reward
            agent.update_est_value(reward, action)
            opt += np.max(true_values) == true_values[action]
            optimal_actions_perc[i] += opt / i
            true_values += np.random.normal(0, 0.01, k)

    avg_rewards = avg_rewards / num_of_agents
    optimal_actions_perc = optimal_actions_perc / num_of_agents

    return avg_rewards, optimal_actions_perc

def plot(avg_rewards, optimal_actions_perc, time_steps, labels):
    fig, ax = plt.subplots(2, 1)
  
    for i in range(len(avg_rewards)):
        ax[0].plot(range(1, time_steps + 1), avg_rewards[i], label=labels[i])
        ax[0].set_ylabel('Average Reward')

        ax[1].plot(range(1, time_steps + 1), optimal_actions_perc[i], label=labels[i])
        ax[1].set_xlabel('Steps')
        ax[1].set_ylabel('% Optimal Action')
    
    plt.show()

time_steps = 1000
num_of_agents = 10000
a = exponential_recency_weighted_average_res = run_agents(num_of_agents, time_steps, 10, 1, 0.1, 0.1)
#b = sample_average_res = run_agents(num_of_agents, time_steps, 10, 0, 0.1)

plot([a[0]], [a[1]], time_steps, ['Exponential Recency Weighted Average', 'Sample Average'])