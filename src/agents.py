from abc import ABC, abstractmethod
import torch
import torch.nn
import numpy as np
import gym
import random
from recordtype import recordtype
import os
from torch.multiprocessing import Value
import csv
import torch.nn.functional as F


#Abstract agent class. Will be used for pre-coded as well as the DRL agents. All it needs to do is receive observations and give actions.
class Agent(ABC):
    def __init__(self):
        self.tag = None

    #This method needs to return a probability distribution over available actions
    @abstractmethod
    def get_policy(self,obs):
        pass
    #This method returns the action as an integer (discrete) 
    @abstractmethod
    def get_action(self,obs):
        pass

    def print_tensorboard(self, plotter, iter_num):
        pass

#An agent that randomly selects a discrete action
class RandomAgent(Agent):
    def __init__(self, n_actions):
        self.n_actions = n_actions
    #Chooses all random actions with equal probability of 1/n_actions
    def get_policy(self,obs):
        return [1.0/self.n_actions for _ in self.n_actions]
    #Randomly sample and return action
    def get_action(self,obs):
        return np.random.randint(self.n_actions)

#An agent that is driven by a neural network
class DeepAgent(Agent):
    def __init__(self, net, tag, write_file=None, debug=False):
        self.net = net
        self.debug = debug
        self.tag = tag
        self.n_actions = net.n_actions
        self.write_file = open(write_file, 'w', newline='')
        self.csv_writer = csv.writer(self.write_file)

    def evaluate(self,obs):
        return self.net(torch.FloatTensor([obs]))

#An agent that selects actions through a discrete probability distrubtion given by a neural network (A2C, CEM, PG)
class PolicyAgent(DeepAgent):
    def __init__(self, net, tag=None, write_file=None, debug=False, epsilon=0):
        super(PolicyAgent, self).__init__(net, tag, write_file, debug)
        self.epsilon=epsilon
        self.csv_writer.writerow(['Prob(a=%d)' % (x) for x in range(self.n_actions)])
    #Samples the neural network for a policy and returns value directly
    def get_policy(self,obs):
        policy = self.evaluate(obs).policy[0]
        self.csv_writer.writerow(policy.detach().numpy())
        return policy
    #Randomly sample the from get_policy to select an action
    def get_action(self,obs):
        policy = self.get_policy(obs).detach()
        policy_dist_np = policy.numpy()
        action = np.random.choice(len(policy_dist_np), p=policy_dist_np)
        if random.random() < self.epsilon:
            print("Used renadom action instead of PolicyAgent action!")
            action = np.random.randint(len(policy_dist_np))
        return action

#An agent that uses the DQN based system of selecting actions as argmax Q
class ValueAgent(DeepAgent):
    def __init__(self, net , tag=None, write_file=None, debug=False, deterministic=False, epsilon=0):
        super(ValueAgent, self).__init__(net, tag, write_file, debug)
        self.epsilon=epsilon
        self.deterministic = deterministic
        if self.deterministic:
            self.headers = ['argmax A(s,a)', 'V(s)']
        else:
            self.headers = ['V(s)']
        for x in range(self.n_actions):
            self.headers.append('A(s,a=%d)' % (x))
            if not self.deterministic:
                self.headers.append('P(s,a=%d)' % (x)) 
        self.csv_writer.writerow(self.headers)
        
    def get_policy(self,obs):
        if self.deterministic:
            return self.get_greedy_policy(obs)
        else:
            return self.get_softmax_policy(obs)

    def get_action(self,obs):
        if self.deterministic:
            action = self.get_greedy_action(obs)
        else:
            action = self.get_softmax_action(obs)
        if random.random() < self.epsilon:
            print("Used renadom action instead of ValueAgent action!")
            action = np.random.randint(self.net.n_actions)
        return action 

    #Given that DQN's are deterministic the policy distrubtion is 0 for all actions except the one that it takes which has probability 1
    def get_greedy_policy(self,obs):
        policy_dist = torch.zeros(self.net.n_actions)
        action = self.get_action(obs)
        policy_dist[action] = 1
        return policy_dist

    #The action the DQN takes is the argmax of the advantage for each action.
    def get_greedy_action(self,obs):
        out = self.evaluate(obs)
        adv_t = out.advantages[0]
        _, action = adv_t.max(dim=-1)
        action = action.item()
        self.csv_writer.writerow([action, out.value[0].item()] + list(adv_t.detach().numpy()))
        return action

    def get_softmax_policy(self, obs):
        out = self.evaluate(obs)
        adv_t = out.advantages[0]
        policy = F.softmax(adv_t / adv_t.std(),dim=0)

        row = [out.value[0].item()]
        for x in range(adv_t.shape[0]):
            row.append(adv_t[x].item())
            row.append(policy[x].item())
        self.csv_writer.writerow(row)
        return policy
    
    def get_softmax_action(self, obs):
        policy = self.get_softmax_policy(obs).detach()
        policy_dist_np = policy.numpy()
        action = np.random.choice(len(policy_dist_np), p=policy_dist_np) 
        return action

    
#An agent that randomly selects a discrete action
class Deterministic(Agent):
    def __init__(self, env):
        self.n_actions = env.action_space.n
        self.env = env

    def get_policy(self,obs):
        policy_dist = torch.zeros(self.net.n_actions)
        action = self.get_action(obs)
        policy_dist[action] = 1
        return policy_dist

class FixedAgent(Deterministic):
    def __init__(self, env, fixed_action):
        super(FixedExpert, self).__init__(env)
        self.fixed_action = fixed_action

    def get_action(self,obs):
        return self.fixed_action


#Note that the decay is per-step and not based on episodes or anything like that
Epsilon = recordtype('Epsilon', field_names=['start', 'finish', 'decay'])

class EpsilonGreedy(Deterministic):
    def __init__(self, env, epsilon_schedule):
        super(EpsilonGreedy, self).__init__(env)
        self.epsilon_schedule = epsilon_schedule
        self.epsilon = epsilon_schedule.start
        #This should only be used as the ensemble agent and only if there are two agents to choose from
        assert env.action_space.n == 2

    def print_tensorboard(self, plotter, iter_num):
        plotter.plot(self.tag, 'Epsilon', iter_num, self.epsilon)

    def get_action(self,obs):
        #The reason I prompt the agent here is so that if they have an LSTM it still gets run through even if not selected
        self.epsilon = max(self.epsilon-self.epsilon_schedule.decay, self.epsilon_schedule.finish)
        return 1 if random.random() < self.epsilon else 0