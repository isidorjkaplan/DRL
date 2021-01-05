#!/usr/bin/env python3
import gym
import gym.spaces
import numpy as np
import collections
import gym
import src.agents as agents
import src.worker as worker
import copy
"""
    This file is responsible for the classes relating to the Ensemble if it is used.
    This file (along with the calling function) could be removed and the rest of the program will still run normally. 
"""

"""
    The heart of the agent is this class. 
    This class wraps the enviornment replacing the action space with Discrete(num_of_agents). 
    The idea is that instead of selecting an action directly the agent will select another agent to take the action for it
    The ensemble agent does not need to be DRL, it can be something simple like epsilon-greedy where based on a decaying epsilon it chooses an agent or an expert
    All this requires is that you have multiple policies and need to choose between them at each step
"""
class SelectAgentWrapper(gym.Wrapper):
    def __init__(self, env, agents, debug=True):
        super(SelectAgentWrapper, self).__init__(env)
        self.agents = agents
        #The action space is to discretely pick another agent
        self.action_space = gym.spaces.Discrete(len(agents))
        self.debug = debug
        if self.debug:
            print("Wrapped Env in Ensemble")
            self.stats = np.zeros(shape=(len(agents),))
    
    #Reset function is wrapped to print debug info and otherwise just returns the observation unmodified
    def reset(self, **kwargs):
        self.obs = self.env.reset(**kwargs)
        if self.debug:
            print("Agent Stats: " + str(self.stats))
            self.stats = np.zeros(shape=(len(self.agents),))
        return self.obs

    #Replaces the step function for the environment
    def step(self, action):
        #The agent's action is to pick another agent and use their action instead. This is implemented here. 
        #We prompt all of the agents instead of only the one we are using. This is in case some agents internally have memory and need to see the observation for future steps. In particular for an LSTM
        actions = [agent.get_action(self.obs) for agent in self.agents]
        #Select the action we are actually going to execute
        actual_action = actions[action]
        if self.debug:
            self.stats[action] += 1
        #Interact with the enviornment based on the action selected by whichever policy our main agent selected
        next_state, reward, done, info = self.env.step(actual_action)
        info['actual_action'] = actual_action
        info['agent_stats'] = self.stats
        info['agents'] = self.agents
        self.obs = next_state
        return next_state, reward, done, info

"""
    If your ensemble agent is choosing between policies that are also training in paralell you need to pass that data to the individual agents. 
    For instance, say you have an Ensemble agent that chooses between a DQN and Policy Gradient agent at each step. 
    The data needs to be passed to the Ensemble (if it is DRL) and it also needs to be passed to the DQN and policy agents. 
    The way this is implemented is by repalcing the queue given to the Worker with a special queue that copies the information into the individual queues of each agent
"""
class EnsembleQueue():
    def __init__(self, ensemble_queue=None, agent_queues=None):
        self.ensemble_queue = ensemble_queue
        self.agent_queues = agent_queues if agent_queues is not None else []

    #This will take a list of steps where the action is which agent was selected and replace the action with the actual action that was taken in the enviornment
    def convert(self,steps):
        return [worker.Step(step.state, step.info['actual_action'], step.reward, step.next_state, step.is_done, step.info) for step in steps]

    #This can be used to add the queues of individual agents. All data fed into this queue will be copied to the queues of each individual agent
    def add_agent_queue(self,agent_queue):
        if agent_queue is not None:self.agent_queues.append(agent_queue)

    def set_ensemble_queue(self, ensemble_queue):
        self.ensemble_queue = ensemble_queue

    #The function to put something into the queue. When this is called it will put the data unmodified into the queue of the ensemble agent 
    #For each of the individual policies it will show the data but modified so that the actions represent the actual actions taken in the enviornment and not which agent was selected
    def put(self, obj):
        if self.ensemble_queue is not None:self.ensemble_queue.put(obj)
        if type(obj) is worker.Episode:
            obj = worker.Episode(self.convert(obj.steps), obj.reward, obj.speed)
        else:
            obj = self.convert(obj)
        #print(str(self) + " => " + str(self.agent_queues))
        for agent_queue in self.agent_queues:
            agent_queue.put(obj)

"""
    TODO: Test this out
    This was an untested idea I had early on
    Basically the idea is that instead of showign the ensemble agent the same data that we show the individual policies
        we can show it the observation from the enviornment but also show it the policies of each agent
        this way it sees what each agent wants to do and then it can select an agent based on that

    WARNING: THIS IS UNTESTED
"""
class PolicyAndStateWrapper(gym.ObservationWrapper):
    def __init__(self, env, agents):
        super(PolicyAndStateWrapper, self).__init__(env)
        self.agents = agents
        self.policy_space = gym.spaces.Box(low=0, high=1, shape=(env.action_space.n, len(agents)), dtype=np.uint8)
        self.env_obs_space = self.observation_space

        self.observation_space = gym.spaces.Tuple(env.observation_space, self.policy_space)

    def observation(self, obs):
        policy_state = np.array([agent.get_policy(obs).detach().numpy() for agent in self.agents])
        return (obs, policy_state)




