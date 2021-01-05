import torch
import torch.nn
import numpy as np
import gym
from torch.multiprocessing import Process
import random
import time

import src.model as model
import src.agents as agents

from recordtype import recordtype

Episode = recordtype('Episode', field_names=['steps', 'reward', 'speed'])
Step = recordtype('Step', field_names=['state', 'action', 'reward', 'next_state', 'is_done', 'info'])

"""
The worker class is an object that handles all interaction with the enviornment and puts all of the data it collects into queues to be used for whatever they will be used for
This class is blind to the usage of the data and is also blind to the type of agent (it would even work with a heuristic or oracle as the agent)
"""
class Worker():
    def __init__(self, agent, plotter, episode_queue=None, batch_queue=None, batch_size=None, debug=False):
        assert episode_queue is not None or batch_queue is not None
        assert plotter is not None
        self.agent = agent
        self.plotter = plotter
        self.episode_queue = episode_queue
        self.batch_queue = batch_queue
        self.threads = []
        self.debug = debug
        self.batch_size=batch_size

    #If you want to start the worker as an asyncronous thread than call this function
    def start_async(self, env):
        p = Process(target=self.datagen, args=(env,))
        p.start()
        self.threads.append(p)

    #If the worker has an asyncronous thread running (it can have multiple) then this tells you if the thread is still running
    def is_alive(self):
        if len(self.threads) == 0:
            return False

        for thread in self.threads:
            if thread.is_alive():
                return True

    #This function starts an infinite loop that just keeps playing episodes forever. It is used in the asyncronous thread if used
    def datagen(self, env):
        episode_num = 0
        while True:
            self.play_episode(env, episode_num)
            episode_num += 1

    #This function is called to play an episode on the enviornment. It will play one full episode and then terminate. All the data will be stored in the queue
    def play_episode(self,env, episode_num=None):
        #If the agent is a deep learning agent and in particular if it uses an LSTM we need to clear the memeory. Unfortunately there was no way to make this self contained since it needs to be reset between episodes
        if hasattr(self.agent, 'net') and self.agent.net is model.LSTM:self.agent.net.clear_memory()
        start = time.time()
        episode_reward = 0
        obs = env.reset()
        is_done = False
        steps = []
        batch = []
        while not is_done:
            action = self.agent.get_action(obs)
            next_obs, reward, is_done, info = env.step(action)

            step = Step(obs, action, reward, next_obs, is_done, info)
            #Once the dat from a step is collected we store that data in the batch queue if enough steps have been collected. The batch queue will receive batches of steps.
            if self.batch_queue is not None:
                batch.append(step)
                if self.batch_size is not None and len(batch) >= self.batch_size:
                    self.batch_queue.put(batch)
                    batch = []

            steps.append(step)
            obs = next_obs
            episode_reward += reward

        #Calculate stats for the episode and than store the entire episode in the episode_queue
        episode_speed = 1/(time.time() - start)
        if self.batch_queue is not None and len(batch) > 0:self.batch_queue.put(batch)
        episode = Episode(steps, episode_reward, episode_speed)
        if self.episode_queue is not None:self.episode_queue.put(episode)

        if self.debug:print("Episode Finished: %s, len=%d, reward=%.3f" % (str(episode_num) if episode_num is not None else '', len(steps), episode_reward))
        if self.plotter is not None:
            self.plotter.plot('episode', 'reward', episode_num, episode_reward)
            self.plotter.plot('episode', 'length', episode_num, len(steps))
            self.plotter.plot('episode', 'speed', episode_num, episode_speed)
            if 'episode_udp_loss' in step.info.keys():
                self.plotter.plot('episode', 'udp_loss', episode_num, step.info['episode_udp_loss'])
                self.plotter.plot('episode', 'fec_loss', episode_num, step.info['episode_fec_loss'])
                self.plotter.plot('episode', 'avg_redundancy', episode_num, step.info['avg_redundancy'])

            last_step = episode.steps[-1]
            if 'agents' in last_step.info.keys():
                for agent_num, agent in enumerate(last_step.info['agents']):
                    percentage = last_step.info['agent_stats'][agent_num] / len(episode.steps)
                    self.plotter.plot(agent.tag, 'percentage_use', episode_num,percentage)
        return episode


