import torch
import torch.nn
import numpy as np
import torch.optim as optim
import gym

import src.model as model
import src.agents as agents
import src.worker as worker
import time
from abc import ABC, abstractmethod
import copy
from torch.multiprocessing import Process
from collections import deque
from tensorboardX import SummaryWriter
import random
import torch.nn.functional as F
"""
    This file handles the actual DRL training algorythems. 
    The structure for training is that we have a "Trainer" class which takes in a DRL agent as well as an input source for observations
    Given observations and a model it then updates that model using some DRL algorythem such as Cross-Entropy or policy gradient
"""

"""
    This is the general Trainer that all specific trainers extend from. It handles things such as Tensorboard, the optimizer, starting threads.
"""
class Trainer(ABC):
    def __init__(self, agent, debug=False, plotter=None, learning_rate=0.01, save_every=10, iter_num=0, backup_file=None):
        #There needs to be a way to read data or this will fail
        self.debug = debug

        self.agent = agent
        self.backup_file = backup_file

        self.last_iter_time = time.time()
        self.plotter, self.tag = plotter, self.agent.tag

        self.iter_num = iter_num
        self.save_every = save_every

        if 'net' in self.agent.__dict__.keys():self.optimizer = optim.Adam(params=agent.net.parameters(), lr=learning_rate)

    #this function starts the trainer in it's own thead where it runs in an infinite loop
    def start_async(self):
        p = Process(target=self.train)
        p.start()
        return p
    
    #This function will train the agent while some condition passed to it holds true (or none for an infinite loop)
    def train(self, condition=None):
        while condition is None or condition():
            self.step()

    #The idea behind taking a step is that it will first read the data and then if it has enough data it will perform a step
    #This function will return true if an update was sucessfulyl performed and false if there was not enough data to perform an update
    def step(self):
        self.read_data()
        if self.ready_for_iter():
            self.iter()
            delta_time = time.time() - self.last_iter_time
            if self.plotter is not None:self.plotter.plot(self.tag,"iter_per_sec",self.iter_num,  1/delta_time)
            self.agent.print_tensorboard(self.plotter, self.iter_num)
            self.save()
            self.iter_num += 1
            self.last_iter_time = time.time()
            return True
        return False

    #A general function to save the model. It will save after a fixed number of steps and will also create backups of the model at a fixed frequency
    def save(self):
        if 'net' in self.agent.__dict__.keys() and self.agent.net.model_path is not None:
            #self.agent.net.save()
            if self.backup_file is not None:
                self.agent.net.save(model_path=self.backup_file)
                if self.save_every != 0 and self.iter_num % self.save_every == 0:
                    self.agent.net.save(model_path=self.backup_file[:-3] + '-iter' + str(self.iter_num) + '.pt')
            #if self.save_every != 0 and self.iter_num % self.save_every == 0:
            #    self.agent.net.save(model_path=self.agent.net.model_path[:-3] + '-iter' + str(self.iter_num) + '.pt')

    @abstractmethod
    def read_data(self):
        pass

    @abstractmethod
    def iter(self):
        pass

    @abstractmethod
    def ready_for_iter(self):
        pass

#The class to implement a trainer for the Cross-Entropy Method 
class CEMTrainer(Trainer):
    def __init__(self, agent, episode_queue, batch_queue=None, batch_size=16, percentile=75, debug=False, plotter=None, elite_buffer=False, save_every=1, learning_rate=0.01, iter_num=0, backup_file=None):
        super(CEMTrainer, self).__init__(agent=agent, debug=debug, plotter=plotter, save_every=save_every, learning_rate=learning_rate, iter_num=iter_num, backup_file=backup_file)
        assert type(agent) is agents.PolicyAgent
        self.batch = []
        self.batch_size = batch_size
        self.percentile = percentile
        self.objective = torch.nn.CrossEntropyLoss()
        self.elite_buffer = elite_buffer
        self.buffer = []
        self.episode_queue = episode_queue
        self.batch_queue = batch_queue
        assert self.episode_queue is not None

    #It is ready to do an update once the number of steps in the batch is more then the batch size
    def ready_for_iter(self):
        return len(self.batch) >= self.batch_size

    #Read all episodes and store them in the batch
    def read_data(self):
        #Flush the batch queue
        if self.batch_queue is not None:
            while self.batch_queue.empty():
                self.batch_queue.get()
        while not self.episode_queue.empty():
            if self.debug:
                print("Loaded episode!")
            self.batch.append(self.episode_queue.get())

    """
        To perform an itteration of Cross-Entropy it will take the top percentile of episodes 
            and then train to minimize the CrossEntropyLoss (-log(policy)) of the actions taken in those elite episodes
    """
    def iter(self):
        assert self.ready_for_iter()
        batch = self.batch
        speed = np.mean([episode.speed for episode in batch])
        reward_m = float(np.mean([episode.reward for episode in batch]))
        #If the elite buffer is enabled then we include elite_buffer episodes into the batch
        if self.elite_buffer:
            batch.extend(self.buffer)
            self.buffer.clear()
        
        rewards = [episode.reward for episode in batch]
        reward_b = np.percentile(rewards, self.percentile)

        #For each episode in the batch, if it's reward is above the reward_bound, add it to a list of elite episodes
        elite_episodes = []
        for episode in batch:
            if episode.reward >= reward_b:
                elite_episodes.append(episode)
        #Set the batch to be the elite_episodes, we can throw out all other episodes here
        batch = elite_episodes

        #For each episode, add up the losses and the entropys to get the sum of all losses and entropys
        arr_action_scores_v, arr_acts_v = [],[]
        entropy_v = 0
        for action_scores_v, acts_v,batch_entropy_v in [self.get_loss(episode) for episode in batch]:
            arr_action_scores_v.append(action_scores_v)
            arr_acts_v.append(acts_v)
            entropy_v = entropy_v + batch_entropy_v
        #For the entropy we are just taking the average value fo the entropy. This is a debugging quantitiy anyways so no worries about episode weightings
        entropy_v /= len(batch)
        #Create a new vector that has all the action scores from each of the episodes in the batch
        action_scores_v = torch.cat(arr_action_scores_v, dim=0)
        acts_v = torch.cat(arr_acts_v, dim=0)
        #Call the objective function on the vector of all action_scores and all action values for the entire batch
        loss_v = self.objective(action_scores_v, acts_v)
        #Clear the optimizer and then perform the gradient updates
        self.optimizer.zero_grad()
        loss_v.backward()
        self.optimizer.step()

        if self.debug:
            print("%d: loss=%.3f, reward_mean=%.1f, reward_bound=%.1f" % (
            self.iter_num, loss_v.item(), reward_m, reward_b))
        if self.plotter is not None:
            self.plotter.plot(self.tag, 'loss', self.iter_num, loss_v.item())
            self.plotter.plot(self.tag, 'reward_bound', self.iter_num, reward_b)
            self.plotter.plot(self.tag, 'reward_mean', self.iter_num, reward_m)
            self.plotter.plot(self.tag, 'entropy', self.iter_num, entropy_v.item())
        self.batch.clear()

    #This function returns the CrossEntropyLoss for a specific episode. The actual loss that the network updates on is the average of the loss on the top episodes
    def get_loss(self, episode):
        acts_v = torch.LongTensor([step.action for step in episode.steps])
        obs_v = torch.FloatTensor([step.state for step in episode.steps])

        if self.agent.net is model.LSTM:self.agent.net.clear_memory()
        data = self.agent.net(obs_v)
        action_scores_v = data.policy_logits

        entropy_v = -1*(data.policy * torch.log(data.policy)).sum(dim=1).mean()

        return action_scores_v, acts_v, entropy_v

"""
    This trainer implements the DQN algorythem. The idea is that it will keep taking in pairs of (s,a,r,s') and then sample that. 
    It will train the q_value(s,a) -> reward(s,a) + GAMMA*value(s')

    Since our neural network has a value and advantage head we use a duelling DQN such that Q(s,a) = V(s) + A(s,a) - MEAN(A)

    All data is stored and sampled from a replay buffer
"""
class DQNTrainer(Trainer):
    def __init__(self, agent, batch_queue, batch_size_episodes=10, max_buffer_size=10000, gamma=1, sync_target=10, learning_rate=0.01, episode_queue=None, plotter=None, debug=False,save_every=50,iter_num=0, backup_file=None):
        super(DQNTrainer, self).__init__(agent=agent, plotter=plotter, debug=debug, learning_rate=learning_rate,save_every=save_every, iter_num=iter_num, backup_file=None)
        assert type(agent) is agents.ValueAgent
        self.episode_queue = episode_queue
        self.batch_queue = batch_queue
        self.tgt_net = copy.deepcopy(agent.net)
        self.replay_buffer = deque()
        self.batch_size = batch_size_episodes
        self.gamma = gamma
        self.sync_target = sync_target
        self.max_buffer_size = max_buffer_size
        if self.episode_queue is not None:
            self.episode_num = 0

    def ready_for_iter(self):
        #if self.debug:print(str(len(self.replay_buffer)) + " >= " + str(self.batch_size))
        return len(self.replay_buffer) >= self.batch_size


    def read_data(self):
        while not self.batch_queue.empty():
            self.replay_buffer.append(self.batch_queue.get())
            if self.debug:print("Loaded Batch")

        while len(self.replay_buffer) > self.max_buffer_size:self.replay_buffer.popleft()

        if self.episode_queue is None:
            return
        while not self.episode_queue.empty():
            episode = self.episode_queue.get()
            #if self.agent.epsilon_schedule is not None:self.writer.add_scalar("epsilon", self.agent.epsilon.value, self.episode_num)
            self.episode_num+=1
        

    #Performs an itteration. In our case I chose to sample the replay buffer for multiple tuples of (s,a,s',r) that are in order instead of just taking individual steps.
    #This is done so that the LSTM is in order if it is used. It does not imact training negatively if it is not used 
    def iter(self):
        if self.iter_num % self.sync_target == 0:
            if self.debug:
                print("Loaded Target Network at iter:  " + str(self.iter_num))
            self.tgt_net.load_state_dict(self.agent.net.state_dict())

        #Contains sequences of steps that go in order. Done this way for the LSTM so that steps are grouped but we randomly decide which groups of steps to include from the replay buffer
        batches = random.sample(self.replay_buffer, self.batch_size)
        loss_v = torch.cat([self.get_loss(batch).view(1) for batch in batches]).mean()
        self.optimizer.zero_grad()
        loss_v.backward()
        self.optimizer.step()

        if self.debug:
            print("%d: loss=%.3f" % (self.iter_num, loss_v.item()))
        if self.plotter is not None:
            self.plotter.plot(self.tag, "loss", self.iter_num, loss_v.item())

    #Takes in a batch of tuples of (s,a,r,s') and calculates the MSE loss Q(s,a) -> R(s,a) + GAMMA*V(s')
    def get_loss(self,batch):
        states_t = torch.FloatTensor([step.state for step in batch])
        rewards_t = torch.FloatTensor([step.reward for step in batch])
        next_states_t = torch.FloatTensor([step.next_state for step in batch])
        dones = torch.BoolTensor([step.is_done for step in batch])
        actions_t = torch.LongTensor([step.action for step in batch])
        
        if self.agent.net is model.LSTM:self.agent.net.clear_memory()
        data = self.agent.net(states_t)
        values_t = data.value
        adv_t = data.advantages
        
        q_vals = (values_t.view(-1,1) + adv_t - adv_t.mean(dim=1, keepdim=True)).gather(1, actions_t.unsqueeze(-1)).squeeze(-1)
        #print("Q: " + str(q_vals.mean()) + ": " + str(q_vals[:6]))

        next_values_t = self.tgt_net(next_states_t).value.detach().view(-1)
        next_values_t[dones] = 0.0
                
        target_values_t = rewards_t + self.gamma*next_values_t
        #print("Target: " + str(target_values_t.mean()) + ": " + str(target_values_t[:6]))

        loss_v = F.mse_loss(q_vals, target_values_t)
        return loss_v

class PolicyTrainer(Trainer):
    def __init__(self, agent, batch_queue, gamma=1, batch_size=100, learning_rate=0.0001, episode_queue=None,plotter=None, debug=False, critic=True, actor_weight=1, critic_weight=1, entropy_weight=0.01, save_every=1000, episode_tensorboard=True, iter_num=0, backup_file=None):
        super(PolicyTrainer, self).__init__(agent=agent, plotter=plotter, debug=debug, learning_rate=learning_rate,save_every=save_every, iter_num=iter_num, backup_file=backup_file)
        assert type(agent) is agents.PolicyAgent
        assert batch_queue is not None
        self.episode_tensorboard = episode_tensorboard
        self.batch_queue = batch_queue
        self.episode_queue = episode_queue
        self.critic = critic
        self.gamma = gamma
        self.actor_weight = actor_weight
        self.critic_weight = critic_weight
        self.episode_num = 0
        self.batch = []
        self.batch_size = batch_size
        self.entropy_weight = 1



    def read_data(self):
        while not self.batch_queue.empty():
            self.batch.extend(self.batch_queue.get())
            if self.batch[-1].is_done:break

        if self.episode_queue is None:
            return
        while not self.episode_queue.empty():
            episode = self.episode_queue.get()
            self.episode_num+=1

    def ready_for_iter(self):
        return len(self.batch) >= self.batch_size

    def iter(self):
        if self.batch[-1].is_done and self.agent.net is model.LSTM:self.agent.net.clear_memory()
        batch = self.batch
        states_t = torch.FloatTensor([step.state for step in batch])
        rewards_t = torch.FloatTensor([step.reward for step in batch])
        actions_t = torch.LongTensor([step.action for step in batch])

        disc_rewards = []
        disc_reward = None
        for i in reversed(range(len(batch))):
            step = batch[i]
            if step.is_done or (disc_reward is None and not self.critic):
                disc_reward = 0
            elif disc_reward is None and self.critic:
                disc_reward = self.agent.evaluate(step.next_state).value[0].detach().item()
            else:
                disc_reward = rewards_t[i].item() + self.gamma*disc_reward
            disc_rewards.insert(0,disc_reward)
        disc_rewards_t = torch.FloatTensor(disc_rewards)

        data = self.agent.net(states_t)
        if self.critic:
            values_t = data.value.detach()
            scale_t = disc_rewards_t - values_t.view(-1)
            #NOTE! MAke sure to check the
            #scale_t = (scale_t - scale_t.mean())/scale_t.std()
        else:
            scale_t = (disc_rewards_t - disc_rewards_t.mean())/disc_rewards_t.std()

        log_prob_v = F.log_softmax(data.policy_logits,dim=1)
        log_prob_actions_v = log_prob_v[range(len(batch)), actions_t]

        actor_loss_t = (-1*self.actor_weight * scale_t * log_prob_actions_v).mean()

        entropy_loss_v = -1*self.entropy_weight * (data.policy * log_prob_v).sum(dim=1).mean()

        if self.critic:
            critic_loss_t = self.critic_weight * F.mse_loss(disc_rewards_t, data.value.view(-1))
            loss_t = actor_loss_t + critic_loss_t - entropy_loss_v
        else:
            loss_t = actor_loss_t - entropy_loss_v

        self.optimizer.zero_grad()
        loss_t.backward()
        self.optimizer.step()
        self.batch.clear()

        if self.debug:
            print("%d: loss=%.3f, actor_loss=%.3f, critic_loss=%.3f" % (self.iter_num, loss_t.item(), actor_loss_t.item(), critic_loss_t.item() if self.critic else 0))

        if self.plotter is not None:
            #self.writer.add_scalar("loss", loss_t.item(), self.iter_num)
            self.plotter.plot(self.tag, "loss actor", self.iter_num,actor_loss_t.item())
            self.plotter.plot(self.tag, "entropy", self.iter_num, entropy_loss_v.item())
            if self.critic:self.plotter.plot(self.tag, "loss critic", self.iter_num, critic_loss_t.item())

