 
import torch
import os

import threading

import torch.nn as nn

import numpy as np
from abc import ABC, abstractmethod

from recordtype import recordtype

import torch.nn.functional as F
"""
    This file contains all of the neural network models that can be used with this program. More can be added as long as they conform to the following protocol:
        Observation => Neural Network => Policy, Value, Advantages
    
    Each model has a head for policy and a value/advantage head. The reason for this is so that the models themselvs are compatable with any of the training algorythems
    At least one head of the model will be unused. For instance in Cross-Entropy method the value/advatnage head is unused and in DQN the Policy head is unused.
"""
Output = recordtype('Output', field_names=['policy', 'value', 'advantages', 'policy_logits'])

#A base file that handles saving and loading models. All other models should extend this file
class Module(nn.Module):
    def __init__(self, model_path=None, debug=True):
        super(Module, self).__init__()
        self.model_path = model_path
        self.debug = debug
        self.share_memory()

    def set_path(self,model_path):
        self.model_path = model_path

    def load(self,model_path=None):
        if model_path is None:model_path=self.model_path
        if self.model_path is None:self.model_path=model_path
        assert model_path is not None# and os.path.exists(self.model_path)
        if self.debug:
            print("Loading Model From: " + str(model_path))
        self.load_state_dict(torch.load(model_path))

    def save(self,model_path=None):
        if model_path is None:model_path=self.model_path
        assert model_path is not None
        torch.save(self.state_dict(), model_path)

#A model that handles convolutional layers. Not fully implemented as the heads are handled by a seperate class below
class ConvBase(Module):
    def __init__(self, input_shape, n_actions):
        super(ConvBase, self).__init__()
        self.input_shape = input_shape
        self.n_actions = n_actions

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.conv_out_size = self._get_conv_out(input_shape)
        self.hidden_layer_size = 512
        self.linear = nn.Sequential(
            nn.Linear(self.conv_out_size, self.hidden_layer_size),
            nn.Sigmoid()
        )
        

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, obs):
        out = self.conv(obs).view(obs.size()[0], -1)
        out = self.linear(out)
        return out

#A convolutional model. Takes in some shape and than performs conolution and then spits out the policy/value/advantage
class ConvNet(ConvBase):
    def __init__(self, input_shape, n_actions):
        super(ConvNet, self).__init__(input_shape, n_actions)

        self.policy = nn.Linear(self.hidden_layer_size, n_actions)
        self.value = nn.Linear(self.hidden_layer_size, 1)
        self.advantages = nn.Linear(self.hidden_layer_size, n_actions)
        
    def forward(self, obs):
        out = super().forward(obs)

        policy_logits_t = self.policy(out) if self.policy is not None else None
        policy_t = F.softmax(policy_logits_t, dim=1)

        value_t = self.value(out).view(-1) if self.value is not None else None
        adv_t = self.advantages(out) if self.advantages is not None else None
        return Output(policy_t, value_t, adv_t, policy_logits_t)

"""
    In many classes of problems a LSTM (long-short term memory) network is required. 
    The idea behind this network is that it has a memory of previous states by passing a hidden state between calls

    I made the design choice of keeping this completely self-contained.
    That is to say, the model stores as a local variable the value of the hidden state and will automatically pass it between function calls
    As a result of this architecture it is required that you call LSTM.clear_memeory() between episodes so that it does not remember data from previous episodes into other episodes

    I also made the design choice of keeping the hidden states stored in a thread-global dictionary so that if you have the same model being run
        in seperate processes then each process will have a seperate remembered hidden_state. This is required so that the calls to the LSTM during training
        and the calls to the LSTM during data collection do not interefere with each other. 

        The only alternative to this would be passing the hidden state during data collection and training but this runs contrary to the phillosiphy of the model being totally self contained
        I do not want the other code to depend on weather or not the model is an LSTM or not. 
"""
class LSTM(Module):
    def __init__(self, input_shape, n_actions, hidden_units=64, batch_size=1, num_layers=2):
        super(LSTM, self).__init__()
        self.input_shape = input_shape[0]
        self.hidden_units = hidden_units
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.n_actions = n_actions

        self.lstm = nn.LSTM(self.input_shape, self.hidden_units, self.num_layers)

        self.policy = nn.Linear(self.hidden_units, self.n_actions)
        self.value = nn.Linear(self.hidden_units, 1)
        self.advantages = nn.Linear(self.hidden_units, self.n_actions)
        
        self.clear_memory()

    #This function will check if the current thread has a map for LSTM hidden states. If it does not then it will create a new thread-specific-global 
    #Map and then store that for this particular LSTM the value of the hidden state is nto
    def setup_hidden(self):
        global hidden_map
        if not "hidden_map" in globals():
            if self.debug:
                print("Setup new thread hidden_map")
            #Note that global variables are thread local
            hidden_map = {}
            hidden_map[self] = None

    #CLear the value of the hidden state for between episodes
    def clear_memory(self):
        self.set_hidden(None)

    #Return what the current value of the hidden state is 
    def get_hidden(self):
        self.setup_hidden()
        if not self in hidden_map:
            self.set_hidden(None)
        return hidden_map[self]

    #Set the hidden state to a new value 
    def set_hidden(self, hidden_state):
        self.setup_hidden()
        global hidden_map
        hidden_map[self] = hidden_state

    #The function to call the network. You can choose to pass it a hidden state instead of the one it has in memory, if not it will use it's internally remembered value
    def forward(self, obs, hidden_state=None):
        if hidden_state is None and self.get_hidden() is not None:hidden_state = self.get_hidden()
    
        if hidden_state is not None:
            hidden = (hidden_state[0].detach(), hidden_state[1].detach())
            lstm_out, hidden_state = self.lstm(obs.view(len(obs), self.batch_size, -1), hidden)
        else:
            lstm_out, hidden_state = self.lstm(obs.view(len(obs), self.batch_size, -1))
        self.set_hidden(hidden_state)

        out = lstm_out.view(len(obs), self.hidden_units)

        policy_logits_t = self.policy(out) if self.policy is not None else None
        policy_t = F.softmax(policy_logits_t, dim=1)

        value_t = self.value(out).view(-1) if self.value is not None else None
        adv_t = self.advantages(out) if self.advantages is not None else None
        return Output(policy_t, value_t, adv_t, policy_logits_t)

#A simple feed-forward neural network consisting of only linear layers 
class Linear(Module):
    def __init__(self, input_shape, n_actions, hidden_layer=64):
        super(Linear, self).__init__()
        self.input_shape = input_shape[0]
        self.n_actions = n_actions
        self.hidden_layer = hidden_layer
        self.linear = nn.Sequential(
            nn.Linear(self.input_shape, self.hidden_layer),
            nn.ReLU(),
        )

        self.policy = nn.Linear(self.hidden_layer, n_actions)
        self.value = nn.Linear(self.hidden_layer, 1)
        self.advantages = nn.Linear(self.hidden_layer, n_actions)
        
    def forward(self, obs):
        obs = obs.view(len(obs),-1)
        out = self.linear(obs)

        policy_logits_t = self.policy(out) if self.policy is not None else None
        policy_t = F.softmax(policy_logits_t, dim=1)

        value_t = self.value(out).view(-1) if self.value is not None else None
        adv_t = self.advantages(out) if self.advantages is not None else None
        return Output(policy_t, value_t, adv_t, policy_logits_t)