import numpy as np
import torch
from torch._C import device
import torch.nn as nn

from collections import deque,namedtuple
import random


class DQN(nn.Module):
    def __init__(
            self,
            n_actions,
            n_features
    ):
        super(DQN, self).__init__()
        self.n_actions = n_actions
        self.n_features = n_features

        self.hidden = nn.Linear(self.n_features,256)
        self.output = nn.Linear(256,self.n_actions)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        x = self.act(self.hidden(x))
        x = self.output(x)
        x = self.softmax(x)

        return x

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
                        
class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
         

def choose_action(DQN_net, observation, epsilon,n_actions):
    if np.random.uniform() < epsilon:
        with torch.no_grad():
            return DQN_net(observation).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=observation.device, dtype=torch.long)

