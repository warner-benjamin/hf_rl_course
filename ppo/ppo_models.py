# PPO Models for Atari Gyms
# Base PPO modified from CleanRL

# Contains code from:
# CleanRL - MIT License - Copyright (c) 2019 CleanRL developers
# Stable Baselines3 - MIT License - Copyright (c) 2019 Antonin Raffin


import torch
from torch import nn
from torch.distributions import Categorical, Normal

import math

## Standalone boilerplate before relative imports from https://stackoverflow.com/a/65780624
import sys
from pathlib import Path
if __package__ is None:                  
    DIR = Path(__file__).resolve().parent
    sys.path.insert(0, str(DIR.parent))
    __package__ = DIR.name

from util.sb3compat import SB3Compat


def layer_init(layer, std=math.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class AtariPPO(nn.Module, SB3Compat):
    def __init__(self, n_actions, c_in=4, normalize=True, act_cls=nn.ReLU):
        super().__init__()
        self.n_actions, self.c_in, self.normalize = n_actions, c_in, normalize
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(c_in, 32, 8, stride=4)),
            act_cls(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            act_cls(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            act_cls(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            act_cls(),
        )
        self.actor = layer_init(nn.Linear(512, n_actions), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def norm(self, x):
        if self.normalize:
            with torch.no_grad():
                return x / 255.0
        else:
            return x

    def get_value(self, x):
        return self.critic(self.network(self.norm(x)))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(self.norm(x))
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)