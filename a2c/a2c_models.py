from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch import Tensor
from torch.distributions import Categorical, Normal

## Standalone boilerplate before relative imports from https://stackoverflow.com/a/65780624
import sys
from pathlib import Path
if __package__ is None:                  
    DIR = Path(__file__).resolve().parent
    sys.path.insert(0, str(DIR.parent))
    __package__ = DIR.name

from util.sb3compat import SB3Compat


class A2C(nn.Module, SB3Compat):
    def __init__(self, n_actions, c_in, hidden=64, act_cls=nn.Tanh):
        super().__init__()
        self.n_actions, self.c_in, self.hidden = n_actions, c_in, hidden
        self.policy = nn.Sequential(
            nn.Linear(c_in, hidden),
            act_cls(),
            nn.Linear(hidden, hidden),
            act_cls(),
            nn.Linear(hidden, n_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(c_in, hidden),
            act_cls(),
            nn.Linear(hidden, hidden),
            act_cls(),
            nn.Linear(hidden, 1),
        )
        self.policy_logstd = nn.Parameter(torch.zeros(1, np.prod(n_actions)))

    def forward(self, x:Tensor, action:Tensor|None=None):
        action_dist = self.policy(x)
        values = self.critic(x)
        action_logstd = self.policy_logstd.expand_as(action_dist)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_dist, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), values

    def get_action(self, x:Tensor):
        action_dist = self.policy(x)
        action_logstd = self.policy_logstd.expand_as(action_dist)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_dist, action_std)
        return probs.sample()

    def get_value(self, x:Tensor):
        return self.critic(x).squeeze()