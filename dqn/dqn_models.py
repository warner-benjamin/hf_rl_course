# DQN Models for Atari Gyms
# Base DQN modified from CleanRL
# Dueling DQN modified from CleanRL & labml.ai
# Impala modified from Fast and Data-Efficient Training of Rainbow

# Contains code from:
# CleanRL - MIT License - Copyright (c) 2019 CleanRL developers
# labml.ai  Annotated Papers - MIT License - Copyright (c) 2020 Varuna Jayasiri
# Stable Baselines3 - MIT License - Copyright (c) 2019 Antonin Raffin
# Fast and Data-Efficient Training of Rainbow - MIT License - Copyright (c) 2021 Dominik Schmidt


import torch
from torch import nn

## Standalone boilerplate before relative imports from https://stackoverflow.com/a/65780624
import sys
from pathlib import Path
if __package__ is None:                  
    DIR = Path(__file__).resolve().parent
    sys.path.insert(0, str(DIR.parent))
    __package__ = DIR.name

from util.sb3compat import SB3Compat


class DQNBase(nn.Module):
    def norm(self, x):
        if self.normalize:
            with torch.no_grad():
                return x / 255.0
        else:
            return x

    def get_action(self, x:torch.Tensor):
        return self.forward(x).argmax(dim=1).reshape(-1)


class DQN(DQNBase, SB3Compat):
    def __init__(self, n_actions, dueling=False, c_in=4, normalize=True, act_cls=nn.ReLU):
        super().__init__()
        self.n_actions, self.dueling = n_actions, dueling
        self.c_in, self.normalize = c_in, normalize
        self.network = nn.Sequential(
            nn.Conv2d(c_in, 32, 8, stride=4),
            act_cls(),
            nn.Conv2d(32, 64, 4, stride=2),
            act_cls(),
            nn.Conv2d(64, 64, 3, stride=1),
            act_cls(),
            nn.Flatten(),
        )
        self.action_value = nn.Sequential(
            nn.Linear(3136, 512),
            act_cls(),
            nn.Linear(512, n_actions),
        )
        self.state_value = nn.Sequential(
            nn.Linear(3136, 512),
            act_cls(),
            nn.Linear(512, 1),
        )

    def forward(self, x:torch.Tensor):
        x = self.network(self.norm(x))
        action_value = self.action_value(x)
        if self.dueling:
            state_value = self.state_value(x)

            action_score_centered = action_value - action_value.mean(dim=-1, keepdim=True)
            return state_value + action_score_centered
        else:
            return action_value



class ImpalaSmall(DQNBase, SB3Compat):
    """
    Implementation of the small variant of the IMPALA CNN introduced in Espeholt et al. (2018).
    """
    def __init__(self, n_actions, dueling=False, c_in=4, normalize=True, act_cls=nn.ReLU):
        super().__init__()
        self.n_actions, self.dueling = n_actions, dueling
        self.c_in, self.normalize = c_in, normalize

        self.main = nn.Sequential(
            nn.Conv2d(c_in, 16, kernel_size=8, stride=4),
            act_cls(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            act_cls(),
        )
        self.pool = torch.nn.AdaptiveMaxPool2d((6, 6))
        self.flatten = nn.Flatten()
        self.action_value = nn.Sequential(
            nn.Linear(1152, 256),
            act_cls(),
            nn.Linear(256, n_actions),
        )
        self.state_value = nn.Sequential(
            nn.Linear(1152, 256),
            act_cls(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        x = self.main(self.norm(x))
        x = self.flatten(self.pool(x))
        action_value = self.action_value(x)
        if self.dueling:
            state_value = self.state_value(x)
            action_score_centered = action_value - action_value.mean(dim=-1, keepdim=True)
            return state_value + action_score_centered
        else:
            return action_value


class ImpalaResidual(nn.Module):
    """
    Simple residual block used in the large IMPALA CNN.
    """
    def __init__(self, depth, norm_func, act_cls=nn.ReLU):
        super().__init__()

        self.relu_0 = act_cls()
        self.conv_0 = norm_func(nn.Conv2d(depth, depth, kernel_size=3, stride=1, padding=1))
        self.relu_1 = act_cls()
        self.conv_1 = norm_func(nn.Conv2d(depth, depth, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        x_ = self.conv_0(self.relu_0(x))
        x_ = self.conv_1(self.relu_1(x_))
        return x+x_


class ImpalaBlock(nn.Module):
    """
    Three of these blocks are used in the large IMPALA CNN.
    """
    def __init__(self, depth_in, depth_out, norm_func, act_cls=nn.ReLU):
        super().__init__()

        self.conv = nn.Conv2d(depth_in, depth_out, kernel_size=3, stride=1, padding=1)
        self.max_pool = nn.MaxPool2d(3, 2, padding=1)
        self.residual_0 = ImpalaResidual(depth_out, norm_func=norm_func, act_cls=act_cls)
        self.residual_1 = ImpalaResidual(depth_out, norm_func=norm_func, act_cls=act_cls)

    def forward(self, x):
        x = self.conv(x)
        x = self.max_pool(x)
        x = self.residual_0(x)
        x = self.residual_1(x)
        return x


class ImpalaLarge(DQNBase, SB3Compat):
    """
    Implementation of the large variant of the IMPALA CNN introduced in Espeholt et al. (2018).
    """
    def __init__(self, n_actions, dueling=False, width=1, spectral_norm=False, c_in=4, normalize=True, act_cls=nn.ReLU):
        super().__init__()
        self.n_actions, self.dueling = n_actions, dueling
        self.c_in, self.width, self.spectral_norm, self.normalize, self.act_cls = c_in, width, spectral_norm, normalize, act_cls
        def identity(p): return p

        norm_func = torch.nn.utils.spectral_norm if (spectral_norm == 'all') else identity
        norm_func_last = torch.nn.utils.spectral_norm if (spectral_norm == 'last' or spectral_norm == 'all') else identity

        self.main = nn.Sequential(
            ImpalaBlock(c_in, 16*width, norm_func=norm_func,act_cls=act_cls),
            ImpalaBlock(16*width, 32*width, norm_func=norm_func, act_cls=act_cls),
            ImpalaBlock(32*width, 32*width, norm_func=norm_func_last, act_cls=act_cls),
            act_cls()
        )
        self.pool = torch.nn.AdaptiveMaxPool2d((8, 8))
        self.flatten = nn.Flatten()
        self.action_value = nn.Sequential(
            nn.Linear(2048*width, 256),
            act_cls(),
            nn.Linear(256, n_actions),
        )
        self.state_value = nn.Sequential(
            nn.Linear(2048*width, 256),
            act_cls(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        x = self.main(self.norm(x))
        x = self.flatten(self.pool(x))
        action_value = self.action_value(x)
        if self.dueling:
            state_value = self.state_value(x)
            action_score_centered = action_value - action_value.mean(dim=-1, keepdim=True)
            return state_value + action_score_centered
        else:
            return action_value



class DQN_MLP(DQNBase, SB3Compat):
    def __init__(self, n_actions, c_in, act_cls=nn.ReLU):
        super().__init__()
        if isinstance(n_actions, tuple): 
            n_actions = n_actions[0]
        if isinstance(c_in, tuple): 
            c_in = c_in[0]
        self.n_actions, self.c_in = n_actions, c_in
        self.network =  nn.Sequential(
            nn.Linear(c_in, 120),
            act_cls(),
            nn.Linear(120, 84),
            act_cls(),
            nn.Linear(84, n_actions),
        )

    def forward(self, x:torch.Tensor):
        return self.network(x)