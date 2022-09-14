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


class SB3Compat():
    "This class duplicates the minimum SB3 code required for compatability"

    @torch.jit.ignore
    def set_training_mode(self, mode: bool):
        self.train(mode)

    @torch.jit.ignore
    def _predict(self, observation: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        q_values = self.forward(observation)
        # Greedy action
        action = q_values.argmax(dim=1).reshape(-1)
        return action

    @torch.jit.ignore
    def _get_constructor_parameters(self):
        return dict(
            n_actions=self.n_actions,
            normalize=self.normalize,
        )

    @torch.jit.ignore
    def save(self, path: str):
        torch.save({"state_dict": self.state_dict(), "data": self._get_constructor_parameters()}, path)

    @classmethod
    @torch.jit.ignore
    def load(cls, path: str, device = None):
        saved_variables = torch.load(path, map_location=device)
        # Create policy object
        model = cls(**saved_variables["data"])  # pytype: disable=not-instantiable
        # Load weights
        model.load_state_dict(saved_variables["state_dict"])
        model.to(device)
        return model



class DQN(nn.Module, SB3Compat):
    def __init__(self, n_actions, dueling=False, c_in=4, normalize=True):
        super().__init__()
        self.n_actions, self.dueling = n_actions, dueling
        self.c_in, self.normalize = c_in, normalize
        self.network = nn.Sequential(
            nn.Conv2d(c_in, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.action_value = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )
        self.state_value = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, x:torch.Tensor):
        if self.normalize:
            with torch.no_grad():
                x = x / 255.0

        x = self.network(x)
        action_value = self.action_value(x)
        if self.dueling:
            state_value = self.state_value(x)

            action_score_centered = action_value - action_value.mean(dim=-1, keepdim=True)
            return state_value + action_score_centered
        else:
            return action_value



class ImpalaSmall(nn.Module, SB3Compat):
    """
    Implementation of the small variant of the IMPALA CNN introduced in Espeholt et al. (2018).
    """
    def __init__(self, n_actions, dueling=False, c_in=4, normalize=True):
        super().__init__()
        self.n_actions, self.dueling = n_actions, dueling
        self.c_in, self.normalize = c_in, normalize

        self.main = nn.Sequential(
            nn.Conv2d(c_in, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
        )
        self.pool = torch.nn.AdaptiveMaxPool2d((6, 6))
        self.flatten = nn.Flatten()
        self.action_value = nn.Sequential(
            nn.Linear(1152, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )
        self.state_value = nn.Sequential(
            nn.Linear(1152, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        if self.normalize:
            with torch.no_grad():
                x = x / 255.0
        x = self.main(x)
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
    def __init__(self, depth, norm_func):
        super().__init__()

        self.relu_0 = nn.ReLU()
        self.conv_0 = norm_func(nn.Conv2d(depth, depth, kernel_size=3, stride=1, padding=1))
        self.relu_1 = nn.ReLU()
        self.conv_1 = norm_func(nn.Conv2d(depth, depth, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        x_ = self.conv_0(self.relu_0(x))
        x_ = self.conv_1(self.relu_1(x_))
        return x+x_


class ImpalaBlock(nn.Module):
    """
    Three of these blocks are used in the large IMPALA CNN.
    """
    def __init__(self, depth_in, depth_out, norm_func):
        super().__init__()

        self.conv = nn.Conv2d(depth_in, depth_out, kernel_size=3, stride=1, padding=1)
        self.max_pool = nn.MaxPool2d(3, 2, padding=1)
        self.residual_0 = ImpalaResidual(depth_out, norm_func=norm_func)
        self.residual_1 = ImpalaResidual(depth_out, norm_func=norm_func)

    def forward(self, x):
        x = self.conv(x)
        x = self.max_pool(x)
        x = self.residual_0(x)
        x = self.residual_1(x)
        return x


class ImpalaLarge(nn.Module, SB3Compat):
    """
    Implementation of the large variant of the IMPALA CNN introduced in Espeholt et al. (2018).
    """
    def __init__(self, n_actions, dueling=False, width=1, spectral_norm=False, c_in=4, normalize=True):
        super().__init__()
        self.n_actions, self.dueling = n_actions, dueling
        self.c_in, self.width, self.spectral_norm, self.normalize = c_in, width, spectral_norm, normalize
        def identity(p): return p

        norm_func = torch.nn.utils.spectral_norm if (spectral_norm == 'all') else identity
        norm_func_last = torch.nn.utils.spectral_norm if (spectral_norm == 'last' or spectral_norm == 'all') else identity

        self.main = nn.Sequential(
            ImpalaBlock(c_in, 16*width, norm_func=norm_func),
            ImpalaBlock(16*width, 32*width, norm_func=norm_func),
            ImpalaBlock(32*width, 32*width, norm_func=norm_func_last),
            nn.ReLU()
        )
        self.pool = torch.nn.AdaptiveMaxPool2d((8, 8))
        self.flatten = nn.Flatten()
        self.action_value = nn.Sequential(
            nn.Linear(2048*width, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )
        self.state_value = nn.Sequential(
            nn.Linear(2048*width, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        if self.normalize:
            with torch.no_grad():
                x = x / 255.0
        x = self.main(x)
        x = self.flatten(self.pool(x))
        action_value = self.action_value(x)
        if self.dueling:
            state_value = self.state_value(x)
            action_score_centered = action_value - action_value.mean(dim=-1, keepdim=True)
            return state_value + action_score_centered
        else:
            return action_value



class DQN_MLP(nn.Module, SB3Compat):
    def __init__(self, n_actions, c_in):
        super().__init__()
        if isinstance(n_actions, tuple): 
            n_actions = n_actions[0]
        if isinstance(c_in, tuple): 
            c_in = c_in[0]
        self.n_actions, self.c_in = n_actions, c_in
        self.network =  nn.Sequential(
            nn.Linear(c_in, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, n_actions),
        )

    def forward(self, x:torch.Tensor):
        return self.network(x)