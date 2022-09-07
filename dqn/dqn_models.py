# DQN Models for Atari Gyms
# Base DQN modified from CleanRL
# Dueling DQN modified from CleanRL & labml.ai

# Contains code from:
# CleanRL - MIT License - Copyright (c) 2019 CleanRL developers
# labml.ai  Annotated Papers - MIT License - Copyright (c) 2020 Varuna Jayasiri
# Stable Baselines3 - MIT License - Copyright (c) 2019 Antonin Raffin


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