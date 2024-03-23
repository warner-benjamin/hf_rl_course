# Contains code from:
# Stable Baselines3 - MIT License - Copyright (c) 2019 Antonin Raffin

import torch


class SB3Compat():
    "This class duplicates the minimum SB3 code required for compatability"

    @torch.jit.ignore
    def set_training_mode(self, mode: bool):
        self.train(mode)

    @torch.jit.ignore
    def _predict(self, observation: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        return self.get_action(observation)

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
