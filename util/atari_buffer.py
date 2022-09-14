# GPU Atari Replay Buffer
# Based on the Stable Baselines3 replay buffer implementation: https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/buffers.py
# Refactored to efficently store Replay Memory in PyTorch Tensors on the GPU

# Contains code from:
# Stable Baselines3 - MIT License - Copyright (c) 2019 Antonin Raffin

import warnings

from typing import Any, Dict, Generator, List, Optional, Union
from stable_baselines3.common.buffers import BaseBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.vec_env import VecNormalize

import torch
import numpy as np

from gym import spaces

numpy_to_torch_dtype_dict = {
    bool          : torch.bool,
    np.bool_      : torch.bool,
    np.uint8      : torch.uint8,
    np.int8       : torch.int8,
    np.int16      : torch.int16,
    np.int32      : torch.int32,
    np.int64      : torch.int64,
    np.float16    : torch.float16,
    np.float32    : torch.float32,
    np.float64    : torch.float64,
    np.complex64  : torch.complex64,
    np.complex128 : torch.complex128,
}

def nptype_to_torch(dtype):
    return numpy_to_torch_dtype_dict[dtype]


class TorchAtariReplayBuffer(BaseBuffer):
    """
    Replay buffer used in off-policy algorithms like SAC/TD3.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device:
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
        Cannot be used in combination with handle_timeout_termination.
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[torch.device, str] = "cpu",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        normalize: bool = False
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)

        # Adjust buffer size
        self.buffer_size = max(buffer_size // n_envs, 1)

        # I think we don't normalize for envpool
        self.normalize = normalize

        # there is a bug if both optimize_memory_usage and handle_timeout_termination are true
        # see https://github.com/DLR-RM/stable-baselines3/issues/934
        if optimize_memory_usage and handle_timeout_termination:
            raise ValueError(
                "ReplayBuffer does not support optimize_memory_usage = True "
                "and handle_timeout_termination = True simultaneously."
            )
        self.optimize_memory_usage = optimize_memory_usage

        self.observations = torch.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype=nptype_to_torch(observation_space.dtype.type), device=device)

        if optimize_memory_usage:
            # `observations` contains also the next observation
            self.next_observations = None
        else:
            self.next_observations = torch.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype=nptype_to_torch(observation_space.dtype.type), device=device)
    
        self.actions = torch.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=nptype_to_torch(action_space.dtype.type), device=device)

        self.rewards = torch.zeros((self.buffer_size, self.n_envs), dtype=torch.float32, device=device)
        self.dones = torch.zeros((self.buffer_size, self.n_envs), dtype=torch.bool, device=device)
        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = torch.zeros((self.buffer_size, self.n_envs), dtype=torch.bool, device=device)


    def add(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        infos: List[Dict[str, Any]],
    ) -> None:

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs,) + self.obs_shape)
            next_obs = next_obs.reshape((self.n_envs,) + self.obs_shape)

        # Same, for actions
        if isinstance(self.action_space, spaces.Discrete):
            action = action.reshape((self.n_envs, self.action_dim))

        # Copy to avoid modification by reference
        self.observations[self.pos] = obs.detach().clone()

        if self.optimize_memory_usage:
            self.observations[(self.pos + 1) % self.buffer_size] = next_obs.detach().clone()
        else:
            self.next_observations[self.pos] = next_obs.detach().clone()

        self.actions[self.pos] = action.detach().clone()
        self.rewards[self.pos] = reward.detach().clone()
        self.dones[self.pos] = done.detach().clone()

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = torch.from_numpy(infos["TimeLimit.truncated"]).to(self.device)

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        """
        Sample elements from the replay buffer.
        Custom sampling when using memory efficient variant,
        as we should not sample the element with index `self.pos`
        See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        if not self.optimize_memory_usage:
            return super().sample(batch_size=batch_size, env=env)
        # Do not sample the element with index `self.pos` as the transitions is invalid
        # (we use only one array to store `obs` and `next_obs`)
        if self.full:
            batch_inds = (torch.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
        else:
            batch_inds = torch.randint(0, self.pos, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    def _get_samples(self, batch_inds: torch.Tensor, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = torch.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :]
            if self.normalize:
                next_obs = self._normalize_obs(next_obs, env)
        else:
            next_obs = self.next_observations[batch_inds, env_indices, :]
            if self.normalize:
                next_obs = self._normalize_obs(next_obs, env)

        obs = self.observations[batch_inds, env_indices, :]
        rewards = self.rewards[batch_inds, env_indices].view(-1, 1)
        data = (
            self._normalize_obs(obs, env) if self.normalize else obs,
            self.actions[batch_inds, env_indices, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (self.dones[batch_inds, env_indices].int() * (1 - self.timeouts[batch_inds, env_indices].int())).view(-1, 1),
            self._normalize_reward(rewards, env) if self.normalize else rewards,
        )
        return ReplayBufferSamples(*data)