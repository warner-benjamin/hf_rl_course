# Contains code from:
# CleanRL - MIT License - Copyright (c) 2019 CleanRL developers
# EnvPool - Apache License 2.0 - Copyright (c) 2022 Garena Online Private Limited

import gym
import numpy as np

from stable_baselines3.common.vec_env import VecEnvWrapper
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs, VecEnvStepReturn

from envpool.python.protocol import EnvPool


# CleanRL wrapper for stats tracking
class RecordEpisodeStatistics(gym.Wrapper):
    def __init__(self, env, deque_size=100):
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.episode_returns = None
        self.episode_lengths = None
        # get if the env has lives
        self.has_lives = False
        env.reset()
        info = env.step(np.zeros(self.num_envs, dtype=int))[-1]
        if info["lives"].sum() > 0:
            self.has_lives = True

    def reset(self, **kwargs):
        observations = super().reset(**kwargs)
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.lives = np.zeros(self.num_envs, dtype=np.int32)
        self.returned_episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.returned_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return observations

    def step(self, action):
        observations, rewards, dones, infos = super().step(action)
        self.episode_returns += infos["reward"]
        self.episode_lengths += 1
        self.returned_episode_returns[:] = self.episode_returns
        self.returned_episode_lengths[:] = self.episode_lengths
        all_lives_exhausted = infos["lives"] == 0
        if self.has_lives:
            self.episode_returns *= 1 - all_lives_exhausted
            self.episode_lengths *= 1 - all_lives_exhausted
        else:
            self.episode_returns *= 1 - dones
            self.episode_lengths *= 1 - dones
        infos["r"] = self.returned_episode_returns
        infos["l"] = self.returned_episode_lengths
        return (
            observations,
            rewards,
            dones,
            infos,
        )

# EnvPool SB3 adaptor for evaluation
class VecAdapter(VecEnvWrapper):
  """
  Convert EnvPool object to a Stable-Baselines3 (SB3) VecEnv.
  :param venv: The envpool object.
  """

  def __init__(self, venv: EnvPool):
    # Retrieve the number of environments from the config
    venv.num_envs = venv.spec.config.num_envs
    super().__init__(venv=venv)

  def step_async(self, actions: np.ndarray) -> None:
    self.actions = actions

  def reset(self) -> VecEnvObs:
    return self.venv.reset()

  def seed(self, seed:int = None) -> None:
    # You can only seed EnvPool env by calling envpool.make()
    pass

  def step_wait(self) -> VecEnvStepReturn:
    obs, rewards, dones, info_dict = self.venv.step(self.actions)
    infos = []
    # Convert dict to list of dict
    # and add terminal observation
    for i in range(self.num_envs):
      infos.append(
        {
          key: info_dict[key][i]
          for key in info_dict.keys()
          if isinstance(info_dict[key], np.ndarray)
        }
      )
      if dones[i]:
        infos[i]["terminal_observation"] = obs[i]
        obs[i] = self.venv.reset(np.array([i]))

    return obs, rewards, dones, infos