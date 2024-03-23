import gym
import pybullet_envs

import os

from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env

import torch 
from torch import nn


env_id = "AntBulletEnv-v0"
# Create the env
env = gym.make(env_id)

# Get the state space and action space
s_size = env.observation_space.shape[0]
a_size = env.action_space

env = make_vec_env(env_id, n_envs=4)

# Adding this wrapper to normalize the observation and the reward
env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)

model = A2C(policy = "MlpPolicy",
            env = env,
            gae_lambda = 0.9,
            gamma = 0.99,
            learning_rate = 0.00096,
            max_grad_norm = 0.5,
            n_steps = 8,
            vf_coef = 0.4,
            ent_coef = 0.0,
            tensorboard_log = "./tensorboard",
            policy_kwargs=dict(
            log_std_init=-2, ortho_init=False),
            normalize_advantage=False,
            use_rms_prop= True,
            use_sde= True,
            verbose=1)

model.learn(2_000_000)