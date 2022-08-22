# Fast DQN for Atari Gyms
# Based on the CleanRL DQN implementation: https://docs.cleanrl.dev/rl-algorithms/dqn/#dqn_ataripy
# Refactored to use PyTorch GPU Replay Memory and EnvPool to train faster

# Contains code from:
# CleanRL - MIT License - Copyright (c) 2019 CleanRL developers
# EnvPool - Apache License 2.0 - Copyright (c) 2022 Garena Online Private Limited

import argparse
import os
import random
import time
from distutils.util import strtobool
from contextlib import nullcontext
from collections import deque
from pathlib import Path

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecEnvWrapper, VecMonitor
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs, VecEnvStepReturn
from stable_baselines3.dqn import DQN, CnnPolicy
from stable_baselines3.common.policies import BasePolicy

import envpool
from envpool.python.protocol import EnvPool

import wandb

## Standalone boilerplate before relative imports from https://stackoverflow.com/a/65780624
import sys
if __package__ is None:                  
    DIR = Path(__file__).resolve().parent
    sys.path.insert(0, str(DIR.parent))
    __package__ = DIR.name

from util.atari_buffer import TorchAtariReplayBuffer


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=42,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--device", type=str, default="cuda",
        help="device to train on, by default uses cuda")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--project-name", type=str, default="Atari",
        help="the wandb's project name")
    parser.add_argument("--group", type=str, default="Atari_DQN",
        help="the wandb's run group")
    parser.add_argument("--entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")
    parser.add_argument("--fp16", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, will train using automatic mixed precision")
    parser.add_argument("--log-frequency", type=int, default=1000,
        help="how often to log training data")
    parser.add_argument("--save-folder", type=str, default="models",
        help="where to save the final model")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="SpaceInvaders-v5",
        help="the id of the environment")
    parser.add_argument("--num-envs", type=int, default=256,
        help="number of environments")
    parser.add_argument("--total-timesteps", type=int, default=10_000_000,
        help="total timesteps of the experiments")
    parser.add_argument("--optimizer", type=str, default='Adam',
        help="supported optimizers: Adam, AdamW, SGD, or RMSprop")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--weight-decay", type=float, default=0,
        help="the weight decay of the optimizer")
    parser.add_argument("--dropout", type=float, default=0.0,
        help="dropout for the DQN model")
    parser.add_argument("--buffer-size", type=int, default=100_000,
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--target-network-frequency", type=int, default=1000,
        help="the timesteps it takes to update the target network")
    parser.add_argument("--batch-size", type=int, default=32,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--start-e", type=float, default=1,
        help="the starting epsilon for exploration")
    parser.add_argument("--end-e", type=float, default=0.01,
        help="the ending epsilon for exploration")
    parser.add_argument("--exploration-fraction", type=float, default=0.10,
        help="the fraction of `total-timesteps` it takes from start-e to go end-e")
    parser.add_argument("--learning-starts", type=int, default=80_000,
        help="timestep to start learning")
    parser.add_argument("--train-frequency", type=int, default=4,
        help="the frequency of training")
    parser.add_argument("--eval-episodes", type=int, default=20,
        help="number of evaluation episides/environments")
    parser.add_argument("--eval-frequency", type=int, default=25_000,
        help="the frequency of evaluation")
    args = parser.parse_args()
    # fmt: on
    return args

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


# ALGO LOGIC: initialize agent here:
class QNetwork(BasePolicy):
    def __init__(self, env, drop):
        super().__init__(None, None)
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Dropout(p=drop),
            nn.Linear(512, env.single_action_space.n),
        )

    def forward(self, x:torch.Tensor):
        with torch.no_grad():
             x = x / 255.0
        return self.network(x)

    ## Compatability with SB3 evaluate_policy
    def set_training_mode(self, mode: bool) -> None:
        self.train(mode)

    @torch.jit.unused
    def _predict(self, observation: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        q_values = self(observation)
        # Greedy action
        action = q_values.argmax(dim=1).reshape(-1)
        return action


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def num_train_steps(global_steps, num_envs, train_freq):
    if num_envs > train_freq:
        return True, int(num_envs/train_freq)
    else:
        return global_steps % train_freq == 0, 1


def eval(model, env, eval_eps, track, step=None):
    start_time = time.time()
    rewards, lengths = evaluate_policy(model, env, n_eval_episodes=eval_eps, return_episode_rewards=True)
    finish_time = time.time()
    mean_reward, std_reward = np.mean(rewards), np.std(rewards)
    mean_ep_length, std_ep_length = np.mean(lengths), np.std(lengths)
    if track:
        wandb.log({"eval/mean_reward": mean_reward,
                   "eval/stdev_reward": std_reward,
                   "eval/mean_ep_length": mean_ep_length,
                   "eval/stdev_ep_length": std_ep_length,
                   "time/eval_fps": int(np.sum(lengths) / (finish_time - start_time))}, step=step)
    print(f"Mean Reward: {mean_reward:>7.2f}  +/- {std_reward:>7.2f}   Mean Ep Len: {mean_ep_length:>7.2f}  +/- {std_ep_length:>7.2f}   Step: {global_step:>8}")


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}_{args.exp_name}_{args.seed}_{int(time.time())}"
    if args.track:
        wandb.init(
            project=args.project_name,
            entity=args.entity,
            group=f'{args.group}_{args.env_id}',
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True
        )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    nprng = np.random.default_rng(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # For mixed precision training using AMP
    scaler = GradScaler() if args.fp16 and torch.cuda.is_available() else None
    ac = autocast() if args.fp16 and torch.cuda.is_available() else nullcontext()

    # env setup
    envs = envpool.make(
        args.env_id,
        env_type="gym",
        num_envs=args.num_envs,
        episodic_life=True,
        reward_clip=True,
        seed=args.seed,
    )
    envs.num_envs = args.num_envs
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space
    envs = RecordEpisodeStatistics(envs)
    assert isinstance(envs.action_space, gym.spaces.Discrete), "only discrete action space is supported"

    # env eval setup
    eval_env = envpool.make(args.env_id, env_type="gym", num_envs=args.eval_episodes, seed=args.seed)
    eval_env.spec.id = args.env_id
    eval_env = VecAdapter(eval_env)
    eval_env = VecMonitor(eval_env)
    dqn_eval = DQN(CnnPolicy, eval_env, buffer_size=1)

    # setup network
    q_network = QNetwork(envs, args.dropout).to(device)
    target_network = QNetwork(envs, args.dropout).to(device)
    target_network.eval()
    target_network.load_state_dict(q_network.state_dict())

    # set the optimizer
    if args.optimizer == 'Adam':
        optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == 'AdamW':
        optimizer = optim.AdamW(q_network.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(q_network.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == 'RMSprop':
        optimizer = optim.RMSprop(q_network.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        raise ValueError(f'Invalid optimizer, {args.optimizer} must be one of Adam, AdamW, SGD, or RMSprop')

    dqn_eval.policy.q_net = q_network

    rb = TorchAtariReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device=device,
        optimize_memory_usage=False,
        handle_timeout_termination=True,
        n_envs=args.num_envs
    )

    num_timesteps, loss_count, target_count, target_updated = 0, 0, 0, 0
    smooth_loss, smooth_q, beta =  torch.tensor(0., device=device), torch.tensor(0., device=device), torch.tensor(0.98, device=device)
    reward_queue, ep_len_queue = deque(maxlen=100), deque(maxlen=100)

    log_frequency = args.num_envs*int(args.log_frequency/args.num_envs) if int(args.log_frequency/args.num_envs) > 0 else args.num_envs
    eval_frequency = args.num_envs*int(args.eval_frequency/args.num_envs) if int(args.eval_frequency/args.num_envs) > 0 else args.eval_frequency

    # TRY NOT TO MODIFY: start the game
    obs = torch.from_numpy(envs.reset()).to(device)
    for global_step in range(0, args.total_timesteps, args.num_envs):
        start_time = time.time()
        num_timesteps += args.num_envs
        # ALGO LOGIC: put action logic here
        if global_step > args.learning_starts:
            epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step-args.learning_starts)
        else:
            epsilon = args.start_e
        
        rand_actions = nprng.integers(0, envs.single_action_space.n, envs.num_envs)
        if epsilon < 1:
            q_network.eval()
            with torch.no_grad(), ac:
                logits = q_network(obs)
                actions = torch.argmax(logits, dim=1)

            actions = actions.cpu().numpy()
            idxs = np.where(nprng.random(args.num_envs) < epsilon)[0]
            if len(idxs) > 0: actions[idxs] = rand_actions[idxs]
        else:
            actions = rand_actions

        next_obs, rewards, dones, infos = envs.step(actions)

        # TRY NOT TO MODIFY: save data to reply buffer
        next_obs = torch.from_numpy(next_obs).to(device)
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions).to(device)
        rewards = torch.from_numpy(rewards).to(device)
        dones = torch.from_numpy(dones)
        real_next_obs = next_obs.clone()

        # TRY NOT TO MODIFY: handle `terminal_observation` since it doesn't exist in envpool
        if dones.any():
            idxs = dones.nonzero().squeeze(1)
            real_next_obs[idxs] = obs[idxs]
            if len(idxs) > 1:
                reward_queue.extend(infos['r'][idxs])
                ep_len_queue.extend(infos['l'][idxs])
            else:
                reward_queue.append(infos['r'][idxs])
                ep_len_queue.append(infos['l'][idxs])
        rb.add(obs, real_next_obs, actions, rewards, dones.to(device), infos)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if global_step % log_frequency == 0 and global_step > 0:
            if args.track:
                if len(reward_queue) > 0:
                    wandb.log({"rollout/ep_rew_mean": np.array(reward_queue).mean(),
                               "rollout/ep_len_mean": np.array(ep_len_queue).mean(),
                               "rollout/exploration_rate": epsilon}, step=global_step)
                else:
                    wandb.log({"rollout/exploration_rate": epsilon}, step=global_step)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        train, steps = num_train_steps(global_step, args.num_envs, args.train_frequency)
        if global_step > args.learning_starts and train:
            train_start = time.time()

            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            q_network.train()
            for r in range(steps):
                data = rb.sample(args.batch_size)
                with ac:
                    with torch.no_grad():
                        target_max, _ = target_network(data.next_observations).max(dim=1)
                        td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.float().flatten())
                    pred = q_network(data.observations).gather(1, data.actions).squeeze()
                    loss = F.mse_loss(td_target, pred)

                # optimize the model
                if args.fp16:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                optimizer.zero_grad()

                # TRY NOT TO MODIFY: record losses for plotting purposes
                smooth_loss = torch.lerp(loss.float(), smooth_loss, beta)
                smooth_q    = torch.lerp(pred.mean().float(), smooth_q, beta)
                loss_count += 1
                target_count += args.train_frequency

                # update the target network
                if target_count >= args.target_network_frequency:
                    target_network.load_state_dict(q_network.state_dict())
                    target_updated += 1
                    target_count = 0
                    if args.track: wandb.log({"train/target_updated": target_updated}, step=global_step)
            train_finish = time.time()

        if global_step % log_frequency == 0 and loss_count > 0:
            smooth_val = smooth_loss.cpu()/(1-0.98**loss_count)
            if args.track:
                wandb.log({"train/loss": loss,
                           "train/smooth_loss": smooth_val,
                           "train/q_values": pred.mean(),
                           "train/smooth_q_values": smooth_q.cpu()/(1-0.98**loss_count),
                           "train/model_updates": loss_count}, step=global_step)

        # TRY NOT TO MODIFY: record frames per second for plotting purposes
        finish_time = time.time() 
        if global_step % log_frequency == 0:
            if args.track:
                if global_step > args.learning_starts and train:
                    wandb.log({"time/fps": int(args.num_envs / (finish_time - start_time)),
                               "time/train_fps": int(args.batch_size*steps / (train_finish - train_start)),
                               "time/play_fps": int(args.num_envs / (train_start - start_time))}, step=global_step)
                else:
                    wandb.log({"time/fps": int(args.num_envs / (finish_time - start_time))}, step=global_step)

        if global_step % eval_frequency == 0 and loss_count > 0:
            eval(dqn_eval, eval_env, args.eval_episodes, args.track, global_step)

    # final eval
    eval(dqn_eval, eval_env, args.eval_episodes, args.track, global_step)

    path = Path(args.save_folder)
    path.mkdir(exist_ok=True)
    q_network.save(path/run_name)
    envs.close()
    eval_env.close()