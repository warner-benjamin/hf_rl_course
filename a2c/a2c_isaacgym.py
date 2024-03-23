# Fast DQN for Atari Gyms
# Based on the CleanRL DQN implementation: https://docs.cleanrl.dev/rl-algorithms/dqn/#dqn_ataripy
# Refactored to use PyTorch GPU Replay Memory and EnvPool to train faster

# Contains code from:
# CleanRL - MIT License - Copyright (c) 2019 CleanRL developers
# EnvPool - Apache License 2.0 - Copyright (c) 2022 Garena Online Private Limited
# Stable Baselines3 - MIT License - Copyright (c) 2019 Antonin Raffin

import isaacgym  # noqa
import isaacgymenvs

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
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_sched
from torch.cuda.amp import autocast, GradScaler

from stable_baselines3.a2c import A2C as SB3A2C, MlpPolicy

import wandb

## Standalone boilerplate before relative imports from https://stackoverflow.com/a/65780624
import sys
if __package__ is None:                  
    DIR = Path(__file__).resolve().parent
    sys.path.insert(0, str(DIR.parent))
    __package__ = DIR.name

from a2c.a2c_models import A2C
from util.env_wrappers import RecordEpisodeStatisticsTorch
from util.helpers import linear_schedule, num_train_steps, evaluate_sb3, get_optimizer, get_eval_env, num_cpus


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
    parser.add_argument("--fp16", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, will train using automatic mixed precision")
    parser.add_argument("--log-frequency", type=int, default=1000,
        help="how often to log training data")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="Ant",
        help="the id of the environment")
    parser.add_argument("--num-envs", type=int, default=64,
        help="number of environments")
    parser.add_argument("--total-timesteps", type=int, default=2_000_000,
        help="total timesteps of the experiments")
    parser.add_argument("--num-steps", type=int, default=8,
        help="total timesteps of the experiments")
    parser.add_argument("--value_weight", type=float, default=0.4,
        help="supported optimizers: Adam, AdamW, SGD, or RMSprop")
    parser.add_argument("--entropy_weight", type=float, default=0.,
        help="supported optimizers: Adam, AdamW, SGD, or RMSprop")
    parser.add_argument("--optimizer", type=str, default='Adam',
        help="supported optimizers: Adam, AdamW, SGD, or RMSprop")
    parser.add_argument("--learning-rate", type=float, default=0.00096,
        help="the learning rate of the optimizer")
    parser.add_argument("--weight-decay", type=float, default=0,
        help="the weight decay of the optimizer")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--target-network-frequency", type=int, default=1000,
        help="the number of steps it takes to update the target network. incremented in training loop by batch-size/samples-step")
    parser.add_argument("--batch-size", type=int, default=32,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--one-cycle", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to use 1cycle policy scheduler")
    parser.add_argument("--auto-eps", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if true, set the optimizers epsilon to 5e-3/bs")
    args, _ = parser.parse_known_args()
    # fmt: on
    return args


def tensor_queue(queue, add):
    return torch.cat((queue[add.shape[0]:], add))


def train(args, parse=False):
    if parse:
        args = parse_args()

    run_name = f"{args.env_id}_{args.exp_name}_{args.seed}_{int(time.time())}"
    if args.track:
        wandb.init(
            project=args.project_name,
            entity=args.entity,
            group=f'{args.group}_{args.env_id}',
            config=vars(args),
            name=run_name,
            monitor_gym=False,
            save_code=True
        )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # For mixed precision training using AMP
    scaler = GradScaler() if args.fp16 and torch.cuda.is_available() else None
    ac = autocast() if args.fp16 and torch.cuda.is_available() else nullcontext()

    # env setup
    envs = isaacgymenvs.make(
        seed=args.seed,
        task=args.env_id,
        num_envs=args.num_envs,
        sim_device='cuda:0',
        rl_device='cuda:0',
        graphics_device_id=0,
        multi_gpu=False,
        virtual_screen_capture=False,
        force_render=False,
        headless=True
    )

    envs.num_envs = args.num_envs
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space
    envs = RecordEpisodeStatisticsTorch(envs)
    # assert isinstance(envs.action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = A2C(envs.single_action_space.shape, envs.single_observation_space.shape).to(device)
    # dqn_eval = SB3A2C(MlpPolicy, eval_env, buffer_size=1)


    # set the optimizer
    eps = 0.005/args.batch_size if args.auto_eps else 1e-8
    optimizer = get_optimizer(args.optimizer, agent.parameters(), lr=args.learning_rate, wd=args.weight_decay, eps=eps)


    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape, dtype=torch.float, device=device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape, dtype=torch.float, device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float, device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float, device=device)
    values = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float, device=device)
    advantages = torch.zeros_like(rewards, device=device)


    train_steps, num_timesteps_phase, total_train_time, loss_count = 0, 0, 0, 0
    smooth_loss, smooth_q, beta = torch.tensor(0., device=device), torch.tensor(0., device=device), torch.tensor(0.98, device=device)
    reward_queue, ep_len_queue = torch.zeros(100, device=device), torch.zeros(100, device=device)

    log_frequency = args.num_envs*int(args.log_frequency/args.num_envs) if int(args.log_frequency/args.num_envs) > 0 else args.num_envs

    if args.one_cycle:
        _, steps = num_train_steps(0, args.num_envs, args.samples_step, args.batch_size)
        training_steps = steps*int(1+args.total_timesteps/args.num_envs)
        scheduler = lr_sched.OneCycleLR(optimizer, max_lr=args.learning_rate, steps_per_epoch=training_steps, pct_start=args.exploration_fraction, epochs=1)
    else: 
        scheduler = None

    # TRY NOT TO MODIFY: start the game
    next_obs = envs.reset()
    next_done = torch.zeros(args.num_envs, dtype=torch.float, device=device)
    start_time = time.time()

    # Take an "extra" step as the model trains on one less step than the env steps
    for global_step in range(0, args.total_timesteps+args.num_envs, args.num_envs):

        agent.eval()
        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad(), ac:
                action, _, _, value = agent(next_obs)
            values[step] = value.flatten()
            actions[step] = action

            next_obs, rewards[step], next_done, infos = envs.step(action)

            if (0 <= step <= 2) and next_done.any():
                idxs = next_done.nonzero().squeeze(1)
                tensor_queue(reward_queue, infos['r'][idxs])
                tensor_queue(ep_len_queue, infos['l'][idxs])

        with torch.no_grad(), ac:
            next_value = agent.get_value(next_obs).reshape(1, -1)
            returns = torch.zeros_like(rewards).to(device)
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    next_return = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    next_return = returns[t + 1]
                returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
            advantages = returns - values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)

        # ALGO LOGIC: training
        agent.train()
        idxs = torch.randperm(args.num_envs, device=device)
        train_start = time.time()
        for end_idxs in range(0, args.num_envs, args.batch_size):
            b_inds = idxs[end_idxs-args.batch_size:end_idxs]

            with ac:
                _, newlogprob, entropy, newvalue = agent(b_obs[b_inds], b_actions[b_inds])
                newvalue = newvalue.flatten()

                # Policy gradient loss
                policy_loss = -(b_advantages[b_inds] * newlogprob).mean()

                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(b_returns[b_inds], newvalue)

                # Entropy loss favor exploration
                entropy_loss = -torch.mean(entropy)

                loss = policy_loss + args.entropy_weight* entropy_loss + args.value_weight * value_loss

            # optimize the model
            if args.fp16:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            if scheduler is not None: 
                scheduler.step()
            optimizer.zero_grad()

            # record losses for plotting purposes
            smooth_loss = torch.lerp(loss.float(), smooth_loss, beta)
            loss_count += 1
            train_steps += args.batch_size

        train_finish = time.time()
        total_train_time += train_finish - train_start

        # record SB3 style rewards for plotting purposes
        if args.track and global_step % log_frequency == 0 and global_step > 0:
            time_elapsed = time.time() - start_time
            total_fps = int((global_step - num_timesteps_phase) / time_elapsed)
            wandb.log({"rollout/ep_rew_mean": reward_queue.mean(),
                       "rollout/ep_len_mean": ep_len_queue.mean(),
                       "train/loss": loss,
                       "train/smooth_loss": smooth_loss.cpu()/(1-0.98**loss_count),
                       "train/model_updates": loss_count,
                       "time/fps": total_fps,
                       "time/train_fps": int(train_steps / total_train_time)}, step=global_step)


if __name__ == "__main__":
    train(None, True)