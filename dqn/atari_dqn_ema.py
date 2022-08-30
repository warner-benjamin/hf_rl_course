# Fast DQN for Atari Gyms with EMA Target Network updates
# Based on the CleanRL DQN implementation: https://docs.cleanrl.dev/rl-algorithms/dqn/#dqn_ataripy
# Refactored to use PyTorch GPU Replay Memory and EnvPool to train faster
# DQN Target Network updated via EMA from timm instead of copying weights every 1000 steps

# Contains code from:
# CleanRL - MIT License - Copyright (c) 2019 CleanRL developers
# EnvPool - Apache License 2.0 - Copyright (c) 2022 Garena Online Private Limited
# Stable Baselines3 - MIT License - Copyright (c) 2019 Antonin Raffin

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
import torch.optim.lr_scheduler as lr_sched
from torch.cuda.amp import autocast, GradScaler

from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.dqn import DQN, CnnPolicy
from stable_baselines3.common.policies import BasePolicy

import envpool

from timm.utils.model_ema import ModelEmaV2

import wandb

## Standalone boilerplate before relative imports from https://stackoverflow.com/a/65780624
import sys
if __package__ is None:                  
    DIR = Path(__file__).resolve().parent
    sys.path.insert(0, str(DIR.parent))
    __package__ = DIR.name

from util.atari_buffer import TorchAtariReplayBuffer
from util.env_wrappers import RecordEpisodeStatistics, VecAdapter
from util.helpers import linear_schedule, num_train_steps, evaluate_sb3, get_optimizer


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
    parser.add_argument("--ema-decay", type=float, default=0.998,
        help="EMA for the DQN target model")
    parser.add_argument("--buffer-size", type=int, default=100_000,
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
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
    parser.add_argument("--samples-step", type=int, default=8,
        help="samples to train on per env step. auto-adjusts to batch-size and num-envs. default of 8 is equivalent of train-frequency=4, batch-size=32, num-envs=1")
    parser.add_argument("--eval-episodes", type=int, default=20,
        help="number of evaluation episides/environments")
    parser.add_argument("--eval-frequency", type=int, default=25_000,
        help="the frequency of evaluation")
    parser.add_argument("--eval-ema", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to evaluate the EMA weights")
    parser.add_argument("--one-cycle", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to use 1cycle policy scheduler")
    args = parser.parse_args()
    # fmt: on
    return args


# ALGO LOGIC: initialize agent here:
class QNetwork(BasePolicy):
    def __init__(self, env, drop, normalize=True):
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
        self.normalize = True

    def forward(self, x:torch.Tensor):
        if self.normalize:
            with torch.no_grad():
                x = x / 255.0
        return self.network(x)

    ## Compatability with SB3 evaluate_policy
    def set_training_mode(self, mode: bool) -> None:
        self.train(mode)

    @torch.jit.unused
    def _predict(self, observation: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        q_values = self.forward(observation)
        # Greedy action
        action = q_values.argmax(dim=1).reshape(-1)
        return action



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
    target_network = ModelEmaV2(q_network, args.ema_decay, device)
    target_network.module.eval()

    # set the optimizer
    optimizer = get_optimizer(args.optimizer, q_network.parameters(), lr=args.learning_rate, wd=args.weight_decay)

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

    train_steps, num_timesteps_phase, eval_time, loss_count, target_count, target_updated = 0, 0, 0, 0, 0, 0
    smooth_loss, smooth_q, beta =  torch.tensor(0., device=device), torch.tensor(0., device=device), torch.tensor(0.98, device=device)
    reward_queue, ep_len_queue = deque(maxlen=100), deque(maxlen=100)
    started_learning = False

    log_frequency = args.num_envs*int(args.log_frequency/args.num_envs) if int(args.log_frequency/args.num_envs) > 0 else args.num_envs
    eval_frequency = args.num_envs*int(args.eval_frequency/args.num_envs) if int(args.eval_frequency/args.num_envs) > 0 else args.eval_frequency

    if args.one_cycle:
        _, steps = num_train_steps(0, args.num_envs, args.samples_step, args.batch_size)
        training_steps = steps*int(1+(args.total_timesteps-args.learning_starts)/args.num_envs+1)
        scheduler = lr_sched.OneCycleLR(optimizer, max_lr=args.learning_rate, steps_per_epoch=training_steps, pct_start=args.exploration_fraction, epochs=1)
    else: 
        scheduler = None

    # TRY NOT TO MODIFY: start the game
    obs = torch.from_numpy(envs.reset()).to(device)
    start_time = time.time()
    for global_step in range(0, args.total_timesteps, args.num_envs):

        # Don't decrease epsilon-greedy until learning starts
        if global_step > args.learning_starts:
            epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step-args.learning_starts)
        else:
            epsilon = args.start_e
        
        # ALGO LOGIC: put action logic here
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

        # play out the next step
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

        # TRY NOT TO MODIFY: record SB3 style rewards for plotting purposes
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
        train, steps = num_train_steps(global_step, args.num_envs, args.samples_step, args.batch_size)
        if global_step > args.learning_starts and train:
            # Reset time metrics for accurate training fps
            if not started_learning:
                started_learning = True
                total_train_time = 0
                start_time = time.time()
                num_timesteps_phase = global_step - args.num_envs
            train_start = time.time()

            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            q_network.train()
            for r in range(steps):
                data = rb.sample(args.batch_size)
                with ac:
                    with torch.no_grad():
                        target_max, _ = target_network.module(data.next_observations).max(dim=1)
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
                if scheduler is not None: 
                    scheduler.step()
                optimizer.zero_grad()

                # TRY NOT TO MODIFY: record losses for plotting purposes
                smooth_loss = torch.lerp(loss.float(), smooth_loss, beta)
                smooth_q    = torch.lerp(pred.mean().float(), smooth_q, beta)
                loss_count += 1
                train_steps += args.batch_size

                # update the target network
                target_network.update(q_network)

            train_finish = time.time()
            total_train_time += train_finish - train_start

        if global_step % log_frequency == 0 and loss_count > 0:
            smooth_val = smooth_loss.cpu()/(1-0.98**loss_count)
            if args.track:
                wandb.log({"train/loss": loss,
                           "train/smooth_loss": smooth_val,
                           "train/q_values": pred.mean(),
                           "train/smooth_q_values": smooth_q.cpu()/(1-0.98**loss_count),
                           "train/model_updates": loss_count}, step=global_step)

        # TRY NOT TO MODIFY: record frames per second for plotting purposes
        # TRY NOT TO MODIFY: record frames per second for plotting purposes
        finish_time = time.time() 
        if global_step % log_frequency == 0:
            if args.track:
                time_elapsed = time.time() - start_time - eval_time
                total_fps = int((global_step - num_timesteps_phase) / time_elapsed)
                if global_step > args.learning_starts and train:
                    wandb.log({"time/fps": total_fps,
                               "time/train_fps": int(train_steps / total_train_time),
                               "time/play_fps": int((global_step - num_timesteps_phase) / (time_elapsed - total_train_time))}, step=global_step)
                else:
                    wandb.log({"time/fps": total_fps}, step=global_step)

        if global_step % eval_frequency == 0 and loss_count > 0:
            eval_start = time.time()
            mean_reward, std_reward, mean_ep_length, std_ep_length, _ = evaluate_sb3(dqn_eval, eval_env, args.eval_episodes, args.track, global_step)
            print(f"    Mean Reward: {mean_reward:>7.2f}  +/- {std_reward:>7.2f}       Mean Ep Len: {mean_ep_length:>7.2f}  +/- {std_ep_length:>7.2f}   Step: {global_step:>8}")
            if args.eval_ema:
                dqn_eval.policy.q_net = target_network.module
                mean_reward, std_reward, mean_ep_length, std_ep_length, _ = evaluate_sb3(dqn_eval, eval_env, args.eval_episodes, args.track, global_step, prefix='ema', log_time=False)
                print(f"EMA Mean Reward: {mean_reward:>7.2f}  +/- {std_reward:>7.2f}   EMA Mean Ep Len: {mean_ep_length:>7.2f}  +/- {std_ep_length:>7.2f}   Step: {global_step:>8}")
                dqn_eval.policy.q_net = q_network
                target_network.module.eval()
            eval_time += time.time() - eval_start

    # final eval
    mean_reward, std_reward, mean_ep_length, std_ep_length, _ = evaluate_sb3(dqn_eval, eval_env, args.eval_episodes, args.track, global_step)
    print(f"    Mean Reward: {mean_reward:>7.2f}  +/- {std_reward:>7.2f}       Mean Ep Len: {mean_ep_length:>7.2f}  +/- {std_ep_length:>7.2f}   Step: {global_step:>8}")

    path = Path(args.save_folder)
    path.mkdir(exist_ok=True)
    q_network.save(path/run_name)

    if args.eval_ema:
        dqn_eval.policy.q_net = target_network.module
        mean_reward, std_reward, mean_ep_length, std_ep_length, _ = evaluate_sb3(dqn_eval, eval_env, args.eval_episodes, args.track, global_step, prefix='ema', log_time=False)
        print(f"EMA Mean Reward: {mean_reward:>7.2f}  +/- {std_reward:>7.2f}   EMA Mean Ep Len: {mean_ep_length:>7.2f}  +/- {std_ep_length:>7.2f}   Step: {global_step:>8}")
        target_network.module.save(path/f"{run_name}_ema")

    envs.close()
    eval_env.close()