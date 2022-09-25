# Fast PPO for Atari Gyms
# Based on the CleanRL DQN implementation: https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_atari_envpoolpy

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
from functools import partial
from pathlib import Path

import gym
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

from stable_baselines3.ppo import PPO as SB3PPO, CnnPolicy

import envpool

import wandb

## Standalone boilerplate before relative imports from https://stackoverflow.com/a/65780624
import sys
if __package__ is None:                  
    DIR = Path(__file__).resolve().parent
    sys.path.insert(0, str(DIR.parent))
    __package__ = DIR.name

from ppo.ppo_models import AtariPPO
from util.env_wrappers import EnvPoolRecordEpisodeStats
from util.helpers import evaluate_sb3, get_optimizer, get_eval_env, num_cpus, lerp


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
    parser.add_argument("--project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--group", type=str, default="Atari_PPO",
        help="the wandb's run group")
    parser.add_argument("--entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--fp16", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, will train using automatic mixed precision")
    parser.add_argument("--log-frequency", type=int, default=1000,
        help="how often to log training data")
    parser.add_argument("--save-folder", type=str, default="models",
        help="where to save the final model")
    parser.add_argument("--num-threads", type=int, default=None,
        help="number of cpu threads to use for envs, defaults to num_cpus")
    parser.add_argument("--jit", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to torch.jit.script the model")
    parser.add_argument("--trace", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to torch.jit.trace the model")
    parser.add_argument("--channels-last", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to torch.jit.script the model")
    parser.add_argument("--non-blocking", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to torch.jit.script the model")

    # Algorithm specific arguments
    parser.add_argument("--model", type=str, default="PPO",
        help="the PPO model type. supports 'PPO' for Atari")
    parser.add_argument("--env-id", type=str, default="Pong-v5",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=10_000_000,
        help="total timesteps of the experiments")
    parser.add_argument("--optimizer", type=str, default='Adam',
        help="supported optimizers: Adam, AdamW, SGD, or RMSprop")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--weight-decay", type=float, default=0,
        help="the weight decay of the optimizer")
    parser.add_argument("--num-envs", type=int, default=16,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=128,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Use GAE for advantage computation")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=8,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=3,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    parser.add_argument("--eval", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="evaluate during training")    
    parser.add_argument("--eval-episodes", type=int, default=20,
        help="number of evaluation episides/environments")
    parser.add_argument("--final-eval-eps", type=int, default=100,
        help="number of final evaluation episides/environments")
    parser.add_argument("--eval-frequency", type=int, default=25_000,
        help="the frequency of evaluation")
    parser.add_argument("--one-cycle", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to use 1cycle policy scheduler")
    parser.add_argument("--auto-eps", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if true, set the optimizers epsilon to 5e-3/bs")
    args, _ = parser.parse_known_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args



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
            monitor_gym=True,
            save_code=True,
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
    envs = envpool.make(
        args.env_id,
        env_type="gym",
        num_envs=args.num_envs,
        episodic_life=True,
        reward_clip=True,
        seed=args.seed,
        num_threads=num_cpus() if args.num_threads is None else min(args.num_threads, num_cpus())
    )
    envs.num_envs = args.num_envs
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space
    envs = EnvPoolRecordEpisodeStats(envs)
    assert isinstance(envs.action_space, gym.spaces.Discrete), "only discrete action space is supported"

    # env eval setup
    eval_env = get_eval_env(args.env_id, "gym", args.eval_episodes, args.seed, num_threads=args.num_threads)
    ppo_eval = SB3PPO(CnnPolicy, eval_env)
    eval_env.close()

    if args.model == 'PPO':
        PPOModel = AtariPPO
    else:
        raise ValueError(f"Unsupported `model`: {args.model}")

    # setup network
    memory_format = torch.channels_last if args.channels_last else torch.contiguous_format
    agent = PPOModel(envs.single_action_space.n).to(device, memory_format=memory_format)
    if args.jit: 
        agent = torch.jit.script(agent)
    if args.trace:
        agent = torch.jit.trace(agent, torch.randn([args.batch_size, 4, 84, 84], device=args.device))

    eps = 0.005/args.batch_size if args.auto_eps else 1e-5
    optimizer = get_optimizer(args.optimizer, agent.parameters(), lr=args.learning_rate, wd=args.weight_decay, eps=eps)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    start_time, loss_count, fps, play_fps, train_fps, steps = 0, 0, 0, 0, 0, 0
    smooth_loss, smooth_pg_loss = torch.tensor(0., device=device), torch.tensor(0., device=device)
    smooth_v_loss, smooth_e_loss, beta = torch.tensor(0., device=device), torch.tensor(0., device=device), torch.tensor(0.98, device=device)
    reward_queue, ep_len_queue = deque(maxlen=100), deque(maxlen=100)

    log_frequency = args.num_envs*int(args.log_frequency/args.num_envs) if int(args.log_frequency/args.num_envs) > 0 else args.num_envs
    eval_frequency = args.num_envs*int(args.eval_frequency/args.num_envs) if int(args.eval_frequency/args.num_envs) > 0 else args.eval_frequency

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size

    for update in range(1, num_updates + 1):
        steps += 1
        start_time = time.time()
        agent.eval()
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        play_start = time.time()
        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad(), ac:
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, infos = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device, non_blocking=args.non_blocking).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device, non_blocking=args.non_blocking), torch.Tensor(done)

            if next_done.any():
                idxs = next_done.nonzero().squeeze(1)
                # Record rollout stats 
                if len(idxs) > 1:
                    reward_queue.extend(infos['r'][idxs])
                    ep_len_queue.extend(infos['l'][idxs])
                else:
                    reward_queue.append(infos['r'][idxs])
                    ep_len_queue.append(infos['l'][idxs])

            next_done = next_done.to(device, non_blocking=args.non_blocking)

        # bootstrap value if not done
        with torch.no_grad(), ac:
            next_value = agent.get_value(next_obs).reshape(1, -1)
            if args.gae:
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values
            else:
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

        play_time = time.time() - play_start

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        agent.train()
        train_start = time.time()
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                with ac:
                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                # optimize the model
                if args.fp16:
                    scaler.scale(loss).backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

                # record losses for plotting purposes
                smooth_loss    = torch.lerp(loss.float(), smooth_loss, beta)
                smooth_pg_loss = torch.lerp(pg_loss.float(), smooth_pg_loss, beta)
                smooth_v_loss  = torch.lerp(v_loss.float(), smooth_v_loss, beta)
                smooth_e_loss  = torch.lerp(entropy_loss.float(), smooth_e_loss, beta)
                loss_count += 1

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        train_time = time.time() - train_start
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        fps = lerp((args.num_envs*args.num_steps)/(time.time()-start_time), fps, 0.98)
        play_fps = lerp((args.num_steps*args.num_envs)/play_time, play_fps, 0.98)
        train_fps = lerp((args.update_epochs*args.batch_size)/train_time, train_fps, 0.98)

        # record losses, SB3 style rollout, and training fps
        if args.track and global_step % log_frequency == 0:
            log = {"train/loss": loss,
                   "train/smooth_loss": smooth_loss/(1-0.98**loss_count),
                   "train/policy_loss": pg_loss,
                   "train/smooth_policy_loss": smooth_pg_loss/(1-0.98**loss_count),
                   "train/value_loss": v_loss,
                   "train/smooth_value_loss": smooth_v_loss/(1-0.98**loss_count),
                   "train/entropy": entropy_loss,
                   "train/smooth_entropy": smooth_e_loss/(1-0.98**loss_count),
                   "train/approx_kl": approx_kl,
                   "train/model_updates": loss_count,
                   "train/clipfrac": np.mean(clipfracs),
                   "train/explained_variance": explained_var,
                   "time/fps": int(fps/(1-0.98**steps)),
                   "time/play_fps": int(play_fps/(1-0.98**steps)),
                   "time/train_fps": int(train_fps/(1-0.98**steps))}
            if len(ep_len_queue) > 0:
                log.update({"rollout/ep_rew_mean": np.array(reward_queue).mean(),
                            "rollout/ep_len_mean": np.array(ep_len_queue).mean()})
            wandb.log(log, step=global_step)

        # evaluate during training
        if args.eval and global_step % eval_frequency == 0 and loss_count > 0:
            ppo_eval.policy.eval()
            eval_env = get_eval_env(args.env_id, "gym", args.eval_episodes, args.seed, num_threads=args.num_threads)
            mean_reward, std_reward, mean_ep_length, std_ep_length, _ = evaluate_sb3(ppo_eval, eval_env, args.eval_episodes, args.track, global_step)
            print(f"Mean Reward: {mean_reward:>7.2f}  ± {std_reward:>7.2f}   Mean Ep Len: {mean_ep_length:>7.2f}  ± {std_ep_length:>7.2f}   Step: {global_step:>8}")
            eval_env.close()

    # final eval
    ppo_eval.policy.eval()
    eval_env.close()
    eval_env = get_eval_env(args.env_id, "gym", args.final_eval_eps, args.seed, num_threads=args.num_threads)
    mean_reward, std_reward, mean_ep_length, std_ep_length, _ = evaluate_sb3(ppo_eval, eval_env, args.final_eval_eps, args.track, global_step, prefix='final_')
    print(f'\nFinal Evaluation on {args.final_eval_eps} episodes:\n')
    print(f"Mean Reward: {mean_reward:>7.2f}  ± {std_reward:>7.2f}   Mean Ep Len: {mean_ep_length:>7.2f}  ± {std_ep_length:>7.2f}   Step: {global_step:>8}\n")

    path = Path(args.save_folder)
    path.mkdir(exist_ok=True)
    agent.save(path/run_name)
    envs.close()
    eval_env.close()
    if args.track:
        wandb.finish()

if __name__ == "__main__":
    train(None, True)