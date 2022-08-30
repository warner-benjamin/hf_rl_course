
import time
import wandb
import numpy as np
import torch.optim as optim

from stable_baselines3.common.evaluation import evaluate_policy


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def num_train_steps(global_steps, num_envs, samples_step, batch_size):
    if num_envs > samples_step:
        return True, int(num_envs*samples_step/batch_size)
    else:
        return global_steps % int(batch_size/samples_step) == 0, 1


def evaluate_sb3(model, env, eval_eps, track, step=None, prefix='', log_time=True):
    start_time = time.time()
    rewards, lengths = evaluate_policy(model, env, n_eval_episodes=eval_eps, return_episode_rewards=True)
    finish_time = time.time()
    mean_reward, std_reward = np.mean(rewards), np.std(rewards)
    mean_ep_length, std_ep_length = np.mean(lengths), np.std(lengths)
    fps = int(np.sum(lengths) / (finish_time - start_time))
    if track:
        log = {f"eval/{prefix}mean_reward": mean_reward,
               f"eval/{prefix}stdev_reward": std_reward,
               f"eval/{prefix}mean_ep_length": mean_ep_length,
               f"eval/{prefix}stdev_ep_length": std_ep_length}
        if log_time: log.update({f"time/{prefix}eval_fps": fps})
        wandb.log(log, step=step)
        
    return mean_reward, std_reward, mean_ep_length, std_ep_length, fps

def get_optimizer(opt_name, parameters, lr, wd):
    if opt_name == 'Adam':
        optimizer = optim.Adam(parameters, lr=lr, weight_decay=wd)
    elif opt_name == 'AdamW':
        optimizer = optim.AdamW(parameters, lr=lr, weight_decay=wd)
    elif opt_name == 'SGD':
        optimizer = optim.SGD(parameters, lr=lr, weight_decay=wd)
    elif opt_name == 'RMSprop':
        optimizer = optim.RMSprop(parameters, lr=lr, weight_decay=wd)
    else:
        raise ValueError(f'Invalid optimizer, {opt_name} must be one of Adam, AdamW, SGD, or RMSprop')
    return optimizer