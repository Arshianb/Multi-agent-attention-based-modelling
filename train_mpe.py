#!/usr/bin/env python
import sys
import os
import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch
from MPE_env import MPEEnv
from env_wrappers import *
from mpe_runner import MPERunner as Runner
"""Train script for MPEs."""

def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            env = MPEEnv(all_args)
            env.seed(all_args["seed"] + rank * 1000)
            return env
        return init_env
    if all_args["n_rollout_threads"] == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args["n_rollout_threads"])])


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            env = MPEEnv(all_args)
            env.seed(all_args["seed"] * 50000 + rank * 10000)
            return env
        return init_env
    if all_args["n_eval_rollout_threads"] == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args["n_eval_rollout_threads"])])


if __name__=="__main__":
    print("choose to use gpu...")
    all_args={  "n_training_threads": 16,
                "n_rollout_threads" : 128,
                "n_eval_rollout_threads":16,
                "seed":0,
                "episode_length":200,
                "num_agents":10,
                "num_landmarks":10,
                "use_eval":True,
                "env_name":"MPE",
                "user_name":"arshianb",
                "algorithm_name":"mat",
                "scenario_name":"simple spread",
                "experiment_name":"exp",
                "use_wandb":False,
                "use_obs_instead_of_state":False,
                "num_env_steps":int(10000000/2),
                "episode_length":25,
                "n_render_rollout_threads":1,
                "use_linear_lr_decay":True,
                "hidden_size":64,
                "recurrent_N":1,
                "save_interval":100,
                "eval_interval":25,
                "log_interval":5,
                "model_dir":None,
                "lr":7e-4,
                "opti_eps":1e-5,
                "weight_decay":0,
                "use_policy_active_masks":True,
                "n_block":1,
                "n_embd":64,
                "n_head":1,
                "encode_state":True,
                "share_actor":True,
                "dec_actor":True,
                "clip_param":0.05,
                "ppo_epoch":15,
                "num_mini_batch":1,
                "data_chunk_length":10,
                "value_loss_coef":1,
                "entropy_coef":0.01,
                "max_grad_norm":10,
                "huber_delta":10,
                "use_recurrent_policy":True,
                "use_naive_recurrent_policy":True,
                "use_max_grad_norm":True,
                "use_clipped_value_loss":True,
                "use_huber_loss":True,
                "use_valuenorm":True,
                "use_value_active_masks":True,
                "gamma":0.99,
                "gae_lambda":0.95,
                "use_gae":True,
                "use_popart":False,
                "use_proper_time_limits":False,
                "use_centralized_V":True,
                "save_gifs":True,
                "use_render":True,
                "render_episodes":5,
                "ifi":0.1
                
             }
    device = torch.device("cuda:0")
    torch.set_num_threads(all_args["n_training_threads"])
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    # print("choose to use cpu...")
    # device = torch.device("cpu")
    torch.set_num_threads(all_args["n_training_threads"])
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                0] + "/results") / all_args["env_name"] / all_args["scenario_name"] / "" / all_args["algorithm_name"]
    
    print(run_dir)
    if not run_dir.exists():
        os.makedirs(str(run_dir))
    
    torch.manual_seed(all_args["seed"])
    torch.cuda.manual_seed_all(all_args["seed"])
    np.random.seed(all_args["seed"])
    
    # wandb
    if all_args["use_wandb"]:
        run = wandb.init(config=all_args,
                         project=all_args["env_name"],
                         entity=all_args["user_name"],
                         notes=socket.gethostname(),
                         name=str(all_args["algorithm_name"]) + "_" +
                         str(all_args["experiment_name"]) +
                         "_seed" + str(all_args["seed"]),
                         group=all_args["scenario_name"],
                         dir=str(run_dir),
                         job_type="training",
                         reinit=True)
    else:
        if not run_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    setproctitle.setproctitle(str(all_args["algorithm_name"]) + "-" + \
        str(all_args["env_name"]) + "-" + str(all_args["experiment_name"]) + "@" + str(all_args["user_name"]))


    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args["use_eval"] else None
    num_agents = all_args["num_agents"]
    
    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }
    
    runner = Runner(config)
    runner.run()
    
    
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
        runner.writter.close()