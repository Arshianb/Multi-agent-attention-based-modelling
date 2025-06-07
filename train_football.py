import sys
import os
import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch

from football.football_env import FootballEnv
from football_runner import FootballRunner as Runner
from env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv


"""Train script for SMAC."""


def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args["env_name"] == "football":
                env_args = {"scenario": all_args["scenario"],
                            "n_agent": all_args["n_agent"],
                            "reward": "scoring"}

                env = FootballEnv(env_args=env_args)
            else:
                print("Can not support the " + all_args["env_name"] + " environment.")
                raise NotImplementedError
            env.seed(all_args["seed"] + rank * 1000)
            return env

        return init_env

    if all_args["n_rollout_threads"] == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args["n_rollout_threads"])])


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args["env_name"] == "football":
                env_args = {"scenario": all_args["scenario"],
                            "n_agent": all_args["n_agent"],
                            "reward": "scoring"}
                env = FootballEnv(env_args=env_args)
            else:
                print("Can not support the " + all_args["env_name"] + " environment.")
                raise NotImplementedError
            env.seed(all_args["seed"] * 50000 + rank * 10000)
            return env

        return init_env

    if all_args["eval_episodes"] == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args["eval_episodes"])])



def main(args):
    all_args={ 
            "env_name":"football",
            "scenario": "academy_counterattack_easy",
            "n_agent": 3,
            "algorithm_name":"mat",
            "experiment_name":"exp",
            "lr":5e-4,
            "entropy_coef":0.01,
            "max_grad_norm":0.5,
            "eval_episodes":32,
            "n_training_threads":16,
            "n_rollout_threads":20,
            "num_mini_batch":1,
            "episode_length": 200,
            "eval_interval":25,
            "num_env_steps":10000000,
            "ppo_epoch":10,
            "clip_param":0.05,
            "use_value_active_masks":True,
            "use_policy_active_masks":True,
            "use_eval":False,
            "add_local_obs":False,
            "add_move_state":False,
            "add_distance_state":False,
            "add_enemy_action_state":False,
            "add_agent_id":False,
            "add_visible_state":False,
            "add_xy_state":False,
            "use_state_agent":False,
            "use_mustalive":False,
            "add_center_xy":False,
            "use_naive_recurrent_policy":False,
            "use_recurrent_policy":False,
            "dec_actor":False,
            "share_actor":False,
            "cuda":True,
            "cuda_deterministic":True,
            "use_wandb":False,
            "user_name":"arshianb",
            "seed":1,
            "map_name":"",
        "use_obs_instead_of_state":False,
        "n_render_rollout_threads":1,
        "use_linear_lr_decay":True,
        "hidden_size":64,
        "recurrent_N":1,
        "save_interval":100,
        "log_interval":5,
        "model_dir":"/media/arshianb/New Volume/master thesis/code/results/football/academy_counterattack_easy/mat/exp/run2/models/transformer_2499.pt",
        "opti_eps":1e-5,
        "weight_decay":0,
        "n_block":1,
        "n_embd":64,
        "n_head":1,
        "encode_state":True,
        "data_chunk_length":10,
        "value_loss_coef":1,
        "entropy_coef":0.01,
        "max_grad_norm":10,
        "huber_delta":10,
        "use_max_grad_norm":True,
        "use_clipped_value_loss":True,
        "use_huber_loss":True,
        "use_valuenorm":True,
        "gamma":0.99,
        "gae_lambda":0.95,
        "use_gae":True,
        "use_popart":False,
        "use_proper_time_limits":False,
        "use_centralized_V":True,
        "save_gifs":True,
        "use_render":True,
        "render_episodes":5,
        "ifi":0.1,
        "n_eval_rollout_threads":1
    }
    print("mumu config: ", all_args)

    if all_args["algorithm_name"] == "rmappo":
        all_args["use_recurrent_policy"] = True
        assert (all_args["use_recurrent_policy"] or all_args["use_naive_recurrent_policy"]), ("check recurrent policy!")
    elif all_args["algorithm_name"] == "mappo" or all_args["algorithm_name"] == "mat" or all_args["algorithm_name"] == "mat_dec":
        assert (all_args["use_recurrent_policy"] == False and all_args["use_naive_recurrent_policy"] == False), (
            "check recurrent policy!")
    else:
        raise NotImplementedError

    if all_args["algorithm_name"] == "mat_dec":
        all_args["dec_actor"] = True
        all_args["share_actor"] = True

    # cuda
    if all_args["cuda"] and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args["n_training_threads"])
        if all_args["cuda_deterministic"]:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args["n_training_threads"])

    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                       0] + "/results") / all_args["env_name"] / all_args["scenario"] / all_args["algorithm_name"] / all_args["experiment_name"]
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    if all_args["use_wandb"]:
        run = wandb.init(config=all_args,
                         project=all_args["env_name"],
                         entity=all_args["user_name"],
                         notes=socket.gethostname(),
                         name=str(all_args['algorithm_name']) + "_" +
                              str(all_args["experiment_name"]) +
                              "_seed" + str(all_args["seed"]),
                         group=all_args["map_name"],
                         dir=str(run_dir),
                         job_type="training",
                         reinit=True)
    else:
        if not run_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if
                             str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    setproctitle.setproctitle(
        str(all_args["algorithm_name"]) + "-" + str(all_args["env_name"]) + "-" + str(all_args["experiment_name"]) + "@" + str(
            all_args["user_name"]))

    # seed
    torch.manual_seed(all_args["seed"])
    torch.cuda.manual_seed_all(all_args["seed"])
    np.random.seed(all_args["seed"])

    # env
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args["use_eval"] else None
    num_agents = envs.n_agents

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    runner = Runner(config)
    # runner.eval(total_num_steps)
    runner.run()

    # post process
    envs.close()
    if all_args["use_eval"] and eval_envs is not envs:
        eval_envs.close()

    if all_args["use_wandb"]:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
        runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
