"""Train a vanilla PPO baseline (no reward shaping, no curriculum)."""
import argparse

import numpy as np
import ray
import torch
from ray import tune

from soccer_env import make_trainer_config
from soccer_env.trainer import make_env_creator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=10_000_000)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--name", type=str, default="PPO_baseline")
    parser.add_argument("--base_port", type=int, default=None,
                        help="Unity base port (defaults to soccer_twos default 50039)")
    parser.add_argument("--quick_test", action="store_true",
                        help="Use tiny batch/fragment sizes for a fast end-to-end training check.")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ray.init()

    tune.registry.register_env("Soccer", make_env_creator())
    temp_env_config = {"base_port": args.base_port} if args.base_port is not None else {}
    temp_env = make_env_creator()(temp_env_config)
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    temp_env.close()

    overrides = {"rollout_fragment_length": 200, "train_batch_size": 2000,
                 "sgd_minibatch_size": 512} if args.quick_test else None
    config = make_trainer_config(
        seed=args.seed,
        obs_space=obs_space,
        act_space=act_space,
        num_workers=args.workers,
        num_gpus=args.gpus,
        base_port=args.base_port,
        hyperparam_overrides=overrides,
    )

    tune.run(
        "PPO",
        name=f"{args.name}_seed{args.seed}",
        config=config,
        stop={"timesteps_total": args.steps},
        checkpoint_freq=100,
        checkpoint_at_end=True,
        local_dir="./ray_results",
    )


if __name__ == "__main__":
    main()
