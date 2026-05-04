"""PPO trainer factory.

Players 0 and 1 share the trained "default" PPO policy; players 2 and 3 use
a frozen RandomPolicy. Training scripts vary the env wrapper (env_wrapper_cls)
and callbacks (callbacks_cls); everything else is fixed in PPO_HYPERPARAMS.
"""
from typing import Callable, Optional, Type

import gym
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.examples.policy.random_policy import RandomPolicy
from soccer_twos import EnvType

from utils import create_rllib_env


PPO_HYPERPARAMS = {
    "lr": 3e-4,
    "gamma": 0.99,
    "lambda": 0.95,
    "rollout_fragment_length": 1000,
    "train_batch_size": 24000,
    "sgd_minibatch_size": 2048,
    "num_sgd_iter": 10,
    "clip_param": 0.2,
    "entropy_coeff": 0.01,
    "model": {
        "vf_share_layers": True,
        "fcnet_hiddens": [512, 512],
        "fcnet_activation": "relu",
    },
}


OUR_TEAM_IDS = (0, 1)


def policy_mapping_fn(agent_id, *args, **kwargs) -> str:
    """Route our team to the trained policy, opponents to the random policy."""
    return "default" if agent_id in OUR_TEAM_IDS else "random_opponent"


def make_env_creator(env_wrapper_cls: Optional[Type[gym.Wrapper]] = None) -> Callable:
    """Return an env creator function that Ray Tune can register."""
    def _creator(env_config: dict = {}):
        if hasattr(env_config, "worker_index"):
            env_config["worker_id"] = (
                env_config.worker_index * env_config.get("num_envs_per_worker", 1)
                + env_config.vector_index
            )
        raw_env = create_rllib_env(env_config)
        if env_wrapper_cls is None:
            return raw_env
        return env_wrapper_cls(raw_env)
    return _creator


def make_trainer_config(
    seed: int,
    obs_space,
    act_space,
    num_workers: int = 8,
    num_envs_per_worker: int = 3,
    num_gpus: int = 1,
    callbacks_cls: Type[DefaultCallbacks] = DefaultCallbacks,
    env_wrapper_cls: Optional[Type[gym.Wrapper]] = None,
    base_port: Optional[int] = None,
    hyperparam_overrides: Optional[dict] = None,
) -> dict:
    """Build the Ray Tune PPO config shared by all three training scripts.

    obs_space / act_space: per-player spaces from a temp env instantiated by
        the caller (so we don't import Unity here).
    hyperparam_overrides: optional dict to override keys in PPO_HYPERPARAMS
        (e.g. {"train_batch_size": 2000} for fast quick-check runs).
    """
    env_config = {
        "num_envs_per_worker": num_envs_per_worker,
        "variation": EnvType.multiagent_player,
        # multiagent=True by default → create_rllib_env wraps in RLLibWrapper
    }
    if base_port is not None:
        env_config["base_port"] = base_port

    hyperparams = dict(PPO_HYPERPARAMS)
    if hyperparam_overrides:
        hyperparams.update(hyperparam_overrides)

    return {
        "seed": seed,
        "num_gpus": num_gpus,
        "num_workers": num_workers,
        "num_envs_per_worker": num_envs_per_worker,
        "log_level": "INFO",
        "framework": "torch",
        "callbacks": callbacks_cls,
        "env": "Soccer",
        "env_config": env_config,
        "batch_mode": "complete_episodes",
        # Ray 1.13 env checker uses np.bool (removed in numpy 1.24+); disable.
        "disable_env_checking": True,
        "multiagent": {
            "policies": {
                "default":         (None,         obs_space, act_space, {}),
                "random_opponent": (RandomPolicy, obs_space, act_space, {}),
            },
            "policy_mapping_fn": policy_mapping_fn,
            "policies_to_train": ["default"],
        },
        **hyperparams,
    }
