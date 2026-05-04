"""Shared environment wrappers, trainer config, and callbacks for SoccerTwos."""
from soccer_env.trainer import make_trainer_config, policy_mapping_fn, PPO_HYPERPARAMS

__all__ = ["make_trainer_config", "policy_mapping_fn", "PPO_HYPERPARAMS"]
