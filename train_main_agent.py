"""Train the GROUP71 reward-shaped PPO agent."""
import argparse
from pathlib import Path

import numpy as np
import ray
import torch
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks

from pack_agent import _map_rllib_to_policynet
from soccer_env import make_trainer_config
from soccer_env.rewards import ShapedSoccerEnv, ShapingConfig
from soccer_env.trainer import make_env_creator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=10_000_000)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.008)
    parser.add_argument("--beta", type=float, default=0.003)
    parser.add_argument("--eps", type=float, default=0.02)
    parser.add_argument("--delta", type=float, default=0.001,
                        help="Behind-ball positioning shaping weight.")
    parser.add_argument("--zeta", type=float, default=0.003,
                        help="Defensive clearance shaping weight (opponent possession).")
    parser.add_argument("--eta", type=float, default=0.001,
                        help="Defensive pressing shaping weight (opponent possession).")
    parser.add_argument("--tackle_bonus", type=float, default=0.04,
                        help="Event bonus when possession flips from opponent to our team.")
    parser.add_argument("--counter_bonus", type=float, default=0.004,
                        help="Bonus per +x ball progress during counter window after a tackle.")
    parser.add_argument("--open_pass_bonus", type=float, default=0.02,
                        help="Bonus for forward pass to an open teammate.")
    parser.add_argument("--pass_after_tackle_bonus", type=float, default=0.03,
                        help="Extra pass bonus when an open pass happens soon after a tackle.")
    parser.add_argument("--name", type=str, default="GROUP71_Agent")
    parser.add_argument("--base_port", type=int, default=None,
                        help="Unity base port (defaults to soccer_twos default 50039)")
    parser.add_argument("--checkpoint_every_steps", type=int, default=100_000,
                        help="Export one model_<k>.pt every N timesteps.")
    parser.add_argument("--agent_dir", type=str,
                        default="/home/srisiddarthc/Desktop/Archive/GROUP71_Agent",
                        help="Absolute path to the GROUP71_Agent module directory.")
    parser.add_argument("--checkpoint_subdir", type=str, default="checkpoints",
                        help="Subfolder created under agent_dir for periodic model_<k>.pt exports.")
    parser.add_argument("--quick_test", action="store_true",
                        help="Use tiny batch/fragment sizes for a fast end-to-end training check.")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ray.init()

    shaping_config = ShapingConfig(
        alpha=args.alpha,
        beta=args.beta,
        eps=args.eps,
        delta=args.delta,
        zeta=args.zeta,
        eta=args.eta,
        tackle_bonus=args.tackle_bonus,
        counter_bonus=args.counter_bonus,
        open_pass_bonus=args.open_pass_bonus,
        pass_after_tackle_bonus=args.pass_after_tackle_bonus,
    )

    if args.checkpoint_every_steps < 1:
        parser.error("--checkpoint_every_steps must be >= 1")

    agent_dir = Path(args.agent_dir).resolve()
    checkpoint_dir = (agent_dir / args.checkpoint_subdir).resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"[checkpoint-export] writing model checkpoints to {checkpoint_dir}")

    def wrapper_cls(env):
        return ShapedSoccerEnv(env, config=shaping_config)

    class PeriodicModelCheckpointCallback(DefaultCallbacks):
        def __init__(self, legacy_callbacks_dict=None):
            super().__init__(legacy_callbacks_dict=legacy_callbacks_dict)
            self.next_export_step = args.checkpoint_every_steps

        def on_train_result(self, *, trainer, result, **kwargs):
            timesteps_total = int(result.get("timesteps_total", 0))
            while timesteps_total >= self.next_export_step:
                policy_state = trainer.get_weights(["default"])["default"]
                mapped = _map_rllib_to_policynet(policy_state)
                out_path = checkpoint_dir / f"model_{self.next_export_step}.pt"
                torch.save(mapped, str(out_path))
                print(
                    f"[checkpoint-export] wrote {out_path} "
                    f"at timesteps_total={self.next_export_step}",
                    flush=True,
                )
                self.next_export_step += args.checkpoint_every_steps

    tune.registry.register_env("Soccer", make_env_creator(env_wrapper_cls=wrapper_cls))
    temp_env_config = {"base_port": args.base_port} if args.base_port is not None else {}
    temp_env = make_env_creator(env_wrapper_cls=wrapper_cls)(temp_env_config)
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
        callbacks_cls=PeriodicModelCheckpointCallback,
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
