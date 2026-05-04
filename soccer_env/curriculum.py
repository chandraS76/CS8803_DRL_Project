"""Curriculum callback for progressive task difficulty.

Loaded by train_curriculum.py. The callback advances tasks based on mean
episode reward crossing a per-task threshold, and resets each episode to
a sampled configuration from the current task's ranges.
"""
from typing import Any, Dict, List

import yaml
from ray.rllib.agents.callbacks import DefaultCallbacks

from utils import sample_player, sample_pos_vel


def load_tasks(yaml_path: str) -> List[Dict[str, Any]]:
    """Load and validate a curriculum YAML into a list of task dicts."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    tasks = data["tasks"]
    required_keys = {"name", "ranges", "advance_when_reward_mean_above"}
    for idx, t in enumerate(tasks):
        missing = required_keys - set(t.keys())
        if missing:
            raise ValueError(f"curriculum task {idx} missing keys: {missing}")
        if "ball" not in t["ranges"] or "players" not in t["ranges"]:
            raise ValueError(f"curriculum task {idx} missing ball/players ranges")
    return tasks


_CONFIG_FNS = {
    "none": lambda *_: None,
    "random_players": lambda env: env.set_policies(lambda *_: env.action_space.sample()),
}


class CurriculumCallback(DefaultCallbacks):
    """RLlib callback that advances through a curriculum of tasks."""

    def __init__(self, tasks: List[Dict[str, Any]], legacy_callbacks_dict=None):
        super().__init__(legacy_callbacks_dict=legacy_callbacks_dict)
        self.tasks = tasks
        self.current = 0

    def on_episode_start(self, *, worker, base_env, policies, episode, env_index, **kwargs) -> None:
        task = self.tasks[self.current]
        for env in base_env.get_unwrapped():
            fn_name = task.get("config_fn", "none")
            _CONFIG_FNS[fn_name](env)
            env.env_channel.set_parameters(
                ball_state=sample_pos_vel(task["ranges"]["ball"]),
                players_states={
                    int(player): sample_player(task["ranges"]["players"][player])
                    for player in task["ranges"]["players"]
                },
            )

    def on_train_result(self, **info) -> None:
        task = self.tasks[self.current]
        threshold = task["advance_when_reward_mean_above"]
        if info["result"]["episode_reward_mean"] > threshold:
            if self.current < len(self.tasks) - 1:
                self.current += 1
                print(f"[curriculum] advancing to task {self.current}: {self.tasks[self.current]['name']}")
