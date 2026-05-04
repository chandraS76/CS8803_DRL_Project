"""Headless evaluation harness for soccer_twos agents.

Usage:
    python eval.py --agent_a GROUP71_Agent --agent_b example_player_agent \
                   --n_matches 10 [--render] [--output results.json] [--seed 42]
"""
import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import soccer_twos
from soccer_twos import EnvType


OUR_TEAM = (0, 1)
OPP_TEAM = (2, 3)


def load_agent(module_name: str, env):
    """Import an agent module and instantiate its AgentInterface class.

    The module is expected to export a single AgentInterface subclass from
    its top-level __init__.py (matching the convention of example_player_agent
    and example_team_agent).
    """
    sys.path.insert(0, str(Path.cwd()))
    module = importlib.import_module(module_name)
    # Find the AgentInterface subclass exported by the module.
    from soccer_twos import AgentInterface
    for name in dir(module):
        obj = getattr(module, name)
        if isinstance(obj, type) and issubclass(obj, AgentInterface) and obj is not AgentInterface:
            return obj(env)
    raise ValueError(f"No AgentInterface subclass found in module '{module_name}'")


def run_matches(agent_a_name: str, agent_b_name: str, n_matches: int,
                render: bool = False, seed: Optional[int] = None,
                base_port: Optional[int] = None) -> dict:
    """Run n_matches of agent_a (team 0) vs agent_b (team 1).

    Returns a dict with per-match results and aggregated counts.
    """
    make_kwargs = dict(variation=EnvType.multiagent_player, render=render)
    if base_port is not None:
        make_kwargs["base_port"] = base_port
    env = soccer_twos.make(**make_kwargs)
    agent_a = load_agent(agent_a_name, env)
    agent_b = load_agent(agent_b_name, env)

    results = []
    a_wins = b_wins = draws = 0

    for match_idx in range(n_matches):
        obs = env.reset()
        team_a_goals = 0
        team_b_goals = 0
        steps = 0
        done_flag = False

        while not done_flag:
            obs_a = {pid: obs[pid] for pid in OUR_TEAM if pid in obs}
            obs_b = {pid: obs[pid] for pid in OPP_TEAM if pid in obs}

            action_a = agent_a.act(obs_a) if obs_a else {}
            action_b_raw = agent_b.act({0: obs[2], 1: obs[3]}) if obs_b else {}
            # Agent B uses its own internal team ids {0,1}; remap to {2,3}.
            action_b = {2: action_b_raw.get(0), 3: action_b_raw.get(1)}

            actions = {**action_a, **action_b}
            obs, rewards, dones, info = env.step(actions)
            team_a_goals += sum(rewards.get(pid, 0) for pid in OUR_TEAM if rewards.get(pid, 0) > 0)
            team_b_goals += sum(rewards.get(pid, 0) for pid in OPP_TEAM if rewards.get(pid, 0) > 0)
            steps += 1
            done_flag = max(dones.values()) if isinstance(dones, dict) else dones

        # Classify match outcome based on goal differential.
        if team_a_goals > team_b_goals:
            winner = "A"; a_wins += 1
        elif team_b_goals > team_a_goals:
            winner = "B"; b_wins += 1
        else:
            winner = "draw"; draws += 1

        results.append({
            "match": match_idx,
            "team_a_goals": team_a_goals,
            "team_b_goals": team_b_goals,
            "winner": winner,
            "steps": steps,
        })

    env.close()
    return {
        "agent_a": agent_a_name,
        "agent_b": agent_b_name,
        "n_matches": n_matches,
        "a_wins": a_wins,
        "b_wins": b_wins,
        "draws": draws,
        "a_win_rate": a_wins / n_matches,
        "per_match": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_a", required=True, help="Python module path for team A")
    parser.add_argument("--agent_b", required=True, help="Python module path for team B")
    parser.add_argument("--n_matches", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--base_port", type=int, default=None,
                        help="Unity communication base port (defaults to soccer_twos default 50039)")
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    result = run_matches(args.agent_a, args.agent_b, args.n_matches,
                         render=args.render, seed=args.seed,
                         base_port=args.base_port)

    print(f"{args.agent_a} vs {args.agent_b} over {args.n_matches} matches:")
    print(f"  A wins: {result['a_wins']}")
    print(f"  Draws:  {result['draws']}")
    print(f"  B wins: {result['b_wins']}")
    print(f"  A win rate: {result['a_win_rate']:.2f}")

    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2))
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
