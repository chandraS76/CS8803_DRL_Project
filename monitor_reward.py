#!/usr/bin/env python3
"""Monitor GROUP71_Agent training reward with a single live progress bar."""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
MAGENTA = "\033[35m"


def _latest_trial_dir(results_root: Path, experiment_prefix: str) -> Optional[Path]:
    pattern = str(results_root / f"{experiment_prefix}*" / "PPO_Soccer_*")
    trials = glob.glob(pattern)
    if not trials:
        return None
    return Path(max(trials, key=os.path.getmtime))


def _format_duration(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    total = int(seconds)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _progress_bar(current: int, total: int, width: int = 32, color: bool = True) -> str:
    if total <= 0:
        total = 1
    frac = max(0.0, min(1.0, current / total))
    filled = int(width * frac)
    if not color:
        return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"
    filled_part = f"{CYAN}{'█' * filled}{RESET}"
    empty_part = "░" * (width - filled)
    return f"[{filled_part}{empty_part}]"


def _reward_color(reward: float, color: bool) -> str:
    if not color:
        return ""
    if reward >= 0.2:
        return GREEN
    if reward >= -0.05:
        return YELLOW
    return RED


def _render_line(
    iteration: int,
    steps: int,
    reward: float,
    total_steps: int,
    eta: str,
    bar_width: int,
    use_color: bool,
) -> str:
    steps_remaining = max(0, total_steps - steps)
    pct = 100.0 * min(1.0, steps / max(1, total_steps))
    bar = _progress_bar(steps, total_steps, width=bar_width, color=use_color)
    reward_prefix = _reward_color(reward, use_color)
    reward_text = f"{reward_prefix}{reward: .4f}{RESET}" if use_color else f"{reward: .4f}"

    if use_color:
        return (
            f"{MAGENTA}iter{RESET}={iteration:4d} "
            f"{CYAN}step{RESET}={steps:9d} "
            f"{BOLD}reward{RESET}={reward_text} "
            f"{bar} "
            f"{YELLOW}{pct:6.2f}%{RESET} "
            f"left={steps_remaining:9d} "
            f"eta={eta}"
        )
    return (
        f"iter={iteration:4d} step={steps:9d} reward={reward: .4f} "
        f"{bar} {pct:6.2f}% left={steps_remaining:9d} eta={eta}"
    )


def _clear_and_write(line: str) -> None:
    # Clear current line and rewrite in-place.
    sys.stdout.write("\r" + line)
    sys.stdout.flush()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_root",
        type=Path,
        default=Path("/home/srisiddarthc/Desktop/Archive/ray_results"),
        help="Path to ray_results directory.",
    )
    parser.add_argument(
        "--experiment_prefix",
        type=str,
        default="GROUP71_Agent_seed",
        help="Experiment directory prefix inside ray_results.",
    )
    parser.add_argument(
        "--total_steps",
        type=int,
        default=10_000_000,
        help="Target total timesteps for the run.",
    )
    parser.add_argument(
        "--poll_interval",
        type=float,
        default=2.0,
        help="Polling interval (seconds).",
    )
    parser.add_argument(
        "--bar_width",
        type=int,
        default=32,
        help="Progress bar width in characters.",
    )
    args = parser.parse_args()

    use_color = sys.stdout.isatty()

    print("Waiting for latest GROUP71_Agent trial...")
    trial_dir = None
    while trial_dir is None:
        trial_dir = _latest_trial_dir(args.results_root, args.experiment_prefix)
        if trial_dir is None:
            time.sleep(args.poll_interval)

    result_path = trial_dir / "result.json"
    print(f"Monitoring: {result_path}")
    seen_lines = 0
    latest_row = None
    last_key = None
    is_tty = sys.stdout.isatty()
    while True:
        if not result_path.exists():
            time.sleep(args.poll_interval)
            continue

        try:
            with result_path.open("r", encoding="utf-8") as f:
                lines = f.readlines()
        except OSError:
            time.sleep(args.poll_interval)
            continue

        new_lines = lines[seen_lines:]
        seen_lines = len(lines)

        for line in new_lines:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            latest_row = row

        if latest_row is not None:
            iteration = int(latest_row.get("training_iteration", 0))
            steps = int(latest_row.get("timesteps_total", 0))
            reward = float(latest_row.get("episode_reward_mean", 0.0))
            key = (iteration, steps, round(reward, 6))

            eta = "--:--:--"
            elapsed_s = float(latest_row.get("time_total_s", 0.0))
            if elapsed_s > 0 and steps > 0:
                speed = steps / elapsed_s
                if speed > 0:
                    eta = _format_duration(max(0, args.total_steps - steps) / speed)

            # Avoid duplicate prints when no new training result has arrived.
            if key != last_key:
                line = _render_line(
                    iteration=iteration,
                    steps=steps,
                    reward=reward,
                    total_steps=args.total_steps,
                    eta=eta,
                    bar_width=args.bar_width,
                    use_color=use_color,
                )
                if is_tty:
                    _clear_and_write(line)
                else:
                    print(line)
                last_key = key

            if steps >= args.total_steps:
                if is_tty:
                    sys.stdout.write("\n")
                sys.stdout.write("Reached target total_steps. Exiting monitor.\n")
                sys.stdout.flush()
                return 0

        time.sleep(args.poll_interval)


if __name__ == "__main__":
    raise SystemExit(main())
