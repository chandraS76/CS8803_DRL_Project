"""Reward shaping wrapper for SoccerTwos.

Adds potential-based and event-based shaping terms on top of the sparse +1/-1
goal reward for our team (player_ids 0 and 1). See ShapingConfig for the
weights and ShapedSoccerEnv.step for the term definitions.
"""
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Tuple

import gym
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv

OUR_TEAM = (0, 1)
OPP_TEAM = (2, 3)


@dataclass
class ShapingConfig:
    alpha: float = 0.001  # distance-to-ball weight
    beta: float = 0.002   # ball-progress weight (gated on possession)
    eps: float = 0.05     # passing bonus per detected pass
    delta: float = 0.0    # behind-ball positioning weight
    zeta: float = 0.0     # defensive clearance weight (opp possession only)
    eta: float = 0.0      # defensive pressing weight (opp possession only)
    tackle_bonus: float = 0.0  # event bonus when possession flips opp -> our team
    counter_bonus: float = 0.0  # bonus per +x ball progress shortly after a tackle
    open_pass_bonus: float = 0.0  # bonus for pass to open forward teammate
    pass_after_tackle_bonus: float = 0.0  # extra pass bonus after recovering ball
    gamma: float = 0.99   # discount for potential-based shaping
    possession_radius: float = 1.5
    pass_window_steps: int = 10
    counter_window_steps: int = 20
    pass_after_tackle_window: int = 15
    open_pass_min_opp_dist: float = 2.5
    forward_pass_min_dx: float = 0.5
    opp_goal_position: Tuple[float, float] = (15.0, 0.0)
    our_goal_position: Tuple[float, float] = (-15.0, 0.0)
    behind_ball_margin: float = 0.0


class ShapedSoccerEnv(gym.Wrapper, MultiAgentEnv):
    def __init__(self, env: gym.Env, config: Optional[ShapingConfig] = None):
        gym.Wrapper.__init__(self, env)
        MultiAgentEnv.__init__(self)
        self.config = config or ShapingConfig()
        self._prev_phi_ball: Dict[int, float] = {0: 0.0, 1: 0.0}
        self._prev_phi_goal: float = 0.0
        self._prev_phi_behind: Dict[int, float] = {0: 0.0, 1: 0.0}
        self._prev_phi_defense: float = 0.0
        self._prev_phi_press: float = 0.0
        self._possession_history: Deque = deque(maxlen=self.config.pass_window_steps + 2)
        self._last_possession_team: Optional[str] = None
        self._last_tackle_step: Optional[int] = None
        self._prev_ball_x: Optional[float] = None
        self._step_idx: int = 0

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self._prev_phi_ball = {0: 0.0, 1: 0.0}
        self._prev_phi_goal = 0.0
        self._prev_phi_behind = {0: 0.0, 1: 0.0}
        self._prev_phi_defense = 0.0
        self._prev_phi_press = 0.0
        self._possession_history.clear()
        self._last_possession_team = None
        self._last_tackle_step = None
        self._prev_ball_x = None
        self._step_idx = 0
        return obs

    def _current_possessor(self, info: Dict) -> Optional[int]:
        """Which of our players (0 or 1) currently has possession, or None."""
        if not self._has_possession(info):
            return None
        ball_pos = np.asarray(info[0]["ball_info"]["position"])
        our_dists = {pid: float(np.linalg.norm(np.asarray(info[pid]["player_info"]["position"]) - ball_pos))
                     for pid in OUR_TEAM}
        return min(our_dists, key=our_dists.get)

    def _team_has_possession(self, info: Dict, team_ids, other_team_ids) -> bool:
        """Return True if `team_ids` currently has possession of the ball."""
        ball_pos = np.asarray(info[0]["ball_info"]["position"])
        team_dists = {
            pid: float(np.linalg.norm(np.asarray(info[pid]["player_info"]["position"]) - ball_pos))
            for pid in team_ids
        }
        other_dists = {
            pid: float(np.linalg.norm(np.asarray(info[pid]["player_info"]["position"]) - ball_pos))
            for pid in other_team_ids
        }
        min_team = min(team_dists.values())
        min_other = min(other_dists.values())
        return (min_team <= self.config.possession_radius) and (min_team < min_other)

    def _has_possession(self, info: Dict) -> bool:
        """Return True if our team currently has possession of the ball."""
        return self._team_has_possession(info, OUR_TEAM, OPP_TEAM)

    def _opponent_has_possession(self, info: Dict) -> bool:
        """Return True if opponent team currently has possession of the ball."""
        return self._team_has_possession(info, OPP_TEAM, OUR_TEAM)

    @staticmethod
    def _pos_xy(info: Dict, pid: int) -> np.ndarray:
        return np.asarray(info[pid]["player_info"]["position"])

    def _receiver_open(self, info: Dict, receiver_pid: int) -> bool:
        receiver_pos = self._pos_xy(info, receiver_pid)
        nearest_opp_dist = min(
            float(np.linalg.norm(self._pos_xy(info, opp) - receiver_pos))
            for opp in OPP_TEAM
        )
        return nearest_opp_dist >= self.config.open_pass_min_opp_dist

    def _is_forward_pass(self, info: Dict, passer_pid: int, receiver_pid: int) -> bool:
        passer_x = float(self._pos_xy(info, passer_pid)[0])
        receiver_x = float(self._pos_xy(info, receiver_pid)[0])
        return (receiver_x - passer_x) >= self.config.forward_pass_min_dx

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        if (
            self.config.alpha == 0
            and self.config.beta == 0
            and self.config.eps == 0
            and self.config.delta == 0
            and self.config.zeta == 0
            and self.config.eta == 0
            and self.config.tackle_bonus == 0
            and self.config.counter_bonus == 0
            and self.config.open_pass_bonus == 0
            and self.config.pass_after_tackle_bonus == 0
        ):
            return obs, reward, done, info

        shaped_reward = dict(reward)
        ball_pos = np.asarray(info[0]["ball_info"]["position"])
        ball_x = float(ball_pos[0])
        our_has_ball = self._has_possession(info)
        opp_has_ball = self._opponent_has_possession(info)
        possession_team = "our" if our_has_ball else ("opp" if opp_has_ball else None)

        # Term 1: distance-to-ball (potential-based).
        if self.config.alpha != 0:
            new_phi_ball = {}
            for pid in OUR_TEAM:
                agent_pos = np.asarray(info[pid]["player_info"]["position"])
                new_phi_ball[pid] = -self.config.alpha * float(np.linalg.norm(agent_pos - ball_pos))
                # F = gamma * phi(s') - phi(s)
                shaping = self.config.gamma * new_phi_ball[pid] - self._prev_phi_ball[pid]
                shaped_reward[pid] = shaped_reward[pid] + shaping
            self._prev_phi_ball = new_phi_ball

        # Term 2: ball-to-goal progress (potential-based, gated on possession).
        if self.config.beta != 0:
            has_possession = self._has_possession(info)
            opp_goal = np.asarray(self.config.opp_goal_position)
            if has_possession:
                new_phi_goal = -self.config.beta * float(np.linalg.norm(ball_pos - opp_goal))
            else:
                new_phi_goal = 0.0
            shaping_goal = self.config.gamma * new_phi_goal - self._prev_phi_goal
            for pid in OUR_TEAM:
                shaped_reward[pid] = shaped_reward[pid] + shaping_goal
            self._prev_phi_goal = new_phi_goal

        # Term 3: passing bonus (event-based).
        if (
            self.config.eps != 0
            or self.config.open_pass_bonus != 0
            or self.config.pass_after_tackle_bonus != 0
        ):
            possessor = self._current_possessor(info)
            bonus = 0.0
            pass_detected = False
            passer_pid: Optional[int] = None
            receiver_pid: Optional[int] = None
            if possessor is not None:
                # Look back through history for a DIFFERENT teammate possessor
                # within pass_window_steps.
                for past_possessor, past_step in reversed(self._possession_history):
                    steps_ago = self._step_idx - past_step
                    if steps_ago > self.config.pass_window_steps:
                        break
                    if past_possessor != possessor and past_possessor in OUR_TEAM:
                        pass_detected = True
                        passer_pid = past_possessor
                        receiver_pid = possessor
                        bonus += self.config.eps
                        break
                # Append current possessor to history (dedupe consecutive same-possessor).
                if not self._possession_history or self._possession_history[-1][0] != possessor:
                    self._possession_history.append((possessor, self._step_idx))
            if pass_detected and passer_pid is not None and receiver_pid is not None:
                receiver_open = self._receiver_open(info, receiver_pid)
                if receiver_open and self._is_forward_pass(info, passer_pid, receiver_pid):
                    bonus += self.config.open_pass_bonus
                if (
                    receiver_open
                    and self._last_tackle_step is not None
                    and (self._step_idx - self._last_tackle_step) <= self.config.pass_after_tackle_window
                ):
                    bonus += self.config.pass_after_tackle_bonus
            if bonus != 0:
                for pid in OUR_TEAM:
                    shaped_reward[pid] = shaped_reward[pid] + bonus

        # Term 4: stay behind the ball (potential-based).
        # For our team moving toward +x, "behind" means agent_x <= ball_x - margin.
        if self.config.delta != 0:
            new_phi_behind = {}
            for pid in OUR_TEAM:
                agent_x = float(np.asarray(info[pid]["player_info"]["position"])[0])
                ahead_amount = max(0.0, agent_x - (float(ball_pos[0]) - self.config.behind_ball_margin))
                new_phi_behind[pid] = -self.config.delta * ahead_amount
                shaping = self.config.gamma * new_phi_behind[pid] - self._prev_phi_behind[pid]
                shaped_reward[pid] = shaped_reward[pid] + shaping
            self._prev_phi_behind = new_phi_behind

        # Term 5a: defensive clearance (potential-based, opponent possession only).
        if self.config.zeta != 0:
            if opp_has_ball:
                our_goal = np.asarray(self.config.our_goal_position)
                new_phi_defense = self.config.zeta * float(np.linalg.norm(ball_pos - our_goal))
            else:
                new_phi_defense = 0.0
            shaping_defense = self.config.gamma * new_phi_defense - self._prev_phi_defense
            for pid in OUR_TEAM:
                shaped_reward[pid] = shaped_reward[pid] + shaping_defense
            self._prev_phi_defense = new_phi_defense

        # Term 5b: defensive pressing (potential-based, opponent possession only).
        if self.config.eta != 0:
            if opp_has_ball:
                nearest_our = min(
                    float(np.linalg.norm(np.asarray(info[pid]["player_info"]["position"]) - ball_pos))
                    for pid in OUR_TEAM
                )
                new_phi_press = -self.config.eta * nearest_our
            else:
                new_phi_press = 0.0
            shaping_press = self.config.gamma * new_phi_press - self._prev_phi_press
            for pid in OUR_TEAM:
                shaped_reward[pid] = shaped_reward[pid] + shaping_press
            self._prev_phi_press = new_phi_press

        # Term 6: tackle recovery event (opponent possession -> our possession).
        tackle_happened = self._last_possession_team == "opp" and possession_team == "our"
        if tackle_happened:
            self._last_tackle_step = self._step_idx
            if self.config.tackle_bonus != 0:
                for pid in OUR_TEAM:
                    shaped_reward[pid] = shaped_reward[pid] + self.config.tackle_bonus

        # Term 7: counterattack progress right after a tackle.
        if (
            self.config.counter_bonus != 0
            and self._last_tackle_step is not None
            and self._prev_ball_x is not None
            and (self._step_idx - self._last_tackle_step) <= self.config.counter_window_steps
            and our_has_ball
        ):
            forward_progress = max(0.0, ball_x - self._prev_ball_x)
            if forward_progress > 0:
                counter_gain = self.config.counter_bonus * forward_progress
                for pid in OUR_TEAM:
                    shaped_reward[pid] = shaped_reward[pid] + counter_gain

        self._last_possession_team = possession_team
        self._prev_ball_x = ball_x
        self._step_idx += 1
        return obs, shaped_reward, done, info
