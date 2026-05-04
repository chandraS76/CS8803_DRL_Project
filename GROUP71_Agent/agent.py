"""GROUP71 team agent: reward-shaped PPO policy at inference time.

Loads model.pt if present, otherwise falls back to random actions so the
module still imports cleanly. Each player runs the policy independently:
336-dim obs in, 9 logits out, split into 3 branches of 3 and argmax per branch.
"""
import os
from typing import Dict

import numpy as np
import torch
from soccer_twos import AgentInterface

from .model import PolicyNet


NUM_ACTION_BRANCHES = 3
NUM_CHOICES_PER_BRANCH = 3


class TeamAgent(AgentInterface):
    name = "GROUP71_Agent"

    def __init__(self, env):
        self.action_space = env.action_space
        self.model = PolicyNet()
        weights_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pt")
        if os.path.isfile(weights_path):
            self.model.load_state_dict(torch.load(weights_path, map_location="cpu"))
            self._has_weights = True
        else:
            print(f"[{self.name}] model.pt not found at {weights_path}; using random actions.")
            self._has_weights = False
        self.model.eval()

    def act(self, observation: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        if not self._has_weights:
            return {pid: self.action_space.sample() for pid in observation}
        actions: Dict[int, np.ndarray] = {}
        with torch.no_grad():
            for pid, obs in observation.items():
                state = torch.from_numpy(obs).float().unsqueeze(0)
                logits = self.model(state)  # (1, 9)
                branches = logits.view(NUM_ACTION_BRANCHES, NUM_CHOICES_PER_BRANCH)
                actions[pid] = torch.argmax(branches, dim=-1).numpy().astype(np.int64)
        return actions
