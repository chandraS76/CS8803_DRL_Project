"""PyTorch policy network matching the RLlib PPO model used during training.

Obs dim 336, action space MultiDiscrete([3, 3, 3]) flattened to 9 logits,
two hidden layers of 512 units (see soccer_env/trainer.py).
"""
import torch.nn as nn


PLAYER_OBS_DIM = 336
PLAYER_ACTION_LOGITS = 9  # MultiDiscrete([3, 3, 3]) flattened


class PolicyNet(nn.Module):
    def __init__(self, state_size: int = PLAYER_OBS_DIM,
                 num_action_logits: int = PLAYER_ACTION_LOGITS):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
        )
        self.logits = nn.Linear(512, num_action_logits)

    def forward(self, x):
        h = self.shared(x)
        return self.logits(h)
