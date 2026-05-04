"""Convert a Ray RLlib checkpoint into a model.pt for an agent module.

We unpickle the worker state directly to avoid re-instantiating the trainer
(which would need the env, which needs Unity). Weights are renamed from the
RLlib layout (_hidden_layers.*, _logits.*) to PolicyNet's (shared.*, logits.*),
and value-branch weights are dropped.

Usage:
    python pack_agent.py --checkpoint <ray_results path>/checkpoint-N \
                         --agent_dir GROUP71_Agent
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import torch


def extract_policy_weights(checkpoint_path: str, policy_name: str = "default") -> dict:
    """Load a Ray checkpoint pickle and return the policy's numpy state_dict."""
    with open(checkpoint_path, "rb") as f:
        data = pickle.load(f)
    worker = pickle.loads(data["worker"])
    return worker["state"][policy_name]["weights"]


def _map_rllib_to_policynet(rllib_state_dict: dict) -> dict:
    """Rename RLlib model keys to PolicyNet's layout.

    RLlib keys → PolicyNet keys:
      _hidden_layers.{i}._model.0.{param} → shared.{i*2}.{param}
      _logits._model.0.{param}            → logits.{param}

    Value-branch weights and anything else are dropped (policy-only inference).
    """
    mapped = {}
    for key, value in rllib_state_dict.items():
        tensor = torch.from_numpy(np.asarray(value)) if not isinstance(value, torch.Tensor) else value
        if key.startswith("_hidden_layers"):
            parts = key.split(".")
            layer_idx = int(parts[1])
            leaf = parts[-1]
            mapped[f"shared.{layer_idx * 2}.{leaf}"] = tensor
        elif key.startswith("_logits"):
            leaf = key.split(".")[-1]
            mapped[f"logits.{leaf}"] = tensor
    return mapped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to RLlib checkpoint file")
    parser.add_argument("--agent_dir", required=True, help="Target agent directory")
    parser.add_argument("--policy", default="default", help="RLlib policy name")
    args = parser.parse_args()

    target = Path(args.agent_dir) / "model.pt"
    print(f"Extracting weights from {args.checkpoint}")
    rllib_state = extract_policy_weights(args.checkpoint, args.policy)
    print(f"Found {len(rllib_state)} RLlib state_dict keys")
    mapped = _map_rllib_to_policynet(rllib_state)
    print(f"Mapped to {len(mapped)} PolicyNet state_dict keys")
    for k, v in mapped.items():
        print(f"  {k}: {tuple(v.shape)}")
    torch.save(mapped, str(target))
    print(f"Wrote {target}")


if __name__ == "__main__":
    main()
