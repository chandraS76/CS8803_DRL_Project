# CS 8803 DRL - Final Project (GROUP_71)

SoccerTwos competition agent for CS 8803 Deep Reinforcement Learning at Georgia Tech.
The submission module is `GROUP71_Agent/`.

## Authors

- Chandra Sekhar Reddy Edula (cedula3@gatech.edu)
- Rithik Appachi Senthilkumar (rsenthilkumar8@gatech.edu)
- Sri Siddarth Chakaravarthy P (sp313@gatech.edu)

Video demo: TBD.

---

## What this does

PPO (Ray RLlib) on the multi-agent SoccerTwos env, with a reward shaping wrapper
on top of the sparse +1/-1 goal reward. Our two players (ids 0, 1) share a
single PPO policy; the opponents (ids 2, 3) are a frozen RandomPolicy during
training.

Reward shaping is split into two parts:

1. Potential-based terms for positioning and progress (distance-to-ball,
   ball-to-goal when in possession, behind-ball support, defensive clearance,
   defensive pressing).
2. Event-based bonuses for tactical actions (passing, tackling, counter-attack,
   open forward passes, passes after a recovery).

The dense feedback gets PPO unstuck from the early all-zero-reward regime that
sparse-only goal rewards run into.

---

## Reward components

See [soccer_env/rewards.py](soccer_env/rewards.py) for the full implementation.

| Term | What it does |
| --- | --- |
| `alpha` | Distance-to-ball shaping. |
| `beta` | Ball progress toward opponent goal when our team has possession. |
| `eps` | Pass-detection bonus (possession handoff between teammates). |
| `delta` | Stay-behind-the-ball positioning. |
| `zeta` | Defensive clearance when opponent has possession. |
| `eta` | Defensive pressing when opponent has possession. |
| `tackle_bonus` | One-shot bonus on possession flip from opponent to us. |
| `counter_bonus` | Bonus per +x ball progress shortly after a tackle. |
| `open_pass_bonus` | Extra bonus for a forward pass to an open teammate. |
| `pass_after_tackle_bonus` | Extra pass bonus shortly after a recovery. |
| `gamma` | Discount used in the potential-based shaping `F = gamma * phi(s') - phi(s)`. |

Detection thresholds: `possession_radius`, `pass_window_steps`, `counter_window_steps`,
`pass_after_tackle_window`, `open_pass_min_opp_dist`, `forward_pass_min_dx`,
`behind_ball_margin`.

---

## Code map

| File | Role |
| --- | --- |
| [train_main_agent.py](train_main_agent.py) | Main training script (Ray Tune + PPO). |
| [soccer_env/rewards.py](soccer_env/rewards.py) | `ShapedSoccerEnv` wrapper (all shaping logic). |
| [soccer_env/trainer.py](soccer_env/trainer.py) | PPO config, policy mapping, env creator. |
| [GROUP71_Agent/agent.py](GROUP71_Agent/agent.py) | `TeamAgent.act(...)` inference entry-point. |
| [GROUP71_Agent/model.py](GROUP71_Agent/model.py) | `PolicyNet` (MLP, [512, 512] hidden, 9-logit head). |
| [scripts/train_group71.sh](scripts/train_group71.sh) | `quick` / `full` training wrapper. |
| [pack_agent.py](pack_agent.py) | Convert an RLlib checkpoint into `GROUP71_Agent/model.pt`. |
| [monitor_reward.py](monitor_reward.py) | Live progress bar over `ray_results/`. |
| [eval.py](eval.py) | Headless head-to-head match runner. |

---

## Setup

```bash
conda create --name soccertwos python=3.8 -y
conda activate soccertwos
pip install pip==23.3.2 setuptools==65.5.0 wheel==0.38.4
pip cache purge
pip install -r requirements.txt
pip install protobuf==3.20.3 pydantic==1.10.13
chmod +x scripts/post_install_patches.sh
./scripts/post_install_patches.sh
```

---

## Training

Quick run (small batch, ~10k steps):

```bash
./scripts/train_group71.sh quick
```

Full run:

```bash
./scripts/train_group71.sh full
```

Or run the python command directly:

```bash
python -u train_main_agent.py \
  --seed 0 \
  --steps 10000000 \
  --gpus 0 \
  --workers 24 \
  --name GROUP71_Agent \
  --base_port 40000 | tee train_main_agent.log
```

Outputs:

- RLlib results in `ray_results/GROUP71_Agent_seed0/...`
- Periodic policy snapshots in `GROUP71_Agent/checkpoints/model_<step>.pt`.

---

## Monitor training

```bash
python monitor_reward.py --experiment_prefix GROUP71_Agent_seed --total_steps 10000000
```

---

## Pack the final model

`TeamAgent` loads `GROUP71_Agent/model.pt` at inference time. Without that
file it falls back to random actions, so you have to pack a checkpoint after
training:

```bash
# Find the latest checkpoint
ls -dt ray_results/GROUP71_Agent_seed0/*/checkpoint_* | head -1

# Convert it into GROUP71_Agent/model.pt
python pack_agent.py \
  --checkpoint ray_results/GROUP71_Agent_seed0/<trial>/checkpoint_NNN/checkpoint-N \
  --agent_dir GROUP71_Agent
```

The periodic snapshots in `GROUP71_Agent/checkpoints/` are already in
`PolicyNet`-compatible format, so you can also just copy one of those to
`GROUP71_Agent/model.pt` instead of running `pack_agent.py`.

---

## Evaluate

```bash
python eval.py --agent_a GROUP71_Agent --agent_b example_player_agent --n_matches 10 --base_port 40000
```

---

## Layout

```
Agent_71/
├── GROUP71_Agent/           Submission package (AgentInterface + PolicyNet + model.pt)
│   ├── agent.py
│   ├── model.py
│   ├── checkpoints/         Periodic model_<step>.pt exports
│   └── requirements.txt
├── soccer_env/              Training utilities
│   ├── rewards.py           ShapedSoccerEnv
│   ├── trainer.py           PPO config + policy mapping + env creator
│   └── curriculum.py
├── scripts/
│   ├── train_group71.sh
│   └── post_install_patches.sh
├── train_main_agent.py
├── pack_agent.py
├── monitor_reward.py
├── eval.py
├── requirements.txt
├── RUN.md
└── README.md
```

---

## Apple Silicon notes

Two site-packages patches are needed to run `soccer_twos` on an arm64 Mac.
`scripts/post_install_patches.sh` applies both:

1. Unity binary path. `soccer_twos/package.py` sets `TRAINING_ENV_PATH` to a
   path inside the `.app` bundle. `mlagents_envs.env_utils.validate_environment_path`
   then strips `.app` from it and the resulting path no longer points at a
   real file. The patch rewrites the path to the bundle name without `.app`.
2. numpy compatibility. `mlagents_envs/rpc_utils.py` uses `np.bool`, which
   was removed in numpy 1.24. The patch replaces it with the builtin `bool`.

Both fixes touch site-packages, so re-run the script if you recreate the conda
env.
