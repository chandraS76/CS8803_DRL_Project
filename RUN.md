# RUN Guide (GROUP71_Agent)

## 1. Setup

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

## 2. Start Full Training

```bash
python -u train_main_agent.py \
  --seed 0 \
  --steps 10000000 \
  --gpus 0 \
  --workers 24 \
  --name GROUP71_Agent \
  --base_port 40000 \
  | tee train_main_agent.log
```

Preferred bash wrapper:

```bash
./scripts/train_group71.sh full
```

Quick check before full run:

```bash
./scripts/train_group71.sh quick
```

## 3. Monitor Reward and Progress

```bash
python monitor_reward.py --experiment_prefix GROUP71_Agent_seed --total_steps 10000000
```

## 4. Locate Latest RLlib Checkpoint

```bash
ls -dt ray_results/GROUP71_Agent_seed0/*/checkpoint_* | head -1
```

## 5. Export Final `model.pt`

```bash
python pack_agent.py \
  --checkpoint ray_results/GROUP71_Agent_seed0/<trial>/checkpoint_NNN/checkpoint-N \
  --agent_dir GROUP71_Agent
```

## 6. Evaluate

```bash
python eval.py --agent_a GROUP71_Agent --agent_b example_player_agent --n_matches 10 --base_port 40000
```
