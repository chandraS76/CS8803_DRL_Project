#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-quick}"
if [[ $# -gt 0 ]]; then
  shift
fi

BASE_ARGS=(
  --seed 0
  --gpus 0
  --base_port 40000
  --alpha 0.008
  --beta 0.003
  --eps 0.01
  --delta 0.001
  --zeta 0.003
  --eta 0.001
  --tackle_bonus 0.04
  --counter_bonus 0.004
  --open_pass_bonus 0.02
  --pass_after_tackle_bonus 0.03
  --agent_dir "${ROOT_DIR}/GROUP71_Agent"
  --checkpoint_subdir checkpoints
)

case "${MODE}" in
  quick)
    STEPS=10000
    WORKERS=2
    NAME="GROUP71_Agent_quick"
    CKPT_EVERY=1000
    LOG_FILE="${ROOT_DIR}/train_main_agent_quick.log"
    MODE_ARGS=(--quick_test)
    ;;
  full)
    STEPS=10000000
    WORKERS=24
    NAME="GROUP71_Agent"
    CKPT_EVERY=100000
    LOG_FILE="${ROOT_DIR}/train_main_agent.log"
    MODE_ARGS=()
    ;;
  *)
    echo "Usage: $0 [quick|full] [extra train_main_agent.py args...]"
    exit 1
    ;;
esac

CMD=(
  python -u "${ROOT_DIR}/train_main_agent.py"
  --steps "${STEPS}"
  --workers "${WORKERS}"
  --name "${NAME}"
  --checkpoint_every_steps "${CKPT_EVERY}"
  "${MODE_ARGS[@]}"
  "${BASE_ARGS[@]}"
  "$@"
)

echo "Running: ${CMD[*]}"
"${CMD[@]}" | tee "${LOG_FILE}"
