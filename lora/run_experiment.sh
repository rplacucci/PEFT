#!/usr/bin/env bash
set -euo pipefail

# Default config
CHECKPOINT="bert-base-uncased"
TASK="mnli"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --task)
            TASK="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# List of seeds
SEEDS=(
    0 
    1 
    2 
    3 
    4
)

# List of LoRA ranks
RANKS=(
    1
    2 
    4 
    8 
    16 
    32 
    64 
    128 
    256 
    512
)

# Loop over each seed and rank with (alpha=rank) and invoke accelerate launch
for RANK in "${RANKS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "========================================================================================================"
        echo " Running GLUE LoRA-based tuning on $TASK for checkpoint $CHECKPOINT with rank=$RANK "
        echo "========================================================================================================"
        accelerate launch -m lora.experiment --checkpoint "$CHECKPOINT" --task_name "$TASK" --rank "$RANK" --alpha "$RANK" --seed "$SEED"
    done
done

echo "Completed experiments for $CHECKPOINT with LoRA on $TASK!"