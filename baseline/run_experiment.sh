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

# List of top-k layers
TOPKS=(
    1
    2 
    3 
    4 
    5 
    6 
    7 
    8 
    9 
    10
    11
    12
)

# Loop over each seed and top-k and invoke accelerate launch
for TOPK in "${TOPKS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "========================================================================================================"
        echo " Running GLUE adapter-based tuning on $TASK for checkpoint $CHECKPOINT with top_k=$TOPK "
        echo "========================================================================================================"
        accelerate launch -m baseline.experiment --checkpoint "$CHECKPOINT" --task_name "$TASK" --top_k "$TOPK" --seed "$SEED"
    done
done

echo "Completed experiments for $CHECKPOINT with vanilla finetuning on $TASK!"