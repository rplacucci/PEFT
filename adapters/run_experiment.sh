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
SEEDS=(0 1 2 3 4)

# List of bottleneck sizes
BOTTLENECKS=(1 2 4 8 16 32 64 128 256 512)

# Loop over each seed and bottleneck size and invoke accelerate launch
for BOTTLENECK in "${BOTTLENECKS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "========================================================================================================"
        echo " Running GLUE adapter-based tuning on $TASK for checkpoint $CHECKPOINT with bottleneck_size=$BOTTLENECK "
        echo "========================================================================================================"
        accelerate launch -m adapters.experiment --checkpoint "$CHECKPOINT" --task_name "$TASK" --bottleneck_size "$BOTTLENECK" --seed "$SEED"
    done
done

echo "Completed experiments for $CHECKPOINT with adapters on $TASK!"