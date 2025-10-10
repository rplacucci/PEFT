#!/usr/bin/env bash
set -euo pipefail

# Default config
CHECKPOINT="bert-base-uncased"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# List of learning rates to tune with
LRS=(
    1e-4
    5e-4
)

# List of batch sizes to tune with
BSS=(
    32
)

# List of epochs to tune with
EPOCHS=(
    3
    20
)

# List of GLUE tasks to tune
TASKS=(
    cola
    mnli
    mrpc
    qqp
    qnli
    rte
    sst2
    stsb
    wnli
)

# List of ranks to tune with
RANKS=(
    4
    16
)

# Loop over each task and invoke accelerate launch
for TASK in "${TASKS[@]}"; do
    for LR in "${LRS[@]}"; do
        for BS in "${BSS[@]}"; do
            for RANK in "${RANKS[@]}"; do
                ALPHA=$((RANK * 2))
                for EPOCH in "${EPOCHS[@]}"; do
                    echo "================================================================================================================================================="
                    echo " Running GLUE LoRA-based tuning on $TASK for checkpoint $CHECKPOINT with lr=$LR, batch_size=$BS, n_epochs=$EPOCH, rank=$RANK, alpha=$ALPHA "
                    echo "================================================================================================================================================="
                    accelerate launch -m lora.tune --checkpoint "$CHECKPOINT" --task_name "$TASK" --rank "$RANK" --alpha "$ALPHA" --lr "$LR" --batch_size "$BS" --n_epochs "$EPOCH"
                done
            done
        done
    done
done

echo "Done tuning $CHECKPOINT with LoRA!"