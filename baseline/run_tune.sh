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
	1e-5
	2e-5
	3e-5
	5e-5
)

# List of batch sizes to tune with
BSS=(
	16
	32
)

# List of epochs to tune with
EPOCHS=(
	2
	3
	4
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

# Loop over each task and invoke accelerate launch
for TASK in "${TASKS[@]}"; do
  	for LR in "${LRS[@]}"; do
    	for BS in "${BSS[@]}"; do
      		for EPOCH in "${EPOCHS[@]}"; do
				echo "========================================================================================================="
				echo " Running GLUE tuning on $TASK for checkpoint $CHECKPOINT with lr=$LR, batch_size=$BS, n_epochs=$EPOCH"
				echo "========================================================================================================="
				accelerate launch -m baseline.tune --checkpoint "$CHECKPOINT" --task_name "$TASK" --lr "$LR" --batch_size "$BS" --n_epochs "$EPOCH"
			done
    	done
  	done
done

echo "Done tuning $CHECKPOINT!"