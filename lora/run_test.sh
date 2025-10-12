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

# List of GLUE tasks to evaluate
TASKS=(
	cola
	mnli-m
	mnli-mm
	mrpc
	qqp
	qnli
	rte
	sst2
	stsb
	wnli
	ax
)

# Loop over each task and invoke accelerate launch
for TASK in "${TASKS[@]}"; do
	echo "============================================================"
	echo " Running GLUE test on $TASK for checkpoint $CHECKPOINT "
	echo "============================================================"
	accelerate launch -m lora.test --checkpoint "$CHECKPOINT" --task_name "$TASK"
done

# Once all tasks are done, zip up the submission folder
SUBMISSION_DIR="./outputs/lora/submission-$CHECKPOINT"

echo "Zipping submission directory: $SUBMISSION_DIR"
zip -r "${SUBMISSION_DIR}.zip" "$SUBMISSION_DIR"
echo "Created ${SUBMISSION_DIR}.zip"
echo "Done testing $CHECKPOINT with LoRA!"