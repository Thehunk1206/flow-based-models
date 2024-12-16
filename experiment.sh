#!/bin/bash

# Experiments configuration script for Flow Matching models

# Create base output and log directories
# mkdir -p ./outputs ./logs

# File to track completed experiments
COMPLETED_EXPERIMENTS_FILE="./completed_experiments.log"

# Function to check if an experiment has been completed
is_experiment_completed() {
    local experiment_key="$1"
    grep -q "$experiment_key" "$COMPLETED_EXPERIMENTS_FILE" 2>/dev/null
}

# Function to mark an experiment as completed
mark_experiment_completed() {
    local experiment_key="$1"
    echo "$experiment_key" >> "$COMPLETED_EXPERIMENTS_FILE"
}

# Function to run experiment
run_experiment() {
    local model=$1
    local lr=$2
    local batch_size=$3
    local total_steps=$4

    # Create a unique experiment key
    local experiment_key="${model}_${lr}_${batch_size}_${total_steps}"

    # Check if this experiment is already completed
    if is_experiment_completed "$experiment_key"; then
        echo "Experiment $experiment_key already completed. Skipping..."
        return 0
    fi

    echo "Running experiment with:"
    echo "Model: $model"
    echo "Learning Rate: $lr"
    echo "Batch Size: $batch_size"
    echo "Total Steps: $total_steps"

    # Construct the output and log directories
    output_dir="./outputs/results_${model}_${batch_size}_${total_steps}_exp/"
    log_dir="./logs/${model}_${batch_size}_${total_steps}_logs/"

    # Ensure output and log directories exist
    mkdir -p "$output_dir" "$log_dir"

    # Run the Python script with error handling
    set +e  # Disable immediate exit on error
    python train_cfm.py \
        --model "$model" \
        --lr "$lr" \
        --batch_size "$batch_size" \
        --total_steps "$total_steps" \
        --output_dir "$output_dir" \
        --log_dir "$log_dir" \
        --save_step 5000 \
        --warmup 20000

    # Capture the exit status
    local exit_status=$?

    # Mark experiment as completed only if it ran successfully
    if [ $exit_status -eq 0 ]; then
        mark_experiment_completed "$experiment_key"
        echo "Experiment $experiment_key completed successfully!"
    else
        echo "Experiment $experiment_key failed with exit status $exit_status"
    fi

    # Re-enable immediate exit on error
    set -e

    return $exit_status
}

# Create the completed experiments log file if it doesn't exist
touch "$COMPLETED_EXPERIMENTS_FILE"

# Run experiments
# Experiment 1: OTCFM with default settings
run_experiment "otcfm" 1e-4 16 1000000 || echo "Experiment 1 failed"

# Experiment 2: OTCFM with bigger batch-size
run_experiment "otcfm" 1e-4 32 1000000 || echo "Experiment 2 failed"

# Experiment 3: ICFM with default settings
run_experiment "icfm" 1e-4 16 1000000 || echo "Experiment 3 failed"

echo "Experiment script completed. Check completed_experiments.log for details."