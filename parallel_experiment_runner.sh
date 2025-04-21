#!/bin/bash

# Configuration - modify these variables as needed
BENCHMARK_SCRIPT="path/to/your/benchmark_script.py"  # Replace with the path to your Python benchmark script
MODELS=("gpt-4" "claude-3-opus" "claude-3-sonnet" "mistral-large" "llama-3-70b")  # List of models to benchmark
COMMON_ARGS="--dataset benchmark_data.jsonl --verbose"  # Common arguments for all benchmark runs

# Main script execution
echo "Starting parallel benchmark runs for models: ${MODELS[*]}"
echo "--------------------------------------------------------------------------------"

# Launch a Terminal window for each model
for model in "${MODELS[@]}"; do
    echo "Launching benchmark for model: $model"
    
    # Create a custom command that shows which model is being benchmarked
    benchmark_cmd="echo 'Running benchmark for $model' && python $BENCHMARK_SCRIPT --model $model $COMMON_ARGS"
    
    # Use osascript to open a new Terminal window and execute the command
    osascript -e "tell application \"Terminal\"
        do script \"$benchmark_cmd\"
        set custom title of front window to \"Benchmark: $model\"
    end tell"
    
    # Small delay to avoid overwhelming the system
    sleep 0.5
done

echo "--------------------------------------------------------------------------------"
echo "All benchmark processes have been launched."
echo "Check the individual terminal windows for progress and results."