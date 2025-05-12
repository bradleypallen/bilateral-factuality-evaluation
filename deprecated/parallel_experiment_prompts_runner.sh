#!/bin/bash

# Configuration - modify these variables as needed
BENCHMARK_SCRIPT="run-experiment-cli-rq2-simpleqa-assessor-prompts.py"  # Replace with the path to your Python benchmark script
# MODELS=("google/gemma-2-27b-it" "microsoft/phi-4" "deepseek/deepseek-r1-distill-llama-8b" "google/gemini-2.0-flash-001" "qwen/qwen-2.5-72b-instruct" "meta-llama/llama-4-scout" "meta-llama/llama-4-maverick" "deepseek/deepseek-chat-v3-0324" "nf-gpt-4o" "nf-gpt-4o-mini" "o3-mini" "claude-3-5-sonnet-20241022" "claude-3-5-haiku-20241022")  # List of models to benchmark
MODELS=("google/gemini-2.5-pro-preview-03-25" "google/gemini-2.5-flash-preview" "meta-llama/llama-4-scout" "meta-llama/llama-4-maverick" "deepseek/deepseek-chat-v3-0324" "deepseek/deepseek-r1-distill-llama-8b" "nf-gpt-4o" "nf-gpt-4o-mini" "claude-3-5-sonnet-20241022" "claude-3-5-haiku-20241022")  # List of models to benchmark
PROMPTS=("contrastive" "adversarial" "confidence" "counterfactual" "multistep" "causal")
COMMON_ARGS="--dataset-size 250 --n-samples 1 --experimental-run-version v27-assessor-prompts-simpleqa-bil-3-uni-3"  # Common arguments for all benchmark runs

# Main script execution
echo "Starting parallel benchmark runs for models: ${MODELS[*]}"
echo "--------------------------------------------------------------------------------"

# Launch a Terminal window for each model
for prompt in "${PROMPTS[@]}"; do
    echo "Launching benchmark for prompt: $prompt"
    
    # Create a custom command that shows which prompt is being benchmarked
    benchmark_cmd="echo 'Running benchmark for $prompt' && cd /Users/bradleyallen/Documents/GitHub/bilateral-factuality-evaluation && source env/bin/activate && clear && python $BENCHMARK_SCRIPT --model nf-gpt-4o --prompt $prompt $COMMON_ARGS"
    
    # Use osascript to open a new Terminal window and execute the command
    osascript -e "tell application \"Terminal\"
        do script \"$benchmark_cmd\"
        set custom title of front window to \"Benchmark: $prompt\"
    end tell"
    
    # Small delay to avoid overwhelming the system
    sleep 0.5
done

echo "--------------------------------------------------------------------------------"
echo "All benchmark processes have been launched."
echo "Check the individual terminal windows for progress and results."