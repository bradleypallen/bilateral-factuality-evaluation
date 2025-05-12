import argparse
from judges import BilateralJudge
from datasets import Dataset
from tqdm import tqdm
import json
import os
import pandas as pd
from dotenv import load_dotenv

def init_environment():
    """Initialize environment variables from .env file"""
    # Load environment variables from .env file
    load_dotenv()
    
    # Check for required environment variables
    required_vars = [
        'OPENAI_API_KEY',
        'ANTHROPIC_API_KEY',
        'HUGGINGFACEHUB_API_TOKEN',
        'OPENROUTER_API_KEY',
        'OPENROUTER_BASE_URL',
        'AI_RESEARCH_PROXY_BASE_URL',
        'AI_RESEARCH_PROXY_API_KEY'
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing_vars)}\n"
            "Please add them to your .env file"
        )

def parse_args():
    parser = argparse.ArgumentParser(description='Run factuality evaluation experiments')
    parser.add_argument('--dataset', type=str, default="simpleqa",
                      help='Dataset for the experimental run (default: simpleqa)')
    parser.add_argument('--model', type=str, default="nf-gpt-4o-mini",
                      help='Model for the experimental run (default: nf-gpt-4o-mini)')
    parser.add_argument('--prompt', type=str, default="baseline",
                      help='Prompt type for the experimental run (default: baseline)')
    parser.add_argument('--dataset_samples', type=int, default=100,
                      help='Number of samples to use from the dataset (default: 100)')
    parser.add_argument('--n_repeated_samples', type=int, default=3,
                      help='Number of samples to generate for each verification/refutation (default: 3)')
    parser.add_argument('--run_version', type=str, required=True,
                      help='Identifier for the experimental run')
    parser.add_argument('--random_seed', type=int, default=9931,
                      help='Random seed for dataset shuffling (default: 9931)')
    return parser.parse_args()

def main():
    # Initialize environment variables
    init_environment()
    
    # Parse command line arguments
    args = parse_args()
    
    # Create experiment directory
    if not os.path.exists(f'experiments/{args.run_version}'):
        os.makedirs(f'experiments/{args.run_version}')
    
    # Evaluation
    filename = f'experiments/{args.run_version}/{args.model.split("/")[-1]}-{args.prompt}-{args.dataset}.json'
    model = BilateralJudge(args.model, args.prompt, temperature=0.1)
    dataset = Dataset(args.dataset, sample_size=args.dataset_samples, random_seed=args.random_seed)

    if os.path.isfile(filename):
        results = json.load(open(filename, "r"))
    else:
        results = []
    i = len(results)
    for datapoint in tqdm(dataset.records[i:], desc=f'{model.model_name:36}', initial=i, total=len(dataset.records)):
        results.append(model.invoke(args.dataset, datapoint, samples=args.n_repeated_samples))
        json.dump(results, open(filename, "w+"))

if __name__ == "__main__":
    main()
