import argparse
from factuality_evaluator_rs_sf import UnilateralFactualityEvaluator, BilateralFactualityEvaluator
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
    parser.add_argument('--model', type=str, default="nf-gpt-4o-mini",
                      help='Model for the experimental run (default: nf-gpt-4o-mini)')
    parser.add_argument('--dataset-size', type=int, default=100,
                      help='Number of samples to use from the dataset (default: 100)')
    parser.add_argument('--n-samples', type=int, default=3,
                      help='Number of samples to generate for each evaluation (default: 3)')
    parser.add_argument('--experimental-run-version', type=str, required=True,
                      help='Version identifier for the experimental run')
    return parser.parse_args()

def load_and_prepare_data(dataset_size):
    df = pd.read_json("data/short-form-factuality/simpleqa_results_gpt-4o_assistant.json")
    mapping = {"A": "t", "B": "f"}
    df["label"] = df["grade_letter"].map(mapping).fillna("n")
    df.rename(columns={"question": "problem", "answer": "ground_truth", "predicted_answer": "answer"}, inplace=True)
    df = df.reset_index(drop=True)   
    return df.to_dict(orient="records")[:dataset_size]

def generate_results(model, mode, filename, dataset, samples):
    if os.path.isfile(filename):
        results = json.load(open(filename, "r"))
    else:
        results = []
    i = len(results)
    for datapoint in tqdm(dataset[i:], desc=f'{model.model_name:36} {mode}', initial=i, total=len(dataset)):
        results.append(model.invoke(datapoint, samples=samples))
        json.dump(results, open(filename, "w+"))

def main():
    # Initialize environment variables
    init_environment()
    
    # Parse command line arguments
    args = parse_args()
    
    # Create experiment directories
    if not os.path.exists(f'experiments/{args.experimental_run_version}'):
        os.makedirs(f'experiments/{args.experimental_run_version}/unilateral')
        os.makedirs(f'experiments/{args.experimental_run_version}/bilateral')
    
    # Load and prepare dataset
    dataset = load_and_prepare_data(args.dataset_size)
    
    # Unilateral evaluation
    generate_results(
        UnilateralFactualityEvaluator(args.model, temperature=None, batch_size=1),
        "(UNI)",
        f'experiments/{args.experimental_run_version}/unilateral/{args.model.split("/")[-1]}-short-form-factuality-simpleqa.json',
        dataset,
        args.n_samples
    )
        
    # Bilateral evaluation
    generate_results(
        BilateralFactualityEvaluator(args.model, temperature=None, batch_size=1),
        "(BIL)",
        f'experiments/{args.experimental_run_version}/bilateral/{args.model.split("/")[-1]}-short-form-factuality-simpleqa.json',
        dataset,
        args.n_samples
    )

if __name__ == "__main__":
    main()
