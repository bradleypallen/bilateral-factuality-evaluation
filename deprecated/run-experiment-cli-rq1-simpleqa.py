import argparse
from factuality_evaluator_rs import UnilateralFactualityEvaluator, BilateralFactualityEvaluator
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
    parser.add_argument('--random-seed', type=int, default=9931,
                      help='Random seed for dataset shuffling (default: 9931)')
    return parser.parse_args()

def load_and_prepare_data(random_seed, dataset_size):
    # Load positive examples
    df_pos = pd.read_csv("data/simple_qa_test_set.csv")
    
    # Load negative examples
    df_neg = pd.read_csv("data/synthetic_dataset_with_wrong_answers.csv")
    df_neg = df_neg[["metadata", "problem", "wrong_answer_1"]]
    df_neg.rename(columns={"wrong_answer_1": "answer"}, inplace=True)
    
    # Split and label the data
    half_size = len(df_pos) // 2
    df_pos = df_pos.iloc[:half_size]
    df_pos["label"] = "t"
    df_neg = df_neg.iloc[half_size:]
    df_neg["label"] = "f"
    
    # Combine and shuffle
    df = pd.concat([df_pos, df_neg])
    df = df.sample(frac=1, random_state=random_seed)
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
    dataset = load_and_prepare_data(args.random_seed, args.dataset_size)
    
    # Unilateral evaluation
    generate_results(
        UnilateralFactualityEvaluator(args.model, temperature=None, batch_size=1),
        "(UNI)",
        f'experiments/{args.experimental_run_version}/unilateral/{args.model.split("/")[-1]}-simpleqa.json',
        dataset,
        args.n_samples
    )
        
    # Bilateral evaluation
    generate_results(
        BilateralFactualityEvaluator(args.model, temperature=None, batch_size=1),
        "(BIL)",
        f'experiments/{args.experimental_run_version}/bilateral/{args.model.split("/")[-1]}-simpleqa.json',
        dataset,
        args.n_samples
    )

if __name__ == "__main__":
    main()
