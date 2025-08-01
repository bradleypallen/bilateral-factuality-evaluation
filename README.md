# Bilateral Factuality Evaluation

This repository contains experiments investigating the use of bilateral semantics for LLM-based evaluation of atomic formulae. The project compares traditional unilateral evaluation approaches (TRUE/FALSE judgments) with bilateral evaluation methods that separate verification and refutation processes.

## Overview

The research explores whether separating the evaluation process into distinct verification (VERIFIED/CANNOT VERIFY) and refutation (REFUTED/CANNOT REFUTE) judgments can improve the accuracy and reliability of LLM-based factuality assessment compared to traditional binary (TRUE/FALSE) evaluation.

## Citation

This work is presented in:

```bibtex
@inproceedings{allen2025sound,
  title={Sound and Complete Neuro-symbolic Reasoning with LLM-Grounded Interpretations},
  author={Allen, Bradley P. and Chhikara, Prateek and Ferguson, Thomas Macaulay and Ilievski, Filip and Groth, Paul},
  booktitle={Proceedings of Machine Learning Research vol 284:1–29, 2025 19th Conference on Neurosymbolic Learning and Reasoning},
  year={2025}
}
```

A [pre-print version of the paper](https://arxiv.org/abs/2507.09751) is accessible on ArXiv.

## Key Features

- **Bilateral vs Unilateral Evaluation**: Implements both traditional TRUE/FALSE evaluation and novel bilateral verification/refutation approaches
- **Multiple Prompt Strategies**: Supports baseline, zero-shot, and few-shot prompting strategies
- **Multi-Model Support**: Compatible with OpenAI, Anthropic, and OpenRouter model providers
- **Datasets**: Evaluation on SimpleQA and GPQA datasets
- **Statistical Analysis**: Includes subsampling and standard error estimation for robust results

## Architecture

### Core Components

- **`model.py`**: Base LLM wrapper supporting multiple API providers
- **`judges.py`**: Evaluation classes implementing unilateral and bilateral judgment approaches
- **`datasets.py`**: Data loading and preprocessing for SimpleQA and GPQA
- **`prompts.py`**: Complete set of evaluation prompts for different strategies
- **`subsampling.py`**: Statistical analysis and confidence interval estimation

### Evaluation Framework

The system implements three prompt variations:
- **Baseline**: Direct evaluation prompts
- **Zero-shot**: Detailed reasoning steps without examples
- **Few-shot**: Guided evaluation with concrete examples

Two evaluation approaches:
- **Unilateral**: Single TRUE/FALSE judgment per question-answer pair
- **Bilateral**: Separate verification and refutation processes

## Quick Start

### Setup

1. Create and activate a virtual environment:
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure API keys in `.env` file:
```bash
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
OPENROUTER_API_KEY=your_key_here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
# Additional keys for research proxies as needed
```

### Running Experiments

#### Single Model Experiment
```bash
python run-experiment.py --model gpt-4o --run_version v1 --dataset_samples 400 --n_repeated_samples 100
```

#### Parallel Multi-Model Experiments
```bash
# Edit MODELS array in runner.sh to specify desired models
./runner.sh
```

#### Unilateral Baseline
```bash
python run-experiment-unilateral-baseline.py --model gpt-4o --run_version v1
```

### Analysis

Results are stored in `experiments/{version}/` and can be analyzed using the provided Jupyter notebooks:
- `evaluation_v3.ipynb`: Main evaluation analysis
- `evaluation_bil.ipynb`: Bilateral-specific analysis  
- `evaluation_uni.ipynb`: Unilateral-specific analysis

The specific runs used to generate the results presented in the paper are:
- `experiments/v30/`: bilateral results
- `experiments/v31/`: unilateral results

## Supported Models

- **OpenAI**: gpt-4o, gpt-4o-mini, o3-mini
- **Anthropic**: claude-3-5-sonnet-20241022, claude-3-5-haiku-20241022
- **OpenRouter**: Various models including Llama, Gemini, DeepSeek variants

## Data

The project uses two primary datasets:
- **SimpleQA**: General knowledge questions with correct/incorrect answer pairs
- **GPQA**: Graduate-level physics questions with expert-validated answers

## Results

Experimental results are automatically saved as JSON files and can be processed into publication-ready tables using the analysis notebooks. The system supports resumption of interrupted experiments by checking existing result files.
