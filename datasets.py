import pandas as pd

class Dataset:

    def __init__(self, dataset_name, sample_size=100, random_seed=9931):
        self.dataset_name = dataset_name
        self.sample_size = sample_size
        self.random_seed = random_seed
        self.records = self._load()

    def _load(self):
        if self.dataset_name == "simpleqa":
            return self._load_simpleqa()
        elif self.dataset_name == "gpqa":
            return self._load_gpqa()
        else:
            raise Exception(f'Dataset {self.dataset_name} not supported')

    def _load_simpleqa(self):
        # Load positive examples
        df_pos = pd.read_csv("data/simpleqa/simple_qa_test_set.csv")
        df_pos.rename(columns={"problem": "question"}, inplace=True)
    
        # Load negative examples
        df_neg = pd.read_csv("data/simpleqa/synthetic_dataset_with_wrong_answers.csv")
        df_neg = df_neg[["metadata", "problem", "wrong_answer_1"]]
        df_neg.rename(columns={"problem": "question", "wrong_answer_1": "answer"}, inplace=True)
    
        # Split and label the data
        half_size = len(df_pos) // 2
        df_pos = df_pos.iloc[:half_size]
        df_pos["label"] = "t"
        df_neg = df_neg.iloc[half_size:]
        df_neg["label"] = "f"
    
        # Combine and shuffle
        df = pd.concat([df_pos, df_neg])
        df = df.sample(frac=1, random_state=self.random_seed)
        df = df.reset_index(drop=True)
    
        return df.to_dict(orient="records")[:self.sample_size]
    
    def _load_gpqa(self):
        # Load examples
        df_pos = pd.read_csv("data/gpqa/gpqa_main.csv")
        df_neg = df_pos[["Question", "Incorrect Answer 1", "High-level domain", "Subdomain"]]
        df_pos = df_pos[["Question", "Correct Answer", "High-level domain", "Subdomain"]]
        df_pos.rename(columns={"Question": "question", "Correct Answer": "answer", "High-level domain": "domain", "Subdomain": "subdomain"}, inplace=True)
        df_neg.rename(columns={"Question": "question", "Incorrect Answer 1": "answer", "High-level domain": "domain", "Subdomain": "subdomain"}, inplace=True)

        # Split and label the data
        half_size = len(df_pos) // 2
        df_pos = df_pos.iloc[:half_size]
        df_pos["label"] = "t"
        df_neg = df_neg.iloc[half_size:]
        df_neg["label"] = "f"

        # Combine and shuffle
        df = pd.concat([df_pos, df_neg])
        df = df.sample(frac=1, random_state=self.random_seed)
        df = df.reset_index(drop=True)
    
        return df.to_dict(orient="records")[:self.sample_size]
    
    def _load_mmlu_pro(self):
        pass
