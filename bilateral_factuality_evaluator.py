from langchain_huggingface import HuggingFaceEndpoint
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from datetime import datetime, timezone
from tqdm import tqdm
from ast import literal_eval
import re, os

class BilateralFactualityEvaluator:
    
    VERIFICATION_PROMPT = """Here is a factual question and a potential answer. 
Your task is to assess whether the answer is true. 
Please first explain your reasoning step by step, 
then conclude with either "TRUE" if you have determined that the answer is true, 
otherwise "CANNOT DETERMINE TRUE", 
and only those conclusions, 
at the very end of your reasoning.

Reasoning steps: 
1. First verify the essential information is present 
2. Check for any supporting facts 
3. Consider question context for implied terms 
4. Note if any missing information is essential vs optional 

Question: {problem}
Answer: {answer}
"""

    FALSIFICATION_PROMPT = """Here is a factual question and a potential answer. 
Your task is to assess whether the answer is false. 
Please first explain your reasoning step by step, 
then conclude with either "FALSE" if you have determined that the answer is false, 
otherwise "CANNOT DETERMINE FALSE", 
and only those conclusions, 
at the very end of your reasoning.

Reasoning steps: 
1. First verify the essential information is present 
2. Check for any contradictory facts
3. Consider question context for implied terms 
4. Note if any missing information is essential vs optional 

Question: {problem}
Answer: {answer}
"""

    def __init__(self, model_name, verification_prompt=VERIFICATION_PROMPT, falsification_prompt=FALSIFICATION_PROMPT, batch_size=1, temperature=0.1):
        self.llm = self._llm(model_name, temperature)
        self.model_name = model_name
        self.batch_size = batch_size
        self.temperature = temperature
        verify_prompt = PromptTemplate(input_variables=["problem", "answer"], template=verification_prompt)
        falsify_prompt = PromptTemplate(input_variables=["problem", "answer"], template=falsification_prompt)
        self.verify_chain = verify_prompt | self.llm
        self.falsify_chain = falsify_prompt | self.llm

    def _truth_value(self, verification, falsification):
        pattern = r'\b(TRUE|CANNOT DETERMINE TRUE|FALSE|CANNOT DETERMINE FALSE)\b'
        v_match = re.search(pattern, verification)
        f_match = re.search(pattern, falsification)
        verification_result = v_match.group(1) if v_match else None
        falsification_result = f_match.group(1) if f_match else None
        if verification_result == 'TRUE':
            if falsification_result == 'FALSE':
                return 'b'
            elif falsification_result == 'CANNOT DETERMINE FALSE': # Should None value be treated as "CANNOT DETERMINE X"?
                return 't'
            else:
                return None
        elif verification_result == 'CANNOT DETERMINE TRUE':
            if falsification_result == 'FALSE':
                return 'f'
            elif falsification_result == 'CANNOT DETERMINE FALSE':
                return 'n'
            else:
                return None
        else:
            return None
    
    def _llm(self, model, temperature=0.1):
        if model in [ "gpt-3.5-turbo", "gpt-4-1106-preview", "gpt-4-0125-preview", "gpt-4o-2024-05-13", "gpt-4o-mini" ]:
            return ChatOpenAI(model_name=model, temperature=temperature)
        elif model in [ "claude-3-opus-20240229", "claude-3-5-sonnet-20240620", "claude-3-haiku-20240307" ]:
            return ChatAnthropic(
                temperature=temperature, 
                anthropic_api_key=os.environ["ANTHROPIC_API_KEY"], 
                model_name=model
            )
        elif model in [ "gemini-1.0-pro" ]:
            return ChatGooglePalm(
                temperature=temperature, 
                google_api_key=os.environ["GOOGLE_API_KEY"], 
                model=model
            )
        elif model in [
            "meta-llama/Llama-2-70b-chat-hf", 
            "mistralai/Mixtral-8x7B-Instruct-v0.1", 
            "mistralai/Mistral-7B-Instruct-v0.3", 
            "google/gemma-2-9b-it",
            "google/gemma-7b-it", 
            "google/gemma-2b-it",
            "meta-llama/Meta-Llama-3-70B-Instruct", 
            "microsoft/Phi-3-mini-128k-instruct",
            ]:
            return HuggingFaceEndpoint(
                repo_id=model, 
                temperature=temperature, 
                timeout=300,
                huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"]
            )
        else:
            raise Exception(f'Model {model} not supported')

    def batch(self, data):
        results = []
        batches = [ data[i:i+self.batch_size] for i in range(0, len(data), self.batch_size) ] 
        for batch in tqdm(batches, desc=f'{self.model_name:36}', total=len(batches)):
            falsifications = self.falsify_chain.batch(batch)
            verifications = self.verify_chain.batch(batch)
            for i in range(len(verifications)):
                results.append({
                    "metadata": literal_eval(batch[i]["metadata"]),
                    "problem": batch[i]["problem"],
                    "answer": batch[i]["answer"],
                    "model_name": self.model_name,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    # "total_tokens": verifications[i].response_metadata["token_usage"]["total_tokens"] + falsifications[i].response_metadata["token_usage"]["total_tokens"],
                    # "verification": verifications[i].content,
                    # "falsification": falsifications[i].content,
                    "verification": verifications[i],
                    "falsification": falsifications[i],
                    # "evaluation": self._truth_value(verifications[i].content, falsifications[i].content)
                    "evaluation": self._truth_value(verifications[i], falsifications[i])
                })
        return results
    
    def invoke(self, datapoint):
        falsification = self.falsify_chain.invoke(datapoint)
        verification = self.verify_chain.invoke(datapoint)
        return {
            "metadata": literal_eval(datapoint["metadata"]),
            "problem": datapoint["problem"],
            "answer": datapoint["answer"],
            "model_name": self.model_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            # "total_tokens": verifications[i].response_metadata["token_usage"]["total_tokens"] + falsifications[i].response_metadata["token_usage"]["total_tokens"],
            # "verification": verifications[i].content,
            # "falsification": falsifications[i].content,
            "verification": verification,
            "falsification": falsification,
            # "evaluation": self._truth_value(verifications[i].content, falsifications[i].content)
            "evaluation": self._truth_value(verification, falsification)
        }