from langchain_huggingface import HuggingFaceEndpoint
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import AIMessage
from datetime import datetime, timezone
from prompts import FACTUALITY_PROMPT_V5, VERIFICATION_PROMPT_V6, REFUTATION_PROMPT_V6
from tqdm import tqdm
from ast import literal_eval
import re, os, time

class Model:

    def __init__(self, model_name, batch_size=1, temperature=0.1):
        self.llm = self._llm(model_name, temperature)
        self.model_name = model_name
        self.batch_size = batch_size
        self.temperature = temperature

    def _llm(self, model_name, temperature=0.1):
        if model_name in [ 
            "gpt-3.5-turbo", 
            "gpt-4-1106-preview", 
            "gpt-4-0125-preview", 
            "gpt-4o", 
            "gpt-4o-mini", 
            "gpt-4o-2024-11-20",
            "o3-mini",
            ]:
            return ChatOpenAI(
                model_name=model_name, 
                temperature=temperature                
            )
        elif model_name in [ "claude-3-opus-20240229", "claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022" ]:
            return ChatAnthropic(
                temperature=temperature, 
                anthropic_api_key=os.environ["ANTHROPIC_API_KEY"], 
                model_name=model_name
            )
        elif model_name in [
            "meta-llama/Llama-2-70b-chat-hf", 
            "mistralai/Mixtral-8x7B-Instruct-v0.1", 
            "mistralai/Mistral-7B-Instruct-v0.3", 
            "google/gemma-2-9b-it",
            "google/gemma-7b-it", 
            "google/gemma-2b-it",
            "meta-llama/Llama-3.3-70B-Instruct", 
            "microsoft/Phi-3-mini-128k-instruct",
            "deepseek-ai/DeepSeek-R1",
            ]:
            return HuggingFaceEndpoint(
                repo_id=model_name, 
                temperature=temperature, 
                timeout=300,
                huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"]
            )
        elif model_name in [
            "google/gemma-2-27b-it",
            "microsoft/phi-4",
            "deepseek/deepseek-r1-distill-llama-8b",
            "google/gemini-2.0-flash-001"
            ]:
            return ChatOpenAI(
                openai_api_key=os.environ["OPENROUTER_API_KEY"],
                openai_api_base=os.environ["OPENROUTER_BASE_URL"],
                model_name=model_name, 
                temperature=temperature
            )
        elif model_name in [
            "nf-gpt-4",
            "nf-gpt-4o-mini",
            "nf-gpt-4o",
            "nf-Llama-3.1-8b-instruct",
            "nf-Llama-3.1-70b-instruct"
            ]:
            return ChatOpenAI(
                openai_api_key=os.environ["AI_RESEARCH_PROXY_API_KEY"],
                openai_api_base=os.environ["AI_RESEARCH_PROXY_BASE_URL"],
                model_name=model_name, 
                temperature=temperature
            )
        else:
            raise Exception(f'Model {model_name} not supported')
     
class UnilateralFactualityEvaluator(Model):
    
    def __init__(self, model_name, batch_size=1, temperature=0.1, factuality_prompt=FACTUALITY_PROMPT_V5):
        super().__init__(model_name, batch_size, temperature)
        prompt = PromptTemplate(input_variables=["problem", "answer"], template=factuality_prompt)
        self.chain = prompt | self.llm

    def _truth_value(self, verifications):
        pattern = r'\b(TRUE|FALSE)\b'
        matches = [ re.findall(pattern, verification) for verification in verifications ]
        results = [ match[-1] if match else 'NOT ATTEMPTED' for match in matches ]
        result = max(set(results), key=results.count)
        if result == 'TRUE':
            return 't'
        elif result == 'FALSE':
                return 'f'
        else:
            return 'n'
    
    def batch(self, data):
        results = []
        batches = [ data[i:i+self.batch_size] for i in range(0, len(data), self.batch_size) ] 
        for batch in tqdm(batches, desc=f'{self.model_name:36}', total=len(batches)):
            wait_time = 10
            while True:
                try:
                    evaluations = self.chain.batch(batch)
                    break
                except Exception as e:
                    print(f"Exception {type(e).__name__}, waiting {wait_time} seconds to retry batch...")
                    time.sleep(wait_time)
                    wait_time *= 2
                    continue
            for i in range(len(evaluations)):
                reasoning = evaluations[i].content if isinstance(evaluations[i], AIMessage) else evaluations[i]
                results.append({
                    "metadata": literal_eval(batch[i]["metadata"]) if "metadata" in batch[i] else None,
                    "problem": batch[i]["problem"],
                    "answer": batch[i]["answer"],
                    "label": batch[i]["label"] if "label" in batch[i] else None,
                    "model_name": self.model_name,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "reasoning": reasoning,
                    "evaluation": self._truth_value([ reasoning ])
                })
        return results
    
    def invoke(self, datapoint, samples=1):
        reasonings = [ self.chain.invoke(datapoint) for i in range(samples) ]
        reasonings = [ reasoning.content if isinstance(reasoning, AIMessage) else reasoning for reasoning in reasonings ]
        return {
            "metadata": literal_eval(datapoint["metadata"]) if "metadata" in datapoint else None,
            "problem": datapoint["problem"],
            "answer": datapoint["answer"],
            "label": datapoint["label"] if "label" in datapoint else None,
            "model_name": self.model_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "reasoning": reasonings,
            "evaluation": self._truth_value(reasonings)
        }
        
class BilateralFactualityEvaluator(Model):
    
    def __init__(self, model_name, batch_size=1, temperature=0.1, verification_prompt=VERIFICATION_PROMPT_V6, refutation_prompt=REFUTATION_PROMPT_V6):
        super().__init__(model_name, batch_size, temperature)
        verify_prompt = PromptTemplate(input_variables=["problem", "answer"], template=verification_prompt)
        falsify_prompt = PromptTemplate(input_variables=["problem", "answer"], template=refutation_prompt)
        self.verify_chain = verify_prompt | self.llm
        self.falsify_chain = falsify_prompt | self.llm

    def _truth_value(self, verifications, refutations):
        pattern = r'\b(VERIFIED|CANNOT VERIFY|REFUTED|CANNOT REFUTE)\b'
        v_matches = [ re.findall(pattern, verification) for verification in verifications ]
        r_matches = [ re.findall(pattern, refutation) for refutation in refutations ]
        verification_results = [ match[-1] if match else 'CANNOT VERIFY' for match in v_matches ]
        refutation_results = [ match[-1] if match else 'CANNOT REFUTE' for match in r_matches ]
        verification_result = max(set(verification_results), key=verification_results.count)
        refutation_result = max(set(refutation_results), key=refutation_results.count)
        if verification_result == 'VERIFIED':
            if refutation_result == 'REFUTED':
                return 'b'
            else:
                return 't'
        else:
            if refutation_result == 'REFUTED':
                return 'f'
            else:
                return 'n'
    
    def batch(self, data):
        results = []
        batches = [ data[i:i+self.batch_size] for i in range(0, len(data), self.batch_size) ] 
        for batch in tqdm(batches, desc=f'{self.model_name:36}', total=len(batches)):
            wait_time = 10
            while True:
                try:
                    verifications = self.verify_chain.batch(batch)
                    break
                except Exception as e:
                    print(f"Exception {type(e).__name__}, waiting {wait_time} seconds to retry verification batch...")
                    time.sleep(wait_time)
                    wait_time *= 2
                    continue
            wait_time = 10
            while True:
                try:
                    refutations = self.falsify_chain.batch(batch)
                    break
                except Exception as e:
                    print(f"Encountered {type(e).__name__}, waiting {wait_time} seconds to retry refutation batch...")
                    time.sleep(wait_time)
                    wait_time *= 2
                    continue
            for i in range(len(verifications)):
                verification = verifications[i].content if isinstance(verifications[i], AIMessage) else verifications[i]
                refutation = refutations[i].content if isinstance(refutations[i], AIMessage) else refutations[i]
                results.append({
                    "metadata": literal_eval(batch[i]["metadata"]) if "metadata" in batch[i] else None,
                    "problem": batch[i]["problem"],
                    "answer": batch[i]["answer"],
                    "label": batch[i]["label"] if "label" in batch[i] else None,
                    "model_name": self.model_name,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "verification": verification,
                    "refutation": refutation,
                    "evaluation": self._truth_value([ verification ], [ refutation ])
                })
        return results
    
    def invoke(self, datapoint, samples=1):
        verifications = [ self.verify_chain.invoke(datapoint) for i in range(samples) ]
        refutations = [ self.falsify_chain.invoke(datapoint) for i in range(samples) ]
        verifications = [ verification.content if isinstance(verification, AIMessage) else verification for verification in verifications ]
        refutations = [ refutation.content if isinstance(refutation, AIMessage) else refutation for refutation in refutations ]
        return {
            "metadata": literal_eval(datapoint["metadata"]) if "metadata" in datapoint else None,
            "problem": datapoint["problem"],
            "answer": datapoint["answer"],
            "label": datapoint["label"] if "label" in datapoint else None,
            "model_name": self.model_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "verification": verifications,
            "refutation": refutations,
            "evaluation": self._truth_value(verifications, refutations)
        }