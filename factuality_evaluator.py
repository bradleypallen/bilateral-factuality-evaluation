from langchain_huggingface import HuggingFaceEndpoint
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import AIMessage
from datetime import datetime, timezone
from tqdm import tqdm
from ast import literal_eval
import re, os

class Model:

    def __init__(self, model_name, batch_size=1, temperature=0.1):
        self.llm = self._llm(model_name, temperature)
        self.model_name = model_name
        self.batch_size = batch_size
        self.temperature = temperature

    def _llm(self, model_name, temperature=0.1):
        if model_name in [ "gpt-3.5-turbo", "gpt-4-1106-preview", "gpt-4-0125-preview", "gpt-4o-2024-05-13", "gpt-4o-mini", "gpt-4o-2024-11-20" ]:
            return ChatOpenAI(model_name=model_name, temperature=temperature)
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
        else:
            raise Exception(f'Model {model_name} not supported')

class UnilateralFactualityEvaluator(Model):
    
    FACTUALITY_PROMPT = """Here is a factual question and a potential answer. 
Your task is to determine the truth value of the answer to the question. 
Please first explain your reasoning step by step. 
Then conclude with "TRUE" if you are certain that your reasoning verifies the answer, 
otherwise conclude with "FALSE" if you are certain that your reasoning refutes the answer.

Reasoning steps: 
1. First verify the essential information is present 
2. Check for any supporting facts 
3. Check for any contradictory facts
4. Consider question context for implied terms 
5. Note if any missing information is essential vs optional 

Question: {problem}
Answer: {answer}
"""

    def __init__(self, model_name, batch_size=1, temperature=0.1, factuality_prompt=FACTUALITY_PROMPT):
        super().__init__(model_name, batch_size, temperature)
        prompt = PromptTemplate(input_variables=["problem", "answer"], template=factuality_prompt)
        self.chain = prompt | self.llm

    def _truth_value(self, verification):
        pattern = r'\b(TRUE|FALSE)\b'
        matches = re.findall(pattern, verification)
        result = matches[-1] if matches else 'NOT ATTEMPTED'
        if result == 'TRUE':
            return 't'
        elif result == 'FALSE':
                return 'f'
        else:
            return 'n'
    
    def batch(self, data):
        results = []
        wait_time = 10
        batches = [ data[i:i+self.batch_size] for i in range(0, len(data), self.batch_size) ] 
        for batch in tqdm(batches, desc=f'{self.model_name:36}', total=len(batches)):
            while True:
                try:
                    evaluations = self.chain.batch(batch)
                    break
                except Exception as e:
                    time.sleep(wait_time)
                    wait_time *= 2
                    continue
            for i in range(len(evaluations)):
                reasoning = evaluations[i].content if isinstance(evaluations[i], AIMessage) else evaluations[i]
                results.append({
                    "metadata": literal_eval(batch[i]["metadata"]) if "metadata" in batch[i] else None,
                    "problem": batch[i]["problem"],
                    "answer": batch[i]["answer"],
                    "model_name": self.model_name,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "reasoning": reasoning,
                    "evaluation": self._truth_value(reasoning)
                })
        return results
    
    def invoke(self, datapoint):
        reasoning = self.chain.invoke(datapoint)
        reasoning = reasoning.content if isinstance(reasoning, AIMessage) else reasoning
        return {
            "metadata": literal_eval(datapoint["metadata"]) if "metadata" in datapoint else None,
            "problem": datapoint["problem"],
            "answer": datapoint["answer"],
            "model_name": self.model_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "reasoning": reasoning,
            "evaluation": self._truth_value(reasoning)
        }
    
class BilateralFactualityEvaluator(Model):
    
#     VERIFICATION_PROMPT = """Here is a factual question and a potential answer. 
# Your task is to verify the answer to the question. 
# Please first explain your reasoning step by step. 
# Then conclude with "TRUE" if you are certain that your reasoning verifies the answer, 
# otherwise conclude with "CANNOT DETERMINE TRUE" if you cannot definitively verify the answer.

# Reasoning steps: 
# 1. First verify the essential information is present 
# 2. Check for any supporting facts 
# 3. Consider question context for implied terms 
# 4. Note if any missing information is essential vs optional 

# Question: {problem}
# Answer: {answer}
# """

#     FALSIFICATION_PROMPT = """Here is a factual question and a potential answer. 
# Your task is to refute the answer to the question. 
# Please first explain your reasoning step by step. 
# Conclude with  "FALSE" if you are certain that your reasoning refutes the answer, 
# otherwise conclude with "CANNOT DETERMINE FALSE" if you cannot definitively refute the answer.

# Reasoning steps: 
# 1. First verify the essential information is present 
# 2. Check for any contradictory facts
# 3. Consider question context for implied terms 
# 4. Note if any missing information is essential vs optional 

# Question: {problem}
# Answer: {answer}
# """

    VERIFICATION_PROMPT = """Evaluate if this answer is definitively TRUE for the given question.

Required steps:
1. Parse key claims in both question and answer
2. Verify each claim against known facts
3. Identify any unstated assumptions
4. Check for temporal/contextual dependencies
5. Validate logical connections between claims

Question: {problem}
Answer: {answer}

Conclude EXACTLY with either:
"TRUE" - Only if every claim is independently verified
"CANNOT DETERMINE TRUE" - If any essential claim cannot be verified

Explain your verification process first, then your conclusion."""

    FALSIFICATION_PROMPT = """Evaluate if this answer is definitively FALSE for the given question.

Required steps:
1. Parse key claims in both question and answer
2. Search for any direct contradictions
3. Test for logical inconsistencies
4. Check for impossible conditions
5. Identify mutually exclusive scenarios

Question: {problem}
Answer: {answer}

Conclude EXACTLY with either:
"FALSE" - Only if a contradiction is found
"CANNOT DETERMINE FALSE" - If no definitive contradiction exists

Explain your falsification process first, then your conclusion."""

    def __init__(self, model_name, batch_size=1, temperature=0.1, verification_prompt=VERIFICATION_PROMPT, falsification_prompt=FALSIFICATION_PROMPT):
        super().__init__(model_name, batch_size, temperature)
        verify_prompt = PromptTemplate(input_variables=["problem", "answer"], template=verification_prompt)
        falsify_prompt = PromptTemplate(input_variables=["problem", "answer"], template=falsification_prompt)
        self.verify_chain = verify_prompt | self.llm
        self.falsify_chain = falsify_prompt | self.llm

    def _truth_value(self, verification, falsification):
        pattern = r'\b(TRUE|CANNOT DETERMINE TRUE|FALSE|CANNOT DETERMINE FALSE)\b'
        v_matches = re.findall(pattern, verification)
        f_matches = re.findall(pattern, falsification)
        verification_result = v_matches[-1] if v_matches else 'CANNOT DETERMINE TRUE'
        falsification_result = f_matches[-1] if f_matches else 'CANNOT DETERMINE FALSE'
        if verification_result == 'TRUE':
            if falsification_result == 'FALSE':
                return 'b'
            else:
                return 't'
        else:
            if falsification_result == 'FALSE':
                return 'f'
            else:
                return 'n'
    
    def batch(self, data):
        results = []
        batches = [ data[i:i+self.batch_size] for i in range(0, len(data), self.batch_size) ] 
        for batch in tqdm(batches, desc=f'{self.model_name:36}', total=len(batches)):
            while True:
                try:
                    verifications = self.verify_chain.batch(batch)
                    break
                except Exception as e:
                    time.sleep(wait_time)
                    wait_time *= 2
                    continue
            while True:
                try:
                    falsifications = self.falsify_chain.batch(batch)
                    break
                except Exception as e:
                    time.sleep(wait_time)
                    wait_time *= 2
                    continue
            for i in range(len(verifications)):
                verification = verifications[i].content if isinstance(verifications[i], AIMessage) else verifications[i]
                falsification = falsifications[i].content if isinstance(falsifications[i], AIMessage) else falsifications[i]
                results.append({
                    "metadata": literal_eval(batch[i]["metadata"]) if "metadata" in batch[i] else None,
                    "problem": batch[i]["problem"],
                    "answer": batch[i]["answer"],
                    "model_name": self.model_name,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "verification": verification,
                    "falsification": falsification,
                    "evaluation": self._truth_value(verification, falsification)
                })
        return results
    
    def invoke(self, datapoint):
        falsification = self.falsify_chain.invoke(datapoint)
        verification = self.verify_chain.invoke(datapoint)
        verification = verification.content if isinstance(verification, AIMessage) else verification
        falsification = falsification.content if isinstance(falsification, AIMessage) else falsification
        return {
            "metadata": literal_eval(datapoint["metadata"]) if "metadata" in datapoint else None,
            "problem": datapoint["problem"],
            "answer": datapoint["answer"],
            "model_name": self.model_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "verification": verification,
            "falsification": falsification,
            "evaluation": self._truth_value(verification, falsification)
        }