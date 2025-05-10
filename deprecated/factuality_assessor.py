from langchain_huggingface import HuggingFaceEndpoint
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import AIMessage
from datetime import datetime, timezone
from ast import literal_eval
import re, os

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
        elif model_name in [ 
            "claude-3-opus-20240229", 
            "claude-3-5-sonnet-20241022", 
            "claude-3-5-haiku-20241022" 
            ]:
            return ChatAnthropic(
                temperature=temperature, 
                anthropic_api_key=os.environ["ANTHROPIC_API_KEY"], 
                model_name=model_name
            )
        elif model_name in [
            "google/gemma-2-27b-it",
            "microsoft/phi-4",
            "deepseek/deepseek-chat-v3-0324",
            "deepseek/deepseek-r1-distill-llama-8b",
            "google/gemini-2.0-flash-001",
            "qwen/qwen-2.5-72b-instruct",
            "qwen/qwq-32b",
            "meta-llama/llama-4-scout",
            "meta-llama/llama-4-maverick",
            "google/gemini-2.5-flash-preview",
            "google/gemini-2.5-pro-preview-03-25"
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
     
class FactualityAssessor(Model):
    
    def __init__(self, model_name, prompt, batch_size=1, temperature=0.0):
        super().__init__(model_name, batch_size, temperature)
        prompt = PromptTemplate(input_variables=["problem", "answer"], template=prompt)
        self.chain = prompt | self.llm

    def _truth_value(self, verifications):
        pattern = r'\b(TRUE|FALSE|BOTH|NEITHER)\b'
        matches = [ re.findall(pattern, verification) for verification in verifications ]
        results = [ match[-1] if match else 'NOT ATTEMPTED' for match in matches ]
        result = max(set(results), key=results.count)
        if result == 'TRUE':
            return 't'
        elif result == 'FALSE':
            return 'f'
        elif result == 'BOTH':
            return 'b'
        elif result == 'NEITHER':
            return 'n'
        else:
            return 'n'
        
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
