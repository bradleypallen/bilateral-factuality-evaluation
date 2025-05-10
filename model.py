from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
import os

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
