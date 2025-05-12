from model import Model
from datetime import datetime, timezone
from prompts import BASELINE_VERIFICATION_PROMPT, BASELINE_REFUTATION_PROMPT
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import AIMessage
from ast import literal_eval
from time import perf_counter
import re

class FactualityEvaluator(Model):
    
    def __init__(self, model_name, temperature=0.0, verification_prompt=BASELINE_VERIFICATION_PROMPT, refutation_prompt=BASELINE_REFUTATION_PROMPT):
        super().__init__(model_name, temperature)
        verify_prompt = PromptTemplate(input_variables=["question", "answer"], template=verification_prompt)
        falsify_prompt = PromptTemplate(input_variables=["question", "answer"], template=refutation_prompt)
        self.verify_chain = verify_prompt | self.llm
        self.falsify_chain = falsify_prompt | self.llm

    def _truth_value(self, verifications, refutations):
        pattern = r'\b(VERIFIED|CANNOT VERIFY|REFUTED|CANNOT REFUTE)\b'
        v_matches = [ re.findall(pattern, verification) for verification in verifications ]
        r_matches = [ re.findall(pattern, refutation) for refutation in refutations ]
        verification_results = [ match[-1] if match else 'MEANINGLESS' for match in v_matches ]
        refutation_results = [ match[-1] if match else 'MEANINGLESS' for match in r_matches ]
        verification_result = max(set(verification_results), key=verification_results.count)
        refutation_result = max(set(refutation_results), key=refutation_results.count)
        if verification_result == 'VERIFIED' and refutation_result == 'REFUTED':
            return [ True, True ]
        elif verification_result == 'VERIFIED' and refutation_result == 'CANNOT REFUTE':
            return [ True, False ]
        elif verification_result == 'VERIFIED' and refutation_result == 'MEANINGLESS':
            return [ True, None ]
        elif verification_result == 'CANNOT VERIFY' and refutation_result == 'REFUTED':
            return [ False, True ]
        elif verification_result == 'CANNOT VERIFY' and refutation_result == 'CANNOT REFUTE':
            return [ False, False ]
        elif verification_result == 'CANNOT VERIFY' and refutation_result == 'MEANINGLESS':
            return [ False, None ]
        elif verification_result == 'MEANINGLESS' and refutation_result == 'REFUTED':
            return [ None, True ]
        elif verification_result == 'MEANINGLESS' and refutation_result == 'CANNOT REFUTE':
            return [ None, False ]
        elif verification_result == 'MEANINGLESS' and refutation_result == 'MEANINGLESS':
            return [ None, None ]
        else:
            raise ValueError(f'Invalid result pair: {verification_result}, {refutation_result}.')
    
    def _truth_value_to_string(self, tv):
        if tv:
            return 't'
        elif tv is None:
            return 'e'
        else:
            return 'f'
        
    def _tokens_used(self, metadata):
        if 'token_usage' in metadata:
            return metadata['token_usage']['total_tokens']
        elif 'usage' in metadata:
            return metadata['usage']['input_tokens'] + metadata['usage']['output_tokens']
        else:
            raise ValueError(f"Bad model response metadata: {metadata}")
        
    def _total_tokens_used(self, verifications, refutations):
        return sum([ self._tokens_used(v) for v in verifications ]) + sum([ self._tokens_used(r) for r in refutations ])
    
    def invoke(self, dataset_name, datapoint, samples=1):
        t1 = perf_counter()
        verification_responses = [ self.verify_chain.invoke(datapoint) for i in range(samples) ]
        refutation_responses = [ self.falsify_chain.invoke(datapoint) for i in range(samples) ]
        t2 = perf_counter()
        verifications_metadata = [ v.response_metadata if isinstance(v, AIMessage) else None for v in verification_responses ]
        refutations_metadata = [ r.response_metadata if isinstance(r, AIMessage) else None for r in refutation_responses ]
        verifications_content = [ v.content if isinstance(v, AIMessage) else v for v in verification_responses ]
        refutations_content = [ r.content if isinstance(r, AIMessage) else r for r in refutation_responses ]
        tokens_used = self._total_tokens_used(verifications_metadata, refutations_metadata)
        truth_value = self._truth_value(verifications_content, refutations_content)
        I_0 = self._truth_value_to_string(truth_value[0])
        I_1 = self._truth_value_to_string(truth_value[1])
        I = f'<{I_0},{I_1}>'
        return {
            "question": datapoint["question"],
            "answer": datapoint["answer"],
            "label": datapoint["label"] if "label" in datapoint else None,
            "model_name": self.model_name,
            "dataset_name": dataset_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "execution_time": t2 - t1,
            "tokens_used": tokens_used,
            "verifications": verifications_content,
            "refutations": refutations_content,
            "I_0": I_0,
            "I_1": I_1,
            "I": I
        }