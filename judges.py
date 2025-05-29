from model import Model
from datetime import datetime, timezone
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import AIMessage
from time import perf_counter
import prompts, re

class UnilateralJudge(Model):
    
    def __init__(self, model_name, prompt_type="baseline", temperature=0.0):
        super().__init__(model_name, temperature)
        if prompt_type == "baseline":
            evaluation_prompt = prompts.DIRECT_EVALUATION_PROMPT
        elif prompt_type == "zero":
            evaluation_prompt = prompts.ZERO_SHOT_EVALUATION_PROMPT
        elif prompt_type == "few":
            evaluation_prompt = prompts.FEW_SHOT_EVALUATION_PROMPT
        else:
            raise ValueError(f'Invalid prompt type: {prompt_type}.')
        self.prompt_type = prompt_type
        evaluation_prompt = PromptTemplate(input_variables=["question", "answer"], template=evaluation_prompt)
        self.evaluate_chain = evaluation_prompt | self.llm

    def _truth_value(self, evaluations):
        pattern = r'\b(TRUE|FALSE)\b'
        matches = [ re.findall(pattern, evaluation) for evaluation in evaluations ]
        evaluation_results = [ match[-1] if match else 'MEANINGLESS' for match in matches ]
        evaluation_result = max(set(evaluation_results), key=evaluation_results.count)
        if evaluation_result == 'TRUE':
            return True
        elif evaluation_result == 'FALSE':
            return False
        elif evaluation_result == 'MEANINGLESS':
            return None
        else:
            raise ValueError(f'Invalid result: {evaluation_result}.')
        
    def _wk_truth_value_to_string(self, tv):
        if tv is None:
            return 'e'
        elif tv:
            return 't'
        else:
            return 'f'
        
    def _tokens_used(self, metadata):
        if 'token_usage' in metadata:
            return metadata['token_usage']['total_tokens']
        elif 'usage' in metadata:
            return metadata['usage']['input_tokens'] + metadata['usage']['output_tokens']
        else:
            raise ValueError(f"Bad model response metadata: {metadata}")
        
    def _total_tokens_used(self, evaluations):
        return sum([ self._tokens_used(e) for e in evaluations ])
    
    def invoke(self, dataset_name, datapoint, samples=1):
        t1 = perf_counter()
        evaluation_responses = [ self.evaluate_chain.invoke(datapoint) for i in range(samples) ]
        t2 = perf_counter()
        evaluations_metadata = [ e.response_metadata if isinstance(e, AIMessage) else None for e in evaluation_responses ]
        evaluations_content = [ e.content if isinstance(e, AIMessage) else e for e in evaluation_responses ]
        tokens_used = self._total_tokens_used(evaluations_metadata)
        truth_value = self._truth_value(evaluations_content)
        wk_v = self._wk_truth_value_to_string(truth_value)
        return {
            "question": datapoint["question"],
            "answer": datapoint["answer"],
            "label": datapoint["label"] if "label" in datapoint else None,
            "model_name": self.model_name,
            "prompt_type": self.prompt_type,
            "dataset_name": dataset_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "execution_time": t2 - t1,
            "tokens_used": tokens_used,
            "evaluations": evaluations_content,
            "wk_v": wk_v
        }
    
class BilateralJudge(Model):
    
    def __init__(self, model_name, prompt_type="direct", temperature=0.0):
        super().__init__(model_name, temperature)
        if prompt_type == "baseline":
            verification_prompt = prompts.DIRECT_VERIFICATION_PROMPT
            refutation_prompt = prompts.DIRECT_REFUTATION_PROMPT
        elif prompt_type == "zero":
            verification_prompt = prompts.ZERO_SHOT_VERIFICATION_PROMPT
            refutation_prompt = prompts.ZERO_SHOT_REFUTATION_PROMPT
        elif prompt_type == "few":
            verification_prompt = prompts.FEW_SHOT_VERIFICATION_PROMPT
            refutation_prompt = prompts.FEW_SHOT_REFUTATION_PROMPT
        else:
            raise ValueError(f'Invalid prompt type: {prompt_type}.')
        self.prompt_type = prompt_type
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
        
    def _truth_value_to_wk_truth_value(self, tv):
        if tv == [ True, False ]:
            return True
        elif tv == [ False, True ]:
            return False
        else:
            return None
    
    def _wk_truth_value_to_string(self, tv):
        if tv is None:
            return 'e'
        elif tv:
            return 't'
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
        I_0 = self._wk_truth_value_to_string(truth_value[0])
        I_1 = self._wk_truth_value_to_string(truth_value[1])
        I = f'<{I_0},{I_1}>'
        wk_v = self._wk_truth_value_to_string(self._truth_value_to_wk_truth_value(truth_value))
        return {
            "question": datapoint["question"],
            "answer": datapoint["answer"],
            "label": datapoint["label"] if "label" in datapoint else None,
            "model_name": self.model_name,
            "prompt_type": self.prompt_type,
            "dataset_name": dataset_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "execution_time": t2 - t1,
            "tokens_used": tokens_used,
            "verifications": verifications_content,
            "refutations": refutations_content,
            "I_0": I_0,
            "I_1": I_1,
            "I": I,
            "wk_v": wk_v
        }