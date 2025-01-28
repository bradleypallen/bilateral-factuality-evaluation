FACTUALITY_PROMPT_V4 = """Here is a factual question and a potential answer. 
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

FACTUALITY_PROMPT_V5 = """Evaluate if the answer is definitively TRUE or FALSE for the given question. 

Reasoning steps: 
1. First verify the essential information is present 
2. Check for any supporting facts 
3. Check for any contradictory facts
4. Consider question context for implied terms 
5. Note if any missing information is essential vs optional 

Question: {problem}
Answer: {answer}

Conclude EXACTLY with either:
"TRUE" - Only if every claim is independently verified
"FALSE" - If any essential claim cannot be verified

Explain your reasoning first, then provide your conclusion.
"""

VERIFICATION_PROMPT_V4 = """Evaluate if this answer is definitively TRUE for the given question.

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

VERIFICATION_PROMPT_V5 = """Evaluate if this answer is definitively TRUE for the given question.

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

Explain your reasoning first, then provide your conclusion."""

FALSIFICATION_PROMPT_V4 = """Evaluate if this answer is definitively FALSE for the given question.

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

FALSIFICATION_PROMPT_V5 = """Evaluate if this answer is definitively FALSE for the given question.

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

Explain your reasoning first, then provide your conclusion."""
