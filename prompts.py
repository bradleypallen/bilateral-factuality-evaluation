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

FACTUALITY_PROMPT_V6 = """Evaluate if the answer is definitively TRUE or FALSE for the given question. 

Question: {problem}
Answer: {answer}

Conclude EXACTLY with either:
"TRUE" - Only if every claim is independently verified
"FALSE" - If any essential claim cannot be verified"""

FACTUALITY_PROMPT_V8 = """Extract the core answer from the response, then evaluate if this extracted answer is definitively TRUE or FALSE for the question.

STEP 1: Extract the specific entity, term, or value that directly answers the question:
Question: {problem}
Response: {answer}

Extracted Answer: 

STEP 2: Evaluate if the extracted answer is definitively TRUE or FALSE:
1. Verify the extracted answer against known facts
2. Check for any contradictory information
3. Consider question context and any implied terms
4. Determine if the extracted answer correctly responds to what was asked

Reasoning:

Conclude EXACTLY with either:
"TRUE" - Only if the extracted answer is definitively correct
"FALSE" - If the extracted answer is incorrect or cannot be verified

Explain your reasoning first, then provide your conclusion."""

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

VERIFICATION_PROMPT_V6 = """Evaluate if this answer is definitively TRUE for the given question.

Required steps:
1. Parse key claims in both question and answer
2. Verify each claim against known facts
3. Identify any unstated assumptions
4. Check for temporal/contextual dependencies
5. Validate logical connections between claims

Question: {problem}
Answer: {answer}

Conclude EXACTLY with either:
"VERIFIED" - Only if every claim is independently verified
"CANNOT VERIFY" - If any essential claim cannot be verified

Explain your reasoning first, then provide your conclusion."""

VERIFICATION_PROMPT_V7 = """Evaluate if this answer is definitively TRUE for the given question.

Question: {problem}
Answer: {answer}

Conclude EXACTLY with either:
"VERIFIED" - Only if every claim is independently verified
"CANNOT VERIFY" - If any essential claim cannot be verified"""

VERIFICATION_PROMPT_V8 = """Extract the core answer from the LLM response, then evaluate if this extracted answer is TRUE for the question.

STEP 1: Extract the specific entity, term, or value that directly answers the question:
Question: {problem}
LLM Response: {answer}

Extracted Answer: 

STEP 2: Evaluate if the extracted answer is definitively TRUE:
1. Verify the extracted answer against known facts
2. Consider temporal/contextual factors
3. Determine if it correctly responds to what the question is asking

Reasoning:

Conclude EXACTLY with either:
"VERIFIED" - Only if the extracted answer is definitively correct
"CANNOT VERIFY" - If the extracted answer cannot be verified as correct"""


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

REFUTATION_PROMPT_V6 = """Evaluate if this answer is definitively FALSE for the given question.

Required steps:
1. Parse key claims in both question and answer
2. Search for any direct contradictions
3. Test for logical inconsistencies
4. Check for impossible conditions
5. Identify mutually exclusive scenarios

Question: {problem}
Answer: {answer}

Conclude EXACTLY with either:
"REFUTED" - Only if a contradiction is found
"CANNOT REFUTE" - If no definitive contradiction exists

Explain your reasoning first, then provide your conclusion."""

REFUTATION_PROMPT_V7 = """Evaluate if this answer is definitively FALSE for the given question.

Question: {problem}
Answer: {answer}

Conclude EXACTLY with either:
"REFUTED" - Only if a contradiction is found
"CANNOT REFUTE" - If no definitive contradiction exists"""

REFUTATION_PROMPT_V8 = """Extract the core answer from the LLM response, then evaluate if this extracted answer is FALSE for the question.

STEP 1: Extract the specific entity, term, or value that directly answers the question:
Question: {problem}
LLM Response: {answer}

Extracted Answer: 

STEP 2: Evaluate if the extracted answer is definitively FALSE:
1. Check if the extracted answer contradicts established facts
2. Consider if temporal/contextual factors make this answer incorrect
3. Determine if it fails to correctly answer what the question is asking

Reasoning:

Conclude EXACTLY with either:
"REFUTED" - Only if the extracted answer is definitively incorrect
"CANNOT REFUTE" - If the extracted answer cannot be proven false"""

ANSWER_EXTRACTION_PROMPT_V1 = """Extract ONLY the specific named entity, term, or value that directly answers the question. Return nothing but this core answer without any explanatory text or context.

Question: {{question}}
LLM Answer: {{predicted_answer}}

Instructions:
1. Identify what specific piece of information the question is asking for (a name, date, number, place, etc.)
2. Find the exact answer to this request in the LLM's response
3. Extract ONLY that specific answer element - no articles, no explanations, no qualifiers
4. The extracted answer should be comparable to this reference format: "{{answer}}"

EXTRACTED ANSWER:"""
