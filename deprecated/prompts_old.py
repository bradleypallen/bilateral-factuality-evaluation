# FACTUALITY_PROMPT_V4 = """Here is a factual question and a potential answer. 
# Your task is to determine the truth value of the answer to the question. 
# Please first explain your reasoning step by step. 
# Then conclude with "TRUE" if you are certain that your reasoning verifies the answer, 
# otherwise conclude with "FALSE" if you are certain that your reasoning refutes the answer.

# Reasoning steps: 
# 1. First verify the essential information is present 
# 2. Check for any supporting facts 
# 3. Check for any contradictory facts
# 4. Consider question context for implied terms 
# 5. Note if any missing information is essential vs optional 

# Question: {problem}
# Answer: {answer}
# """

# FACTUALITY_PROMPT_V5 = """Evaluate if the answer is definitively TRUE or FALSE for the given question. 

# Reasoning steps: 
# 1. First verify the essential information is present 
# 2. Check for any supporting facts 
# 3. Check for any contradictory facts
# 4. Consider question context for implied terms 
# 5. Note if any missing information is essential vs optional 

# Question: {problem}
# Answer: {answer}

# Conclude EXACTLY with either:
# "TRUE" - Only if every claim is independently verified
# "FALSE" - If any essential claim cannot be verified

# Explain your reasoning first, then provide your conclusion.
# """

# FACTUALITY_PROMPT_V6 = """Evaluate if the answer is definitively TRUE or FALSE for the given question. 

# Question: {problem}
# Answer: {answer}

# Conclude EXACTLY with either:
# "TRUE" - Only if every claim is independently verified
# "FALSE" - If any essential claim cannot be verified"""

# FACTUALITY_PROMPT_V8 = """Extract the core answer from the response, then evaluate if this extracted answer is definitively TRUE or FALSE for the question.

# STEP 1: Extract the specific entity, term, or value that directly answers the question:
# Question: {problem}
# Response: {answer}

# Extracted Answer: 

# STEP 2: Evaluate if the extracted answer is definitively TRUE or FALSE:
# 1. Verify the extracted answer against known facts
# 2. Check for any contradictory information
# 3. Consider question context and any implied terms
# 4. Determine if the extracted answer correctly responds to what was asked

# Reasoning:

# Conclude EXACTLY with either:
# "TRUE" - Only if the extracted answer is definitively correct
# "FALSE" - If the extracted answer is incorrect or cannot be verified

# Explain your reasoning first, then provide your conclusion."""

# VERIFICATION_PROMPT_V4 = """Evaluate if this answer is definitively TRUE for the given question.

# Required steps:
# 1. Parse key claims in both question and answer
# 2. Verify each claim against known facts
# 3. Identify any unstated assumptions
# 4. Check for temporal/contextual dependencies
# 5. Validate logical connections between claims

# Question: {problem}
# Answer: {answer}

# Conclude EXACTLY with either:
# "TRUE" - Only if every claim is independently verified
# "CANNOT DETERMINE TRUE" - If any essential claim cannot be verified

# Explain your verification process first, then your conclusion."""

# VERIFICATION_PROMPT_V5 = """Evaluate if this answer is definitively TRUE for the given question.

# Required steps:
# 1. Parse key claims in both question and answer
# 2. Verify each claim against known facts
# 3. Identify any unstated assumptions
# 4. Check for temporal/contextual dependencies
# 5. Validate logical connections between claims

# Question: {problem}
# Answer: {answer}

# Conclude EXACTLY with either:
# "TRUE" - Only if every claim is independently verified
# "CANNOT DETERMINE TRUE" - If any essential claim cannot be verified

# Explain your reasoning first, then provide your conclusion."""

# VERIFICATION_PROMPT_V6 = """Evaluate if this answer is definitively TRUE for the given question.

# Required steps:
# 1. Parse key claims in both question and answer
# 2. Verify each claim against known facts
# 3. Identify any unstated assumptions
# 4. Check for temporal/contextual dependencies
# 5. Validate logical connections between claims

# Question: {problem}
# Answer: {answer}

# Conclude EXACTLY with either:
# "VERIFIED" - Only if every claim is independently verified
# "CANNOT VERIFY" - If any essential claim cannot be verified

# Explain your reasoning first, then provide your conclusion."""

# VERIFICATION_PROMPT_V7 = """Evaluate if this answer is definitively TRUE for the given question.

# Question: {problem}
# Answer: {answer}

# Conclude EXACTLY with either:
# "VERIFIED" - Only if every claim is independently verified
# "CANNOT VERIFY" - If any essential claim cannot be verified"""

# VERIFICATION_PROMPT_V8 = """Extract the core answer from the LLM response, then evaluate if this extracted answer is TRUE for the question.

# STEP 1: Extract the specific entity, term, or value that directly answers the question:
# Question: {question}
# LLM Response: {answer}

# Extracted Answer: 

# STEP 2: Evaluate if the extracted answer is definitively TRUE:
# 1. Verify the extracted answer against known facts
# 2. Consider temporal/contextual factors
# 3. Determine if it correctly responds to what the question is asking

# Reasoning:

# Conclude EXACTLY with either:
# "VERIFIED" - Only if the extracted answer is definitively correct
# "CANNOT VERIFY" - If the extracted answer cannot be verified as correct"""


# FALSIFICATION_PROMPT_V4 = """Evaluate if this answer is definitively FALSE for the given question.

# Required steps:
# 1. Parse key claims in both question and answer
# 2. Search for any direct contradictions
# 3. Test for logical inconsistencies
# 4. Check for impossible conditions
# 5. Identify mutually exclusive scenarios

# Question: {problem}
# Answer: {answer}

# Conclude EXACTLY with either:
# "FALSE" - Only if a contradiction is found
# "CANNOT DETERMINE FALSE" - If no definitive contradiction exists

# Explain your falsification process first, then your conclusion."""

# FALSIFICATION_PROMPT_V5 = """Evaluate if this answer is definitively FALSE for the given question.

# Required steps:
# 1. Parse key claims in both question and answer
# 2. Search for any direct contradictions
# 3. Test for logical inconsistencies
# 4. Check for impossible conditions
# 5. Identify mutually exclusive scenarios

# Question: {problem}
# Answer: {answer}

# Conclude EXACTLY with either:
# "FALSE" - Only if a contradiction is found
# "CANNOT DETERMINE FALSE" - If no definitive contradiction exists

# Explain your reasoning first, then provide your conclusion."""

# REFUTATION_PROMPT_V6 = """Evaluate if this answer is definitively FALSE for the given question.

# Required steps:
# 1. Parse key claims in both question and answer
# 2. Search for any direct contradictions
# 3. Test for logical inconsistencies
# 4. Check for impossible conditions
# 5. Identify mutually exclusive scenarios

# Question: {problem}
# Answer: {answer}

# Conclude EXACTLY with either:
# "REFUTED" - Only if a contradiction is found
# "CANNOT REFUTE" - If no definitive contradiction exists

# Explain your reasoning first, then provide your conclusion."""

# REFUTATION_PROMPT_V7 = """Evaluate if this answer is definitively FALSE for the given question.

# Question: {problem}
# Answer: {answer}

# Conclude EXACTLY with either:
# "REFUTED" - Only if a contradiction is found
# "CANNOT REFUTE" - If no definitive contradiction exists"""

# REFUTATION_PROMPT_V8 = """Extract the core answer from the LLM response, then evaluate if this extracted answer is FALSE for the question.

# STEP 1: Extract the specific entity, term, or value that directly answers the question:
# Question: {question}
# LLM Response: {answer}

# Extracted Answer: 

# STEP 2: Evaluate if the extracted answer is definitively FALSE:
# 1. Check if the extracted answer contradicts established facts
# 2. Consider if temporal/contextual factors make this answer incorrect
# 3. Determine if it fails to correctly answer what the question is asking

# Reasoning:

# Conclude EXACTLY with either:
# "REFUTED" - Only if the extracted answer is definitively incorrect
# "CANNOT REFUTE" - If the extracted answer cannot be proven false"""

# ANSWER_EXTRACTION_PROMPT_V1 = """Extract ONLY the specific named entity, term, or value that directly answers the question. Return nothing but this core answer without any explanatory text or context.

# Question: {{question}}
# LLM Answer: {{predicted_answer}}

# Instructions:
# 1. Identify what specific piece of information the question is asking for (a name, date, number, place, etc.)
# 2. Find the exact answer to this request in the LLM's response
# 3. Extract ONLY that specific answer element - no articles, no explanations, no qualifiers
# 4. The extracted answer should be comparable to this reference format: "{{answer}}"

# EXTRACTED ANSWER:"""

# UNILATERAL_PROMPT_CONTRASTIVE = """Extract the core answer from the LLM response, then evaluate if this extracted answer is FALSE for the question. 
# Evaluate if this answer is definitively TRUE:

# Question: {problem}
# LLM Response: {answer}

# STEP 1: Extract the core claim being evaluated.

# STEP 2: Conduct verification assessment:
# - What evidence supports this claim?
# - Is this evidence sufficient to confirm the answer?

# Conclude EXACTLY with either:
# "TRUE" - The answer is verifiably correct
# "FALSE" - The answer cannot be verified or is incorrect"""

# BILATERAL_PROMPT_CONTRASTIVE = """Extract the core answer from the LLM response, then evaluate if this extracted answer is FALSE for the question.
# Evaluate the factuality of this answer through bilateral analysis:

# Question: {problem}
# LLM Response: {answer}

# STEP 1: Extract the core claim being evaluated.

# STEP 2: Conduct a bilateral assessment:
# A) VERIFICATION PERSPECTIVE: What evidence supports this claim?
# B) REFUTATION PERSPECTIVE: What evidence contradicts this claim?
# C) CONFLICT ANALYSIS: Identify any tensions between these perspectives.

# Conclude with exactly ONE of the following:
# "TRUE" - Verification successful, refutation unsuccessful
# "FALSE" - Verification unsuccessful, refutation successful 
# "NEITHER" - Both verification and refutation unsuccessful
# "BOTH" - Both verification and refutation successful (indicating contradiction)"""

# UNILATERAL_PROMPT_MULTISTEP_SEQUENTIAL = """Extract the core answer from the LLM response, then evaluate if this extracted answer is FALSE for the question.
# Evaluate this answer through the following sequential steps:

# Question: {problem}
# LLM Response: {answer}

# 1. INITIAL IMPRESSION: State your initial assessment of factuality.
# 2. VERIFICATION SEARCH: Actively look for evidence that confirms this answer.
# 3. EVIDENCE WEIGHING: Evaluate the strength of confirming evidence.
# 4. UNCERTAINTY ASSESSMENT: Identify any areas where knowledge is insufficient.
# 5. FINAL JUDGMENT: Based on steps 1-4, provide your final assessment.

# Conclude with exactly ONE of:
# "TRUE" - Strong confirming evidence found
# "FALSE" - Insufficient or no confirming evidence found"""

# BILATERAL_PROMPT_MULTISTEP_SEQUENTIAL = """Extract the core answer from the LLM response, then evaluate if this extracted answer is FALSE for the question.
# Evaluate this answer through the following sequential steps:

# Question: {problem}
# LLM Response: {answer}

# 1. INITIAL IMPRESSION: State your initial assessment of factuality.
# 2. VERIFICATION SEARCH: Actively look for evidence that confirms this answer.
# 3. REFUTATION SEARCH: Actively look for evidence that contradicts this answer.
# 4. EVIDENCE WEIGHING: Compare the strength of confirming vs contradicting evidence.
# 5. UNCERTAINTY ASSESSMENT: Identify any areas where knowledge is insufficient.
# 6. FINAL JUDGMENT: Based on steps 1-5, provide your final assessment.

# Conclude with exactly ONE of:
# "TRUE" - Strong confirming evidence, minimal contradicting evidence
# "FALSE" - Strong contradicting evidence, minimal confirming evidence
# "NEITHER" - Insufficient evidence for either confirmation or contradiction
# "BOTH" - Substantial evidence for both confirmation and contradiction"""

# UNILATERAL_PROMPT_ADVERSARIAL_DEBATE = """Extract the core answer from the LLM response, then evaluate if this extracted answer is FALSE for the question.
# Make the strongest possible case for the truth of this answer:

# Question: {problem}
# LLM Response: {answer}

# VERIFICATION ADVOCATE: Makes the strongest possible case that the answer is true. Begins with "The evidence supporting this answer includes..."

# JUDGE: Determines if the verification argument is compelling enough.

# Final ruling (choose one):
# "TRUE" - Verification argument is compelling
# "FALSE" - Verification argument is not compelling"""

# BILATERAL_PROMPT_ADVERSARIAL_DEBATE = """Extract the core answer from the LLM response, then evaluate if this extracted answer is FALSE for the question.
# For this answer, simulate a debate between two reasoning systems:

# Question: {problem}
# LLM Response: {answer}

# SYSTEM A (Verification Advocate): Makes the strongest possible case that the answer is true. Begins with "The evidence supporting this answer includes..."

# SYSTEM B (Refutation Advocate): Makes the strongest possible case that the answer is false. Begins with "The evidence against this answer includes..."

# DEBATE JUDGE: Weighs both perspectives and determines which is stronger, or if they are equally balanced.

# Final ruling (choose one):
# "TRUE" - Verification argument clearly stronger
# "FALSE" - Refutation argument clearly stronger
# "NEITHER" - Neither argument is compelling
# "BOTH" - Both arguments are equally compelling"""

# UNILATERAL_PROMPT_CONFIDENCE_ASSESSMENT = """Extract the core answer from the LLM response, then evaluate if this extracted answer is FALSE for the question.
# Assess your confidence in the truth of this answer:

# Question: {problem}
# LLM Response: {answer}

# STEP 1: Assign a confidence score (0-100%) that the answer is TRUE.
# Reasoning:
# Confidence: ___%

# STEP 2: Based on your confidence score, classify as:
# "TRUE" - High confidence in truth (>70%)
# "FALSE" - Low confidence in truth (≤70%)"""

# BILATERAL_PROMPT_CONFIDENCE_ASSESSMENT = """Extract the core answer from the LLM response, then evaluate if this extracted answer is FALSE for the question.
# Assess your confidence in the truth and falsehood of this answer:

# Question: {problem}
# LLM Response: {answer}

# STEP 1: Assign a confidence score (0-100%) that the answer is TRUE.
# Reasoning:
# Confidence: ___%

# STEP 2: Assign a confidence score (0-100%) that the answer is FALSE.
# Reasoning:
# Confidence: ___%

# STEP 3: Based on the above confidence scores, classify as:
# "TRUE" - High confidence in truth (>70%), low confidence in falsehood (<30%)
# "FALSE" - High confidence in falsehood (>70%), low confidence in truth (<30%)
# "NEITHER" - Low confidence in both truth and falsehood (<50% each)
# "BOTH" - High confidence in both truth and falsehood (>50% each)"""

# UNILATERAL_PROMPT_COUNTERFACTUAL = """Extract the core answer from the LLM response, then evaluate if this extracted answer is FALSE for the question.
# Assess this answer through verification reasoning:

# Question: {problem}
# LLM Response: {answer}

# STEP 1: Imagine this answer is TRUE. What would need to be true about the world for this to be correct? Are these conditions met?

# STEP 2: Based on your findings from step 1, determine:
# "TRUE" - Conditions for truth are met
# "FALSE" - Conditions for truth are not met"""

# BILATERAL_PROMPT_COUNTERFACTUAL = """Extract the core answer from the LLM response, then evaluate if this extracted answer is FALSE for the question.
# Assess this answer through counterfactual reasoning:

# Question: {problem}
# LLM Response: {answer}

# STEP 1: Imagine this answer is TRUE. What would need to be true about the world for this to be correct? Are these conditions met?

# STEP 2: Imagine this answer is FALSE. What would need to be true about the world for this to be incorrect? Are these conditions met?

# STEP 3: Compare your findings from steps 1 and 2 to determine:
# "TRUE" - Conditions for truth are met, conditions for falsehood are not
# "FALSE" - Conditions for falsehood are met, conditions for truth are not
# "NEITHER" - Neither set of conditions is clearly met
# "BOTH" - Both sets of conditions appear to be met (indicating a paradox)"""

# UNILATERAL_PROMPT_CAUSAL_ANALYSIS = """Extract the core answer from the LLM response, then evaluate if this extracted answer is FALSE for the question.
# Evaluate this answer through verification analysis:

# Question: {problem}
# LLM Response: {answer}

# STEP 1: What knowledge would cause you to AFFIRM this answer as true? Do you possess this knowledge?

# STEP 2: Based on your answer to step 1, classify as:
# "TRUE" - Possess knowledge that causes affirmation
# "FALSE" - Lack sufficient knowledge to affirm"""

# BILATERAL_PROMPT_CAUSAL_ANALYSIS = """Extract the core answer from the LLM response, then evaluate if this extracted answer is FALSE for the question.
# Evaluate this answer through causal analysis:

# Question: {problem}
# LLM Response: {answer}

# STEP 1: What knowledge would cause you to AFFIRM this answer as true? Do you possess this knowledge?

# STEP 2: What knowledge would cause you to REJECT this answer as false? Do you possess this knowledge?

# STEP 3: Based on your answers to steps 1 and 2, classify as:
# "TRUE" - Possess knowledge that causes affirmation, lack knowledge that causes rejection
# "FALSE" - Possess knowledge that causes rejection, lack knowledge that causes affirmation
# "NEITHER" - Lack knowledge that would cause either affirmation or rejection
# "BOTH" - Possess conflicting knowledge that simultaneously causes affirmation and rejection"""

BASELINE_VERIFICATION_PROMPT = """Determine whether the following answer to the given question is correct. 
Conclude with a single line containing ONLY one of these two phrases:
VERIFIED
CANNOT VERIFY

Question: {question}
Proposed answer: {answer}"""

BASELINE_REFUTATION_PROMPT = """Determine whether the following answer to the given question can be refuted. 
Conclude with a single line containing ONLY one of these two phrases:
REFUTED
CANNOT REFUTE

Question: {question}
Proposed answer: {answer}"""

ZERO_SHOT_VERIFICATION_PROMPT = """I'll provide you with a question and its proposed answer. 
Your task is to verify whether this answer is correct by following these steps:

1. Analyze the exact meaning of both the question and answer, identifying any key terms that need clarification.
2. Establish specific conditions that would make this answer true for this question.
3. Provide direct evidence supporting the answer, including specific facts, examples, or authoritative references that confirm its accuracy.
4. Test if the answer remains valid across all contexts where the question applies, noting any limitations or exceptions.
5. Check for consistency with established knowledge in the relevant domain.

Based on your analysis, determine whether the answer is verified and explain your reasoning with specific supporting evidence. 
Your goal is not to find fault but to determine if positive evidence exists to confirm the answer.

After your complete analysis, conclude with a single line containing ONLY one of these two phrases:
VERIFIED
CANNOT VERIFY

Question: {question}
Proposed answer: {answer}"""

ZERO_SHOT_REFUTATION_PROMPT = """I'll provide you with a question and its proposed answer. 
Your task is to determine if this answer can be refuted by following these steps:

1. Analyze the exact meaning of both the question and the proposed answer.
2. Identify what specific conditions would need to be true for this answer to be false (not merely the absence of evidence).
3. Search for direct counterexamples or contradicting evidence that actively demonstrates why the answer is incorrect.
4. Construct specific scenarios where the answer fails to hold true, even if the question's premises are accepted.
5. Identify any logical inconsistencies, factual errors, or category mistakes within the answer.

Focus on building an affirmative case for why the answer is incorrect, rather than simply noting a lack of supporting evidence. 
Provide specific counterevidence and explain precisely how it contradicts the proposed answer.

After your complete analysis, conclude with a single line containing ONLY one of these two phrases:
REFUTED
CANNOT REFUTE

Question: {question}
Proposed answer: {answer}"""

FEW_SHOT_VERIFICATION_PROMPT = """I'll provide you with a question and its proposed answer. 
Your task is to verify whether this answer is correct by following these steps:

1. Analyze the exact meaning of both the question and answer, identifying any key terms that need clarification.
2. Establish specific conditions that would make this answer true for this question.
3. Provide direct evidence supporting the answer, including specific facts, examples, or authoritative references that confirm its accuracy.
4. Test if the answer remains valid across all contexts where the question applies, noting any limitations or exceptions.
5. Check for consistency with established knowledge in the relevant domain.

Based on your analysis, determine whether the answer is verified and explain your reasoning with specific supporting evidence. 
Your goal is not to find fault but to determine if positive evidence exists to confirm the answer.

Here are some examples of how to approach verification:

Example 1:
Question: Who was the first man to walk on the Moon?
Proposed answer: Neil Armstrong
Let's verify this answer systematically:
Key Terms Analysis:
"First man to walk on the Moon" requires precise historical verification
Specific date and mission details are crucial
Conditions for Verification:
Must be the first human to physically step onto lunar surface
Must have documented evidence of this historic moment
Must be part of a specific NASA mission
Direct Evidence:
Neil Armstrong stepped onto the Moon's surface on July 20, 1969
During Apollo 11 mission
Famous quote: "That's one small step for man, one giant leap for mankind"
Extensive NASA documentation confirms this
Broadcast live on global television
Verified by multiple independent sources including NASA, scientific records, and historical archives
Context and Limitations:
Specifically refers to first human, not first spacecraft
Specifically lunar surface, not orbital passage
Confirmed during Cold War space race
Part of NASA's Apollo program
Consistency Check:
Matches scientific and historical consensus
No credible alternative claims exist
Extensively documented in multiple international records
This answer meets all verification criteria with robust, multi-source evidence.
VERIFIED

Example 2:
Question: Chocolate is made from material
Proposed answer: Sugar
Let's analyze this systematically:
Key Term Analysis:
"Chocolate" refers to a food product derived from cacao/cocoa beans
"Material" suggests the fundamental ingredient or component
"Sugar" is a sweetener, not the primary material of chocolate
Conditions for Accuracy:
Sugar is an ingredient in chocolate
But sugar is NOT the primary material of chocolate
Chocolate's primary material is cocoa/cacao solids
Direct Evidence:
Chocolate is primarily made from cocoa beans processed into cocoa solids
Cocoa solids come from cacao tree seeds/beans
Sugar is added as a sweetener, not the base material
Chocolate composition typically includes:

Cocoa solids (primary material)
Cocoa butter
Sugar (secondary ingredient)
Milk (in milk chocolate)

Context Testing:
In all chocolate production processes, cocoa is the fundamental material
Sugar is always a supplementary ingredient, not the base material
Domain Consistency:
Culinary and food science consistently define cocoa/cacao as chocolate's primary material
Based on comprehensive analysis, the proposed answer is incorrect.
CANNOT VERIFY

After your complete analysis, conclude with a single line containing ONLY one of these two phrases:
VERIFIED
CANNOT VERIFY

Question: {question}
Proposed answer: {answer}"""

FEW_SHOT_REFUTATION_PROMPT = """I'll provide you with a question and its proposed answer. 
Your task is to determine if this answer can be refuted by following these steps:

1. Analyze the exact meaning of both the question and the proposed answer.
2. Identify what specific conditions would need to be true for this answer to be false (not merely the absence of evidence).
3. Search for direct counterexamples or contradicting evidence that actively demonstrates why the answer is incorrect.
4. Construct specific scenarios where the answer fails to hold true, even if the question's premises are accepted.
5. Identify any logical inconsistencies, factual errors, or category mistakes within the answer.

Focus on building an affirmative case for why the answer is incorrect, rather than simply noting a lack of supporting evidence. 
Provide specific counterevidence and explain precisely how it contradicts the proposed answer.

Here are some examples of how to approach refutation:

Example 1:
Question: Are penguins birds?
Proposed answer: No
Let's analyze this systematically:
Meaning Analysis:
Question asks about the taxonomic classification of penguins
Proposed answer claims penguins are NOT birds
Conditions for Falsity:
Penguins must meet standard biological criteria for birds
Must share key avian characteristics
Counterevidence:
Penguins have ALL classic bird characteristics:

Feathered body
Lay eggs
Warm-blooded
Have beaks
Descended from dinosaur lineage
Classified in scientific taxonomy under Class Aves
Specifically, penguins belong to the order Sphenisciformes, which is a recognized bird order
Biological and genetic evidence conclusively places penguins within bird classification

Specific Scenarios Contradicting Answer:
Penguins have wing-like flippers adapted for swimming
They have respiratory and skeletal structures identical to other bird species
Genetic sequencing confirms their bird lineage
Logical Inconsistencies:
Rejecting penguins as birds would require rejecting fundamental biological classification systems
No scientific basis exists for excluding penguins from bird category
REFUTED

Example 2:
Question: Who was the first man to walk on the Moon?
Proposed answer: Neil Armstrong
Let's analyze this systematically:
Meaning Analysis:
Question seeks the definitive first human male to set foot on lunar surface
Proposed answer: Neil Armstrong (Apollo 11 mission, July 20, 1969)
Potential Conditions for Falsity:
Documented evidence of another person walking on Moon before Armstrong
Proof that Armstrong was not actually the first
Historical record showing a different individual preceded him
Counterevidence Search:
No credible historical evidence exists contradicting Armstrong's first Moon walk
NASA records and global documentation consistently confirm Armstrong as first
Extensive photographic and video evidence supports this claim
Scenario Testing:
No alternative scenarios emerge that could plausibly replace Armstrong's achievement
Extensive verification by multiple nations and independent researchers confirms his primacy
Logical Consistency Check:
Armstrong's Moon walk is extensively documented
Multiple witnesses and technological records corroborate the event
No logical inconsistencies detected in the claim
The proposed answer is completely accurate and supported by overwhelming historical evidence.
CANNOT REFUTE

After your complete analysis, conclude with a single line containing ONLY one of these two phrases:
REFUTED
CANNOT REFUTE

Question: {question}
Proposed answer: {answer}"""
