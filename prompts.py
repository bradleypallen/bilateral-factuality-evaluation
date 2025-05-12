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
