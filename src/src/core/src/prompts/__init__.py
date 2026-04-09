"""
Prompt templates for each processing step.

Organized by pattern:
- Chaining: Sequential pipeline prompts
- Routing: Branch-specific prompts
- Parallelization: Concurrent task prompts
- Reflection: Self-improvement prompts
"""

PREPROCESSING_SYSTEM = """You are a text preprocessing specialist. Your task is to clean and normalize customer support messages.
Focus on fixing typos, expanding abbreviations, and standardizing the format while preserving the original meaning.
Do not add or remove information - only improve readability."""

PREPROCESSING_PROMPT = """Clean and normalize the following customer support message:

Original message: {raw_input}

Tasks:
1. Fix obvious spelling errors and typos
2. Expand common abbreviations (e.g., 'pls' -> 'please', 'asap' -> 'as soon as possible')
3. Standardize punctuation and capitalization
4. Preserve all original information and intent

Output the cleaned message and list the corrections made in JSON format:
{{
    "cleaned_message": "the corrected message text",
    "corrections": [
        {{"original": "acount", "corrected": "account"}},
        {{"original": "pls", "expanded_to": "please"}}
    ]
}}"""


CLASSIFICATION_SYSTEM = """You are a customer support ticket classifier. Analyze the message and determine:
- The appropriate category (technical, billing, general, complaint)
- The urgency level (high, medium, low)
- Key entities like product name and issue type

Be precise and consistent in your classification."""

CLASSIFICATION_PROMPT = """Classify the following customer support message:

Cleaned message: {preprocessing.cleaned_message}

Determine:
1. Category: technical (login, app issues), billing (payments, refunds), general (information requests), or complaint (escalations)
2. Urgency: high (immediate attention needed), medium (timely response needed), low (informational)
3. Product name mentioned (if any)
4. Issue type (specific problem described)
5. Key entities (important nouns and phrases)

Output in JSON format:
{{
    "category": "technical|billing|general|complaint",
    "urgency": "high|medium|low",
    "product_name": "extracted product name or null",
    "issue_type": "brief description of the issue",
    "key_entities": ["entity1", "entity2"],
    "reasoning": "brief explanation of classification"
}}"""


RESPONSE_GENERATION_SYSTEM = """You are a customer support representative. Draft professional, helpful, and empathetic responses.
Use the structured information provided to craft an appropriate reply.
Match the tone to the situation - technical for issues, empathetic for complaints, informative for general inquiries."""

RESPONSE_GENERATION_PROMPT = """Create a draft response based on the following structured information:

Ticket Category: {classification.category}
Urgency: {classification.urgency}
Product: {classification.product_name}
Issue: {classification.issue_type}
Key Entities: {classification.key_entities}
Sentiment: {parallel_analysis.sentiment}
Keywords: {parallel_analysis.keywords}

Original message: {raw_input}

Draft a professional response that:
1. Acknowledges the customer's concern
2. Addresses the specific issue mentioned
3. Sets appropriate expectations for resolution
4. Uses a tone matching the situation

Output in JSON format:
{{
    "content": "the full response text",
    "tone": "empathetic|professional|technical|apologetic",
    "key_points": ["point1", "point2"],
    "action_items": ["action1", "action2"]
}}"""

TECHNICAL_BRANCH_PROMPT = """You are handling a TECHNICAL SUPPORT ticket.

Issue: {issue_type}
Product: {product_name}
Urgency: {urgency}

Generate troubleshooting steps for this technical issue:
1. Identify the likely root cause
2. Provide 3-5 clear, actionable troubleshooting steps
3. Include verification steps to confirm resolution
4. Note any known issues or workarounds

Format as a helpful response with numbered steps."""

BILLING_BRANCH_PROMPT = """You are handling a BILLING/REFUND ticket.

Issue: {issue_type}
Urgency: {urgency}
Key details: {key_entities}

Address this billing concern:
1. Acknowledge the billing issue specifically
2. Reference company refund/credit policy
3. Outline the investigation process
4. Provide timeline expectations
5. Request any additional needed information

Format as a professional, reassuring response."""

GENERAL_BRANCH_PROMPT = """You are handling a GENERAL INQUIRY ticket.

Question topic: {issue_type}
Key entities: {key_entities}

Provide helpful information:
1. Direct answer to the question
2. Relevant additional context
3. Links or references to more resources (simulated)
4. Invitation for follow-up questions

Format as a friendly, informative response."""

COMPLAINT_BRANCH_PROMPT = """You are handling a COMPLAINT/ESCALATION ticket.

Issue: {issue_type}
Urgency: {urgency}
Sentiment: {sentiment}

Address this complaint with care:
1. Sincere apology and acknowledgment
2. Specific acknowledgment of their frustration
3. Clear statement of how this will be addressed
4. Escalation notice (this is being prioritized)
5. Concrete next steps and timeline
6. Direct contact information for follow-up

Format as an empathetic, action-oriented response."""


SENTIMENT_ANALYSIS_PROMPT = """Analyze the sentiment of this customer message:

Message: {message}

Classify as:
- positive: satisfied, grateful, happy
- neutral: factual, questioning, neutral
- negative: frustrated, angry, disappointed

Output in JSON format:
{{
    "sentiment": "positive|neutral|negative",
    "score": 0.0 to 1.0 (confidence),
    "key_phrases": ["phrase1", "phrase2"]
}}"""

KEYWORD_EXTRACTION_PROMPT = """Extract the most important keywords and phrases from:

Message: {message}

Identify:
1. Product names or features
2. Action verbs indicating what customer wants
3. Problem descriptors
4. Time-related terms

Return as JSON array of strings: ["keyword1", "keyword2", ...]"""

PRIORITY_SCORING_PROMPT = """Score the priority of this support ticket (1-5, where 5 is highest priority):

Message: {message}
Category: {category}

Consider:
- Urgency language (e.g., "asap", "urgent", "immediately")
- Impact (e.g., "can't access", "lost data", "charged twice")
- Customer frustration level

Return only the integer score (1-5)."""

LANGUAGE_DETECTION_PROMPT = """Detect the primary language of this message:

Message: {message}

Return only the language name in English (e.g., 'English', 'Spanish', 'French')."""

REFLECTION_CRITIQUE_PROMPT = """You are a quality assurance reviewer. Critically evaluate this draft customer support response.

Original customer message: {original_message}

Ticket category: {category}
Urgency: {urgency}
Customer sentiment: {sentiment}

Draft response to evaluate:
---
{draft_response}
---

Evaluate on these criteria (score 1-5 for each):
1. Tone appropriateness: Does it match the situation?
2. Completeness: Does it address all customer concerns?
3. Empathy: Does it show understanding of the customer's situation?
4. Clarity: Is the response clear and easy to understand?
5. Actionability: Does it provide clear next steps?

Provide your critique in JSON format:
{{
    "overall_score": 1-5,
    "criterion_scores": {{
        "tone": 1-5,
        "completeness": 1-5,
        "empathy": 1-5,
        "clarity": 1-5,
        "actionability": 1-5
    }},
    "strengths": ["strength1", "strength2"],
    "weaknesses": ["weakness1", "weakness2"],
    "specific_improvements": ["specific suggestion1", "suggestion2"],
    "requires_revision": true/false
}}"""

REFLECTION_IMPROVEMENT_PROMPT = """Improve this draft response based on the critique provided.

Original customer message: {original_message}

Original draft response:
---
{draft_response}
---

Critique received:
{critique}

Specific improvements needed:
{improvements}

Generate an improved version of the response that addresses all feedback while maintaining professionalism.
Return only the improved response text."""
