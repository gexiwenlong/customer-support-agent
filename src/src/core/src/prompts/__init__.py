"""Prompt templates for the customer support ticket processor."""

PREPROCESSING_SYSTEM = "You are a text preprocessing specialist."
PREPROCESSING_PROMPT = "Clean this message: {raw_input}"

CLASSIFICATION_SYSTEM = "You are a ticket classifier."
CLASSIFICATION_PROMPT = "Classify: {preprocessing.cleaned_message}"

RESPONSE_GENERATION_SYSTEM = "You are a support representative."
RESPONSE_GENERATION_PROMPT = "Draft response for: {raw_input}"

SENTIMENT_ANALYSIS_PROMPT = "Analyze sentiment: {message}"
KEYWORD_EXTRACTION_PROMPT = "Extract keywords: {message}"
PRIORITY_SCORING_PROMPT = "Score priority (1-5): {message}"
LANGUAGE_DETECTION_PROMPT = "Detect language: {message}"

TECHNICAL_BRANCH_PROMPT = "Technical issue: {issue_type}"
BILLING_BRANCH_PROMPT = "Billing issue: {issue_type}"
GENERAL_BRANCH_PROMPT = "General inquiry: {issue_type}"
COMPLAINT_BRANCH_PROMPT = "Complaint: {issue_type}"

REFLECTION_CRITIQUE_PROMPT = "Critique this response: {draft_response}"
REFLECTION_IMPROVEMENT_PROMPT = "Improve this response: {draft_response}"
