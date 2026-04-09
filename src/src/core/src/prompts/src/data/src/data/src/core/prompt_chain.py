"""
Prompt Chaining Pattern Implementation.

This module implements a sequential pipeline where each step's output
feeds as input to the next step. The chain consists of three main stages:
1. Preprocessing - Clean and normalize the raw message
2. Classification - Determine category, urgency, and extract entities
3. Response Generation - Create initial draft response
"""

import json
import time
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, field
import logging
import os

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


@dataclass
class ChainStep:
    """Definition of a single step in the prompt chain."""
    name: str
    system_prompt: str
    user_prompt_template: str
    output_parser: Callable[[str], Any]
    required_context: list = field(default_factory=list)


class PromptChain:
    """
    Sequential prompt chaining processor.
    
    Executes a series of LLM calls where each step can access
    outputs from previous steps through context placeholders.
    """
    
    def __init__(self, model: str = "gpt-3.5-turbo", temperature: float = 0.3):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.steps: list[ChainStep] = []
    
    def add_step(self, step: ChainStep) -> 'PromptChain':
        """Add a custom step to the chain."""
        self.steps.append(step)
        return self
    
    def _parse_preprocessing_output(self, output: str) -> Dict[str, Any]:
        """Parse the JSON output from preprocessing step."""
        try:
            start_idx = output.find('{')
            end_idx = output.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = output[start_idx:end_idx]
                return json.loads(json_str)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse preprocessing output as JSON")
        
        return {
            "cleaned_message": output.strip(),
            "corrections": []
        }
    
    def _parse_classification_output(self, output: str) -> Dict[str, Any]:
        """Parse the JSON output from classification step."""
        try:
            start_idx = output.find('{')
            end_idx = output.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = output[start_idx:end_idx]
                data = json.loads(json_str)
                
                from .models import TicketCategory, UrgencyLevel
                
                category_map = {
                    "technical": TicketCategory.TECHNICAL,
                    "billing": TicketCategory.BILLING,
                    "general": TicketCategory.GENERAL,
                    "complaint": TicketCategory.COMPLAINT
                }
                
                urgency_map = {
                    "high": UrgencyLevel.HIGH,
                    "medium": UrgencyLevel.MEDIUM,
                    "low": UrgencyLevel.LOW
                }
                
                return {
                    "category": category_map.get(data.get("category", "general"), TicketCategory.GENERAL),
                    "urgency": urgency_map.get(data.get("urgency", "medium"), UrgencyLevel.MEDIUM),
                    "product_name": data.get("product_name"),
                    "issue_type": data.get("issue_type"),
                    "key_entities": data.get("key_entities", []),
                    "reasoning": data.get("reasoning", "")
                }
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse classification output: {e}")
        
        from .models import TicketCategory, UrgencyLevel
        return {
            "category": TicketCategory.GENERAL,
            "urgency": UrgencyLevel.MEDIUM,
            "product_name": None,
            "issue_type": "general inquiry",
            "key_entities": [],
            "reasoning": "Fallback classification due to parsing error"
        }
    
    def _parse_response_output(self, output: str) -> Dict[str, Any]:
        """Parse the JSON output from response generation step."""
        try:
            start_idx = output.find('{')
            end_idx = output.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = output[start_idx:end_idx]
                return json.loads(json_str)
        except json.JSONDecodeError:
            logger.warning("Failed to parse response output as JSON")
        
        return {
            "content": output.strip(),
            "tone": "professional",
            "key_points": [],
            "action_items": []
        }
    
    def _build_prompt(self, step: ChainStep, context: Dict[str, Any]) -> str:
        """Build the prompt by replacing placeholders with context values."""
        prompt = step.user_prompt_template
        
        for key, value in context.items():
            if isinstance(value, dict):
                for nested_key, nested_value in value.items():
                    placeholder = f"{{{key}.{nested_key}}}"
                    if placeholder in prompt:
                        prompt = prompt.replace(placeholder, str(nested_value))
            else:
                placeholder = f"{{{key}}}"
                if placeholder in prompt:
                    prompt = prompt.replace(placeholder, str(value))
        
        return prompt
    
    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Execute a single LLM call."""
        start_time = time.time()
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=self.temperature
        )
        
        elapsed = (time.time() - start_time) * 1000
        content = response.choices[0].message.content
        
        logger.debug(f"LLM call completed in {elapsed:.2f}ms")
        return content
    
    def execute(self, raw_input: str, parallel_analysis: Optional[Dict] = None) -> Dict[str, Any]:
        """Execute the entire prompt chain."""
        context = {
            "raw_input": raw_input,
            "parallel_analysis": parallel_analysis or {}
        }
        
        print(f"\n{'='*70}")
        print(f"[PROMPT CHAIN] Processing: {raw_input[:50]}...")
        print(f"{'='*70}")
        
        for step in self.steps:
            print(f"\n[Step] {step.name.upper()}")
            print("-" * 40)
            
            user_prompt = self._build_prompt(step, context)
            response = self._call_llm(step.system_prompt, user_prompt)
            
            parsed = step.output_parser(response)
            context[step.name] = parsed
            
            if step.name == "preprocessing":
                print(f"  Cleaned: {parsed.get('cleaned_message', '')[:60]}...")
                print(f"  Corrections: {len(parsed.get('corrections', []))} made")
            elif step.name == "classification":
                print(f"  Category: {parsed.get('category', 'unknown')}")
                print(f"  Urgency: {parsed.get('urgency', 'unknown')}")
                print(f"  Issue: {parsed.get('issue_type', 'unknown')[:40]}...")
            elif step.name == "response_generation":
                print(f"  Tone: {parsed.get('tone', 'professional')}")
                print(f"  Response length: {len(parsed.get('content', ''))} chars")
        
        print(f"\n[Prompt Chain Complete]")
        return context
