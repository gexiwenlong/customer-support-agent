"""Prompt Chaining Pattern Implementation."""
import json
import time
from typing import Dict, Any, Callable
from dataclasses import dataclass
import os
from openai import OpenAI

@dataclass
class ChainStep:
    name: str
    system_prompt: str
    user_prompt_template: str
    output_parser: Callable
    required_context: list

class PromptChain:
    def __init__(self, model: str = "gpt-3.5-turbo", temperature: float = 0.3):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.temperature = temperature
        self.steps = []

    def add_step(self, step: ChainStep):
        self.steps.append(step)
        return self

    def _parse_preprocessing_output(self, output: str):
        try:
            start = output.find('{')
            end = output.rfind('}') + 1
            if start != -1 and end > start:
                return json.loads(output[start:end])
        except:
            pass
        return {"cleaned_message": output.strip(), "corrections": []}

    def _parse_classification_output(self, output: str):
        try:
            start = output.find('{')
            end = output.rfind('}') + 1
            if start != -1 and end > start:
                data = json.loads(output[start:end])
                from src.core.models import TicketCategory, UrgencyLevel
                return {
                    "category": TicketCategory.TECHNICAL,
                    "urgency": UrgencyLevel.HIGH,
                    "product_name": data.get("product_name"),
                    "issue_type": data.get("issue_type", "unknown"),
                    "key_entities": data.get("key_entities", []),
                    "reasoning": data.get("reasoning", "")
                }
        except:
            pass
        from src.core.models import TicketCategory, UrgencyLevel
        return {"category": TicketCategory.GENERAL, "urgency": UrgencyLevel.MEDIUM,
                "product_name": None, "issue_type": "unknown", "key_entities": [], "reasoning": ""}

    def _parse_response_output(self, output: str):
        try:
            start = output.find('{')
            end = output.rfind('}') + 1
            if start != -1 and end > start:
                return json.loads(output[start:end])
        except:
            pass
        return {"content": output.strip(), "tone": "professional", "key_points": [], "action_items": []}

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": user_prompt}],
            temperature=self.temperature
        )
        return response.choices[0].message.content

    def execute(self, raw_input: str, parallel_analysis: Dict = None) -> Dict[str, Any]:
        context = {"raw_input": raw_input, "parallel_analysis": parallel_analysis or {}}
        for step in self.steps:
            prompt = step.user_prompt_template
            for key, value in context.items():
                if isinstance(value, dict):
                    for nk, nv in value.items():
                        prompt = prompt.replace(f"{{{key}.{nk}}}", str(nv))
                else:
                    prompt = prompt.replace(f"{{{key}}}", str(value))
            output = self._call_llm(step.system_prompt, prompt)
            context[step.name] = step.output_parser(output)
        return context
