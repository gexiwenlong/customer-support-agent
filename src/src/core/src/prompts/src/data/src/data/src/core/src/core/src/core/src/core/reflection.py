"""
Reflection Pattern Implementation.

Implements a self-improvement loop where the LLM evaluates its own
response, provides critique, and generates an improved version.
The process runs for multiple iterations until quality meets threshold.
"""

import json
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import logging
import os

from openai import OpenAI
from dotenv import load_dotenv

from .models import ReflectionResult, DraftResponse

load_dotenv()
logger = logging.getLogger(__name__)


@dataclass
class ReflectionConfig:
    """Configuration for the reflection process."""
    max_iterations: int = 2
    quality_threshold: float = 3.5
    temperature_critique: float = 0.3
    temperature_improvement: float = 0.5
    verbose: bool = True


@dataclass
class CritiqueResult:
    """Result of a single critique evaluation."""
    overall_score: float
    criterion_scores: Dict[str, float]
    strengths: List[str]
    weaknesses: List[str]
    specific_improvements: List[str]
    requires_revision: bool
    raw_critique: str


class ReflectionLoop:
    """
    Self-improvement loop for response quality enhancement.
    """
    
    def __init__(self, model: str = "gpt-3.5-turbo", config: Optional[ReflectionConfig] = None):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.config = config or ReflectionConfig()
        self.iteration_history: List[Dict[str, Any]] = []
    
    def _call_critique(self, original_message: str, draft_response: str,
                       category: str, urgency: str, sentiment: str) -> CritiqueResult:
        """Call the LLM to critique a draft response."""
        from ..prompts import REFLECTION_CRITIQUE_PROMPT
        
        prompt = REFLECTION_CRITIQUE_PROMPT.format(
            original_message=original_message,
            category=category,
            urgency=urgency,
            sentiment=sentiment,
            draft_response=draft_response
        )
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.config.temperature_critique
        )
        
        content = response.choices[0].message.content
        
        try:
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                data = json.loads(content[start_idx:end_idx])
                
                return CritiqueResult(
                    overall_score=float(data.get("overall_score", 3.0)),
                    criterion_scores=data.get("criterion_scores", {}),
                    strengths=data.get("strengths", []),
                    weaknesses=data.get("weaknesses", []),
                    specific_improvements=data.get("specific_improvements", []),
                    requires_revision=data.get("requires_revision", True),
                    raw_critique=content
                )
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"Failed to parse critique response: {e}")
        
        return CritiqueResult(
            overall_score=3.0,
            criterion_scores={},
            strengths=["Response was generated"],
            weaknesses=["Unable to properly evaluate"],
            specific_improvements=["Review and improve the response"],
            requires_revision=True,
            raw_critique=content
        )
    
    def _call_improvement(self, original_message: str, draft_response: str,
                          critique: str, improvements: List[str]) -> str:
        """Call the LLM to generate an improved response based on critique."""
        from ..prompts import REFLECTION_IMPROVEMENT_PROMPT
        
        prompt = REFLECTION_IMPROVEMENT_PROMPT.format(
            original_message=original_message,
            draft_response=draft_response,
            critique=critique,
            improvements="\n".join(f"- {imp}" for imp in improvements)
        )
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.config.temperature_improvement
        )
        
        return response.choices[0].message.content
    
    def _compare_responses(self, original: str, improved: str) -> List[str]:
        """Identify what changed between original and improved responses."""
        changes = []
        
        orig_len = len(original)
        imp_len = len(improved)
        
        if abs(orig_len - imp_len) > 50:
            if imp_len > orig_len:
                changes.append(f"Response length increased from {orig_len} to {imp_len} chars (more detailed)")
            else:
                changes.append(f"Response length decreased from {orig_len} to {imp_len} chars (more concise)")
        
        tone_words = ["apologize", "sorry", "understand", "appreciate", "thank", "please"]
        orig_tone_count = sum(1 for word in tone_words if word in original.lower())
        imp_tone_count = sum(1 for word in tone_words if word in improved.lower())
        
        if imp_tone_count > orig_tone_count:
            changes.append("Added more empathetic language")
        elif imp_tone_count < orig_tone_count:
            changes.append("Made tone more direct and professional")
        
        if "1." in improved and "1." not in original:
            changes.append("Added numbered steps for clarity")
        if "\n\n" in improved and "\n\n" not in original:
            changes.append("Improved paragraph structure and readability")
        
        return changes if changes else ["Subtle refinements to wording and flow"]
    
    def reflect_and_improve(self, original_message: str, initial_response: str,
                            category: str, urgency: str, sentiment: str) -> ReflectionResult:
        """Execute the complete reflection and improvement loop."""
        print(f"\n{'='*70}")
        print(f"[REFLECTION LOOP] Starting self-improvement process")
        print(f"{'='*70}")
        print(f"  Max iterations: {self.config.max_iterations}")
        print(f"  Quality threshold: {self.config.quality_threshold}/5.0")
        
        current_response = initial_response
        iteration_results = []
        changes_made = []
        final_critique = None
        
        for iteration in range(1, self.config.max_iterations + 1):
            print(f"\n[Iteration {iteration}/{self.config.max_iterations}]")
            print("-" * 40)
            
            print("  [Critique] Evaluating current response...")
            critique = self._call_critique(
                original_message=original_message,
                draft_response=current_response,
                category=category,
                urgency=urgency,
                sentiment=sentiment
            )
            
            print(f"    Overall Score: {critique.overall_score:.1f}/5.0")
            print(f"    Strengths: {len(critique.strengths)} identified")
            print(f"    Weaknesses: {len(critique.weaknesses)} identified")
            print(f"    Improvements suggested: {len(critique.specific_improvements)}")
            
            iteration_data = {
                "iteration": iteration,
                "score": critique.overall_score,
                "weaknesses": critique.weaknesses,
                "improvements": critique.specific_improvements,
                "response_before": current_response[:200] + "..."
            }
            
            if not critique.requires_revision or critique.overall_score >= self.config.quality_threshold:
                print(f"\n  [Quality Threshold Met] Score {critique.overall_score:.1f} >= {self.config.quality_threshold}")
                final_critique = critique.raw_critique
                iteration_results.append(iteration_data)
                break
            
            print("\n  [Improvement] Generating enhanced response...")
            improved_response = self._call_improvement(
                original_message=original_message,
                draft_response=current_response,
                critique=critique.raw_critique,
                improvements=critique.specific_improvements
            )
            
            iteration_changes = self._compare_responses(current_response, improved_response)
            changes_made.extend(iteration_changes)
            
            print(f"    Changes identified: {len(iteration_changes)}")
            for change in iteration_changes[:3]:
                print(f"      - {change}")
            
            iteration_data["response_after"] = improved_response[:200] + "..."
            iteration_data["changes"] = iteration_changes
            iteration_results.append(iteration_data)
            
            current_response = improved_response
            final_critique = critique.raw_critique
        
        result = ReflectionResult(
            original_response=initial_response,
            critique=final_critique or "No critique generated",
            improved_response=current_response,
            changes_made=changes_made,
            iteration_count=len(iteration_results)
        )
        
        self.iteration_history.extend(iteration_results)
        
        print(f"\n{'='*70}")
        print(f"[REFLECTION COMPLETE]")
        print(f"{'='*70}")
        print(f"  Iterations performed: {len(iteration_results)}")
        print(f"  Total changes made: {len(changes_made)}")
        print(f"  Final response length: {len(current_response)} chars")
        
        if len(initial_response) > 0 and len(current_response) > 0:
            print(f"\n[Comparison] Original vs Improved")
            print("-" * 40)
            print(f"Original (first 150 chars): {initial_response[:150]}...")
            print(f"Improved (first 150 chars): {current_response[:150]}...")
        
        return result
    
    def get_improvement_summary(self) -> Dict[str, Any]:
        """Get a summary of the improvement process."""
        if not self.iteration_history:
            return {"message": "No reflection history available"}
        
        scores = [item["score"] for item in self.iteration_history if "score" in item]
        
        return {
            "total_iterations": len(self.iteration_history),
            "initial_score": scores[0] if scores else None,
            "final_score": scores[-1] if scores else None,
            "score_improvement": scores[-1] - scores[0] if len(scores) >= 2 else 0,
            "changes_made": sum(len(item.get("changes", [])) for item in self.iteration_history)
        }


class QualityEvaluator:
    """Standalone quality evaluation for responses."""
    
    def __init__(self, model: str = "gpt-3.5-turbo"):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        self.client = OpenAI(api_key=api_key)
        self.model = model
    
    def evaluate_response(self, original_message: str, response: str,
                          category: str = "general") -> Dict[str, Any]:
        """Perform a single quality evaluation of a response."""
        prompt = f"""Evaluate this customer support response quality (1-5 scale):

Original customer message: {original_message}
Category: {category}

Response to evaluate:
---
{response}
---

Evaluate on:
1. Accuracy: Does it correctly address the issue?
2. Completeness: Is all necessary information included?
3. Tone: Is it appropriate and professional?
4. Clarity: Is it easy to understand?
5. Helpfulness: Will this actually help the customer?

Return JSON with scores and brief feedback:
{{
    "accuracy": 1-5,
    "completeness": 1-5,
    "tone": 1-5,
    "clarity": 1-5,
    "helpfulness": 1-5,
    "overall": 1-5,
    "feedback": "brief constructive feedback"
}}"""
        
        llm_response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        content = llm_response.choices[0].message.content
        
        try:
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                return json.loads(content[start_idx:end_idx])
        except json.JSONDecodeError:
            pass
        
        return {
            "overall": 3.0,
            "feedback": "Unable to properly evaluate response",
            "error": True
        }
