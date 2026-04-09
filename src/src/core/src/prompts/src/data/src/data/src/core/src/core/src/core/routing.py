"""
Routing Pattern Implementation.

Dynamically routes tickets to specialized processing branches based on
classification results. Each branch has its own prompt template and
processing logic tailored to the ticket type.
"""

import json
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import logging
import os

from openai import OpenAI
from dotenv import load_dotenv

from .models import TicketCategory, ClassificationResult, DraftResponse

load_dotenv()
logger = logging.getLogger(__name__)


class RouteType(str, Enum):
    """Available routing branches."""
    TECHNICAL = "technical"
    BILLING = "billing"
    GENERAL = "general"
    COMPLAINT = "complaint"


@dataclass
class RouteConfig:
    """Configuration for a single route."""
    name: str
    category: TicketCategory
    prompt_template: str
    description: str
    requires_escalation: bool = False
    default_tone: str = "professional"


class TicketRouter:
    """
    Routes tickets to specialized processing branches.
    
    Based on classification results, selects the appropriate branch
    and generates a tailored response using branch-specific prompts.
    """
    
    def __init__(self, model: str = "gpt-3.5-turbo", temperature: float = 0.5):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.routes: Dict[TicketCategory, RouteConfig] = {}
    
    def _setup_default_routes(self) -> None:
        """Configure the default routing branches."""
        from ..prompts import (
            TECHNICAL_BRANCH_PROMPT, BILLING_BRANCH_PROMPT,
            GENERAL_BRANCH_PROMPT, COMPLAINT_BRANCH_PROMPT
        )
        
        self.routes[TicketCategory.TECHNICAL] = RouteConfig(
            name="technical_support",
            category=TicketCategory.TECHNICAL,
            prompt_template=TECHNICAL_BRANCH_PROMPT,
            description="Technical issues requiring troubleshooting steps",
            default_tone="technical"
        )
        
        self.routes[TicketCategory.BILLING] = RouteConfig(
            name="billing_support",
            category=TicketCategory.BILLING,
            prompt_template=BILLING_BRANCH_PROMPT,
            description="Billing inquiries, refunds, and payment issues",
            default_tone="professional"
        )
        
        self.routes[TicketCategory.GENERAL] = RouteConfig(
            name="general_inquiry",
            category=TicketCategory.GENERAL,
            prompt_template=GENERAL_BRANCH_PROMPT,
            description="General information requests and FAQs",
            default_tone="friendly"
        )
        
        self.routes[TicketCategory.COMPLAINT] = RouteConfig(
            name="complaint_escalation",
            category=TicketCategory.COMPLAINT,
            prompt_template=COMPLAINT_BRANCH_PROMPT,
            description="Complaints and escalations requiring special handling",
            requires_escalation=True,
            default_tone="empathetic"
        )
    
    def add_route(self, route: RouteConfig) -> 'TicketRouter':
        """Add a custom route configuration."""
        self.routes[route.category] = route
        return self
    
    def determine_route(self, classification: ClassificationResult) -> RouteConfig:
        """Determine which route to take based on classification."""
        self._setup_default_routes()
        
        category = classification.category
        
        if category in self.routes:
            return self.routes[category]
        
        logger.warning(f"No route for category {category}, falling back to general")
        return self.routes[TicketCategory.GENERAL]
    
    def _build_branch_prompt(self, route: RouteConfig, context: Dict[str, Any]) -> str:
        """Build the branch-specific prompt by interpolating context values."""
        prompt = route.prompt_template
        
        replacements = {
            "{issue_type}": str(context.get("issue_type", "general inquiry")),
            "{product_name}": str(context.get("product_name", "your product")),
            "{urgency}": str(context.get("urgency", "medium")),
            "{key_entities}": ", ".join(context.get("key_entities", [])),
            "{sentiment}": str(context.get("sentiment", "neutral")),
            "{original_message}": str(context.get("original_message", ""))
        }
        
        for placeholder, value in replacements.items():
            prompt = prompt.replace(placeholder, value)
        
        return prompt
    
    def _call_llm(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Execute an LLM call for a specific branch."""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature
        )
        
        return response.choices[0].message.content
    
    def process_branch(self, route: RouteConfig, classification: ClassificationResult,
                       original_message: str, draft_response: Optional[DraftResponse] = None,
                       sentiment: str = "neutral") -> Dict[str, Any]:
        """Process a ticket through its assigned branch."""
        print(f"\n{'='*70}")
        print(f"[ROUTING] Branch Selected: {route.name.upper()}")
        print(f"{'='*70}")
        print(f"  Description: {route.description}")
        print(f"  Requires Escalation: {route.requires_escalation}")
        print(f"  Default Tone: {route.default_tone}")
        
        context = {
            "issue_type": classification.issue_type or "general inquiry",
            "product_name": classification.product_name or "your product",
            "urgency": classification.urgency.value if classification.urgency else "medium",
            "key_entities": classification.key_entities,
            "sentiment": sentiment,
            "original_message": original_message
        }
        
        branch_prompt = self._build_branch_prompt(route, context)
        
        system_prompts = {
            TicketCategory.TECHNICAL: "You are a technical support specialist. Provide clear, actionable troubleshooting steps.",
            TicketCategory.BILLING: "You are a billing support specialist. Be professional and precise with financial matters.",
            TicketCategory.GENERAL: "You are a customer service representative. Be friendly and helpful.",
            TicketCategory.COMPLAINT: "You are a senior customer service manager. Be empathetic and solution-oriented."
        }
        
        system_prompt = system_prompts.get(route.category, "You are a helpful customer support representative.")
        
        start_time = time.time()
        branch_response = self._call_llm(branch_prompt, system_prompt)
        elapsed_ms = (time.time() - start_time) * 1000
        
        print(f"\n[Branch Response Generated] ({elapsed_ms:.2f}ms)")
        print("-" * 40)
        print(f"Response preview: {branch_response[:150]}...")
        
        return {
            "route_name": route.name,
            "route_category": route.category.value,
            "requires_escalation": route.requires_escalation,
            "branch_response": branch_response,
            "processing_time_ms": elapsed_ms
        }
    
    def route_and_process(self, classification: ClassificationResult,
                          original_message: str,
                          sentiment: str = "neutral") -> Dict[str, Any]:
        """Complete routing workflow: determine route and process through branch."""
        route = self.determine_route(classification)
        
        result = self.process_branch(
            route=route,
            classification=classification,
            original_message=original_message,
            sentiment=sentiment
        )
        
        return result


class RoutingDecisionLogger:
    """Utility for logging and analyzing routing decisions."""
    
    def __init__(self):
        self.decisions: List[Dict[str, Any]] = []
    
    def log_decision(self, ticket_id: str, original_message: str,
                     classification: ClassificationResult,
                     selected_route: RouteConfig) -> None:
        """Log a routing decision for later analysis."""
        self.decisions.append({
            "ticket_id": ticket_id,
            "timestamp": time.time(),
            "message_preview": original_message[:100],
            "classified_category": classification.category.value,
            "classified_urgency": classification.urgency.value if classification.urgency else "unknown",
            "selected_route": selected_route.name,
            "reasoning": classification.reasoning
        })
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about routing decisions."""
        if not self.decisions:
            return {"total_decisions": 0}
        
        route_counts = {}
        category_counts = {}
        
        for decision in self.decisions:
            route = decision["selected_route"]
            category = decision["classified_category"]
            
            route_counts[route] = route_counts.get(route, 0) + 1
            category_counts[category] = category_counts.get(category, 0) + 1
        
        return {
            "total_decisions": len(self.decisions),
            "route_distribution": route_counts,
            "category_distribution": category_counts
        }
