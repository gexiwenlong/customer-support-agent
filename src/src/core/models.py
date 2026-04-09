"""
Core data models for the customer support ticket processor.

This module defines the structure for tickets, classifications,
and all intermediate processing results.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum
from datetime import datetime


class TicketCategory(str, Enum):
    """Enumeration of possible ticket categories."""
    TECHNICAL = "technical"
    BILLING = "billing"
    GENERAL = "general"
    COMPLAINT = "complaint"


class UrgencyLevel(str, Enum):
    """Enumeration of urgency levels."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class SentimentType(str, Enum):
    """Enumeration of sentiment types."""
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


@dataclass
class PreprocessedTicket:
    """Result of the preprocessing step."""
    original_message: str
    cleaned_message: str
    corrected_spelling: List[Dict[str, str]] = field(default_factory=list)
    expanded_abbreviations: List[Dict[str, str]] = field(default_factory=list)
    processing_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ClassificationResult:
    """Result of the classification step."""
    category: TicketCategory
    urgency: UrgencyLevel
    product_name: Optional[str] = None
    issue_type: Optional[str] = None
    key_entities: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    reasoning: str = ""


@dataclass
class SentimentResult:
    """Result of sentiment analysis."""
    sentiment: SentimentType
    score: float
    key_phrases: List[str] = field(default_factory=list)


@dataclass
class ParallelAnalysisResult:
    """Container for parallel analysis results."""
    sentiment: SentimentResult
    keywords: List[str]
    priority_score: int
    language: str
    execution_time_ms: float = 0.0


@dataclass
class DraftResponse:
    """Container for draft response data."""
    content: str
    tone: str
    key_points: List[str] = field(default_factory=list)
    action_items: List[str] = field(default_factory=list)


@dataclass
class ReflectionResult:
    """Result of the reflection/self-improvement step."""
    original_response: str
    critique: str
    improved_response: str
    changes_made: List[str] = field(default_factory=list)
    iteration_count: int = 0


@dataclass
class ProcessedTicket:
    """Complete processing result for a single ticket."""
    ticket_id: str
    original_message: str
    preprocessed: PreprocessedTicket
    classification: ClassificationResult
    parallel_analysis: ParallelAnalysisResult
    initial_response: DraftResponse
    reflection: ReflectionResult
    final_response: str
    route_taken: str
    processing_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    total_processing_time_ms: float = 0.0


@dataclass
class ProcessingStatistics:
    """Statistics for the processing run."""
    total_tickets: int
    successful_tickets: int
    failed_tickets: int
    average_processing_time_ms: float
    category_distribution: Dict[str, int] = field(default_factory=dict)
    route_distribution: Dict[str, int] = field(default_factory=dict)
    reflection_improvements: List[Dict[str, Any]] = field(default_factory=list)
