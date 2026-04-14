"""Core data models for the customer support ticket processor."""
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum
from datetime import datetime

class TicketCategory(str, Enum):
    TECHNICAL = "technical"
    BILLING = "billing"
    GENERAL = "general"
    COMPLAINT = "complaint"

class UrgencyLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class SentimentType(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"

@dataclass
class PreprocessedTicket:
    original_message: str
    cleaned_message: str
    corrected_spelling: List[Dict[str, str]] = field(default_factory=list)
    expanded_abbreviations: List[Dict[str, str]] = field(default_factory=list)
    processing_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

@dataclass
class ClassificationResult:
    category: TicketCategory
    urgency: UrgencyLevel
    product_name: Optional[str] = None
    issue_type: Optional[str] = None
    key_entities: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    reasoning: str = ""

@dataclass
class SentimentResult:
    sentiment: SentimentType
    score: float
    key_phrases: List[str] = field(default_factory=list)

@dataclass
class ParallelAnalysisResult:
    sentiment: SentimentResult
    keywords: List[str]
    priority_score: int
    language: str
    execution_time_ms: float = 0.0

@dataclass
class DraftResponse:
    content: str
    tone: str
    key_points: List[str] = field(default_factory=list)
    action_items: List[str] = field(default_factory=list)

@dataclass
class ReflectionResult:
    original_response: str
    critique: str
    improved_response: str
    changes_made: List[str] = field(default_factory=list)
    iteration_count: int = 0

@dataclass
class ProcessedTicket:
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
    total_tickets: int
    successful_tickets: int
    failed_tickets: int
    average_processing_time_ms: float
    category_distribution: Dict[str, int] = field(default_factory=dict)
    route_distribution: Dict[str, int] = field(default_factory=dict)
