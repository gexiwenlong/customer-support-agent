"""
Core module containing all agentic design pattern implementations.

This package includes:
- models: Data structures for tickets and processing results
- prompt_chain: Sequential pipeline processing pattern
- parallelization: Concurrent task execution pattern
- routing: Dynamic branch selection pattern
- reflection: Self-improvement loop pattern
"""

from .models import (
    TicketCategory,
    UrgencyLevel,
    SentimentType,
    PreprocessedTicket,
    ClassificationResult,
    SentimentResult,
    ParallelAnalysisResult,
    DraftResponse,
    ReflectionResult,
    ProcessedTicket,
    ProcessingStatistics
)

from .prompt_chain import PromptChain, ChainStep
from .parallelization import ParallelExecutor, BatchParallelProcessor
from .routing import TicketRouter, RouteConfig, RouteType
from .reflection import ReflectionLoop, ReflectionConfig, QualityEvaluator

__all__ = [
    # Models
    'TicketCategory',
    'UrgencyLevel',
    'SentimentType',
    'PreprocessedTicket',
    'ClassificationResult',
    'SentimentResult',
    'ParallelAnalysisResult',
    'DraftResponse',
    'ReflectionResult',
    'ProcessedTicket',
    'ProcessingStatistics',
    # Patterns
    'PromptChain',
    'ChainStep',
    'ParallelExecutor',
    'BatchParallelProcessor',
    'TicketRouter',
    'RouteConfig',
    'RouteType',
    'ReflectionLoop',
    'ReflectionConfig',
    'QualityEvaluator',
]
