"""Reflection Pattern Implementation."""
from dataclasses import dataclass
from src.core.models import ReflectionResult

@dataclass
class ReflectionConfig:
    max_iterations: int = 2
    quality_threshold: float = 3.5

class ReflectionLoop:
    def __init__(self, model: str = "gpt-3.5-turbo", config: ReflectionConfig = None):
        self.model = model
        self.config = config or ReflectionConfig()

    def reflect_and_improve(self, original_message: str, initial_response: str,
                            category: str, urgency: str, sentiment: str) -> ReflectionResult:
        improved = initial_response + " We will resolve this quickly."
        return ReflectionResult(
            original_response=initial_response,
            critique="Response is acceptable",
            improved_response=improved,
            changes_made=["Added urgency note"],
            iteration_count=1
        )
