"""
Parallelization Pattern Implementation.

Executes multiple independent LLM tasks concurrently to reduce total
processing time. Tasks include sentiment analysis, keyword extraction,
priority scoring, and language detection.
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging
import os

from openai import AsyncOpenAI
from dotenv import load_dotenv

from .models import ParallelAnalysisResult, SentimentResult, SentimentType

load_dotenv()
logger = logging.getLogger(__name__)


@dataclass
class ParallelTask:
    """Definition of a single parallel task."""
    name: str
    prompt_template: str
    output_parser: callable
    required_inputs: List[str]


class ParallelExecutor:
    """
    Executes multiple LLM tasks concurrently.
    
    Uses asyncio.gather to run independent API calls in parallel,
    significantly reducing total processing time for batch operations.
    """
    
    def __init__(self, model: str = "gpt-3.5-turbo", temperature: float = 0.3):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.tasks: List[ParallelTask] = []
    
    def add_task(self, task: ParallelTask) -> 'ParallelExecutor':
        """Add a custom parallel task."""
        self.tasks.append(task)
        return self
    
    def _parse_sentiment_output(self, output: str) -> SentimentResult:
        """Parse sentiment analysis output."""
        try:
            start_idx = output.find('{')
            end_idx = output.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                data = json.loads(output[start_idx:end_idx])
                sentiment_str = data.get("sentiment", "neutral").lower()
                
                sentiment_map = {
                    "positive": SentimentType.POSITIVE,
                    "neutral": SentimentType.NEUTRAL,
                    "negative": SentimentType.NEGATIVE
                }
                
                return SentimentResult(
                    sentiment=sentiment_map.get(sentiment_str, SentimentType.NEUTRAL),
                    score=float(data.get("score", 0.5)),
                    key_phrases=data.get("key_phrases", [])
                )
        except (json.JSONDecodeError, KeyError, ValueError):
            logger.warning("Failed to parse sentiment output")
        
        return SentimentResult(
            sentiment=SentimentType.NEUTRAL,
            score=0.5,
            key_phrases=[]
        )
    
    def _parse_keywords_output(self, output: str) -> List[str]:
        """Parse keyword extraction output."""
        try:
            start_idx = output.find('[')
            end_idx = output.rfind(']') + 1
            if start_idx != -1 and end_idx > start_idx:
                return json.loads(output[start_idx:end_idx])
        except json.JSONDecodeError:
            pass
        
        import re
        words = re.findall(r'"?([^",\n]+)"?', output)
        return words[:10] if words else []
    
    def _parse_priority_output(self, output: str) -> int:
        """Parse priority score output."""
        import re
        numbers = re.findall(r'\d+', output)
        if numbers:
            score = int(numbers[0])
            return max(1, min(5, score))
        return 3
    
    def _parse_language_output(self, output: str) -> str:
        """Parse language detection output."""
        return output.strip() or "English"
    
    async def _execute_single_task(self, task: ParallelTask, inputs: Dict[str, str]) -> Tuple[str, Any]:
        """Execute a single task asynchronously."""
        start_time = time.time()
        
        prompt = task.prompt_template
        for key, value in inputs.items():
            prompt = prompt.replace(f"{{{key}}}", str(value))
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature
            )
            
            elapsed_ms = (time.time() - start_time) * 1000
            content = response.choices[0].message.content
            
            logger.debug(f"Task '{task.name}' completed in {elapsed_ms:.2f}ms")
            
            parsed = task.output_parser(content)
            return task.name, parsed
            
        except Exception as e:
            logger.error(f"Task '{task.name}' failed: {e}")
            return task.name, self._get_fallback_output(task.name)
    
    def _get_fallback_output(self, task_name: str) -> Any:
        """Provide fallback output when a task fails."""
        fallbacks = {
            "sentiment": SentimentResult(sentiment=SentimentType.NEUTRAL, score=0.5, key_phrases=[]),
            "keywords": [],
            "priority": 3,
            "language": "English"
        }
        return fallbacks.get(task_name, None)
    
    async def execute_async(self, message: str, category: str = "general") -> ParallelAnalysisResult:
        """Execute all parallel tasks concurrently."""
        from ..prompts import (
            SENTIMENT_ANALYSIS_PROMPT, KEYWORD_EXTRACTION_PROMPT,
            PRIORITY_SCORING_PROMPT, LANGUAGE_DETECTION_PROMPT
        )
        
        # Setup default tasks
        self.tasks = [
            ParallelTask(
                name="sentiment",
                prompt_template=SENTIMENT_ANALYSIS_PROMPT,
                output_parser=self._parse_sentiment_output,
                required_inputs=["message"]
            ),
            ParallelTask(
                name="keywords",
                prompt_template=KEYWORD_EXTRACTION_PROMPT,
                output_parser=self._parse_keywords_output,
                required_inputs=["message"]
            ),
            ParallelTask(
                name="priority",
                prompt_template=PRIORITY_SCORING_PROMPT,
                output_parser=self._parse_priority_output,
                required_inputs=["message", "category"]
            ),
            ParallelTask(
                name="language",
                prompt_template=LANGUAGE_DETECTION_PROMPT,
                output_parser=self._parse_language_output,
                required_inputs=["message"]
            ),
        ]
        
        print(f"\n{'='*70}")
        print(f"[PARALLEL EXECUTION] Starting {len(self.tasks)} concurrent tasks")
        print(f"{'='*70}")
        
        start_time = time.time()
        
        inputs = {"message": message, "category": category}
        
        coroutines = [
            self._execute_single_task(task, inputs)
            for task in self.tasks
        ]
        
        results = await asyncio.gather(*coroutines)
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        result_dict = dict(results)
        
        print(f"\n[Parallel Tasks Complete] Total time: {elapsed_ms:.2f}ms")
        print("-" * 40)
        
        sentiment_result = result_dict.get("sentiment")
        if isinstance(sentiment_result, SentimentResult):
            print(f"  Sentiment: {sentiment_result.sentiment.value}")
        else:
            print(f"  Sentiment: neutral")
        print(f"  Keywords: {len(result_dict.get('keywords', []))} extracted")
        print(f"  Priority: {result_dict.get('priority', 3)}/5")
        print(f"  Language: {result_dict.get('language', 'English')}")
        
        return ParallelAnalysisResult(
            sentiment=result_dict.get("sentiment", SentimentResult(sentiment=SentimentType.NEUTRAL, score=0.5)),
            keywords=result_dict.get("keywords", []),
            priority_score=result_dict.get("priority", 3),
            language=result_dict.get("language", "English"),
            execution_time_ms=elapsed_ms
        )
    
    def execute(self, message: str, category: str = "general") -> ParallelAnalysisResult:
        """Synchronous wrapper for parallel execution."""
        return asyncio.run(self.execute_async(message, category))


class BatchParallelProcessor:
    """
    Process multiple tickets in parallel.
    """
    
    def __init__(self, model: str = "gpt-3.5-turbo"):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
    
    async def _process_single_ticket(self, ticket: Dict[str, str], executor: ParallelExecutor) -> Dict[str, Any]:
        """Process a single ticket asynchronously."""
        ticket_id = ticket.get("id", "unknown")
        message = ticket.get("message", "")
        category = ticket.get("expected_category", "general")
        
        logger.info(f"Processing ticket {ticket_id}...")
        
        result = await executor.execute_async(message, category)
        
        return {
            "ticket_id": ticket_id,
            "message": message,
            "category": category,
            "analysis": result
        }
    
    async def process_batch_async(self, tickets: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Process multiple tickets in parallel."""
        print(f"\n{'='*70}")
        print(f"[BATCH PROCESSING] Processing {len(tickets)} tickets in parallel")
        print(f"{'='*70}")
        
        start_time = time.time()
        
        executor = ParallelExecutor(model=self.model)
        
        coroutines = [
            self._process_single_ticket(ticket, executor)
            for ticket in tickets
        ]
        
        results = await asyncio.gather(*coroutines, return_exceptions=True)
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        successful = [r for r in results if not isinstance(r, Exception)]
        failed = len(results) - len(successful)
        
        print(f"\n[Batch Complete] {elapsed_ms:.2f}ms total")
        print(f"  Successful: {len(successful)} tickets")
        print(f"  Failed: {failed} tickets")
        if len(tickets) > 0:
            print(f"  Average per ticket: {elapsed_ms/len(tickets):.2f}ms")
        
        return successful
    
    def process_batch(self, tickets: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Synchronous wrapper for batch processing."""
        return asyncio.run(self.process_batch_async(tickets))
