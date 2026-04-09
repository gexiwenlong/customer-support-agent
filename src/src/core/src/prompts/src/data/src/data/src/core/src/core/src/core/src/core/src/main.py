"""
Customer Support Ticket Processor - Main Application

Integrates all four agentic design patterns:
1. Prompt Chaining - Sequential processing pipeline
2. Routing - Dynamic branch selection
3. Parallelization - Concurrent task execution
4. Reflection - Self-improvement loop

Usage:
    python -m src.main --input src/data/sample_tickets.json --output output/results.json
"""

import json
import time
import argparse
from typing import Dict, List, Any
from pathlib import Path
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.models import (
    ProcessedTicket, PreprocessedTicket, ClassificationResult,
    ParallelAnalysisResult, DraftResponse, ReflectionResult,
    ProcessingStatistics, TicketCategory, UrgencyLevel, SentimentType
)
from src.core.prompt_chain import PromptChain, ChainStep
from src.core.parallelization import ParallelExecutor
from src.core.routing import TicketRouter
from src.core.reflection import ReflectionLoop, ReflectionConfig
from src.prompts import (
    PREPROCESSING_SYSTEM, PREPROCESSING_PROMPT,
    CLASSIFICATION_SYSTEM, CLASSIFICATION_PROMPT,
    RESPONSE_GENERATION_SYSTEM, RESPONSE_GENERATION_PROMPT
)


class TicketProcessor:
    """
    Main ticket processor that orchestrates all patterns.
    """
    
    def __init__(self, model: str = "gpt-3.5-turbo"):
        self.model = model
        
        self.parallel_executor = ParallelExecutor(model=model)
        self.router = TicketRouter(model=model)
        self.reflection_loop = ReflectionLoop(
            model=model,
            config=ReflectionConfig(max_iterations=2, quality_threshold=3.5)
        )
        
        self.processing_stats = []
    
    def _build_prompt_chain(self) -> PromptChain:
        """Build the prompt chain with all steps."""
        chain = PromptChain(model=self.model)

        chain.add_step(ChainStep(
            name="preprocessing",
            system_prompt=PREPROCESSING_SYSTEM,
            user_prompt_template=PREPROCESSING_PROMPT,
            output_parser=chain._parse_preprocessing_output,
            required_context=["raw_input"]
        ))
      
        chain.add_step(ChainStep(
            name="classification",
            system_prompt=CLASSIFICATION_SYSTEM,
            user_prompt_template=CLASSIFICATION_PROMPT,
            output_parser=chain._parse_classification_output,
            required_context=["raw_input", "preprocessing"]
        ))
        
        chain.add_step(ChainStep(
            name="response_generation",
            system_prompt=RESPONSE_GENERATION_SYSTEM,
            user_prompt_template=RESPONSE_GENERATION_PROMPT,
            output_parser=chain._parse_response_output,
            required_context=["raw_input", "classification", "parallel_analysis"]
        ))
        
        return chain
    
    def process_ticket(self, ticket_id: str, message: str) -> ProcessedTicket:
        """Process a single ticket through all patterns."""
        start_time = time.time()
        
        print(f"\n{'#'*80}")
        print(f"# PROCESSING TICKET: {ticket_id}")
        print(f"# Message: {message[:80]}...")
        print(f"{'#'*80}")
        
        print(f"\n[PHASE 1] Parallel Analysis")
        parallel_result = self.parallel_executor.execute(message, "general")
        
        parallel_dict = {
            "sentiment": parallel_result.sentiment.sentiment.value,
            "keywords": parallel_result.keywords,
            "priority": parallel_result.priority_score,
            "language": parallel_result.language
        }
        
        print(f"\n[PHASE 2] Prompt Chaining Pipeline")
        chain = self._build_prompt_chain()
        chain_context = chain.execute(
            raw_input=message,
            parallel_analysis=parallel_dict
        )
        
        preprocessed_data = chain_context.get("preprocessing", {})
        preprocessed = PreprocessedTicket(
            original_message=message,
            cleaned_message=preprocessed_data.get("cleaned_message", message),
            corrected_spelling=[c for c in preprocessed_data.get("corrections", []) if "corrected" in c],
            expanded_abbreviations=[c for c in preprocessed_data.get("corrections", []) if "expanded_to" in c]
        )
        
        class_data = chain_context.get("classification", {})
        classification = ClassificationResult(
            category=class_data.get("category", TicketCategory.GENERAL),
            urgency=class_data.get("urgency", UrgencyLevel.MEDIUM),
            product_name=class_data.get("product_name"),
            issue_type=class_data.get("issue_type"),
            key_entities=class_data.get("key_entities", []),
            reasoning=class_data.get("reasoning", ""),
            confidence_score=parallel_result.priority_score / 5.0
        )
        
        resp_data = chain_context.get("response_generation", {})
        draft_response = DraftResponse(
            content=resp_data.get("content", ""),
            tone=resp_data.get("tone", "professional"),
            key_points=resp_data.get("key_points", []),
            action_items=resp_data.get("action_items", [])
        )
        
        print(f"\n[PHASE 3] Routing to Specialized Branch")
        route_result = self.router.route_and_process(
            classification=classification,
            original_message=message,
            sentiment=parallel_result.sentiment.sentiment.value
        )
        
        print(f"\n[PHASE 4] Reflection and Improvement")
        initial_response = route_result["branch_response"]
        
        reflection_result = self.reflection_loop.reflect_and_improve(
            original_message=message,
            initial_response=initial_response,
            category=classification.category.value,
            urgency=classification.urgency.value if classification.urgency else "medium",
            sentiment=parallel_result.sentiment.sentiment.value
        )
        
        total_time_ms = (time.time() - start_time) * 1000
        
        processed = ProcessedTicket(
            ticket_id=ticket_id,
            original_message=message,
            preprocessed=preprocessed,
            classification=classification,
            parallel_analysis=parallel_result,
            initial_response=draft_response,
            reflection=reflection_result,
            final_response=reflection_result.improved_response,
            route_taken=route_result["route_name"],
            total_processing_time_ms=total_time_ms
        )
        
        self.processing_stats.append({
            "ticket_id": ticket_id,
            "processing_time_ms": total_time_ms,
            "route": route_result["route_name"],
            "reflection_iterations": reflection_result.iteration_count
        })
        
        print(f"\n{'#'*80}")
        print(f"# TICKET {ticket_id} PROCESSING COMPLETE")
        print(f"# Total time: {total_time_ms:.2f}ms")
        print(f"# Route: {route_result['route_name']}")
        print(f"# Final response length: {len(reflection_result.improved_response)} chars")
        print(f"{'#'*80}")
        
        return processed
    
    def process_batch(self, tickets: List[Dict[str, str]]) -> List[ProcessedTicket]:
        """Process multiple tickets sequentially."""
        results = []
        
        for i, ticket in enumerate(tickets):
            ticket_id = ticket.get("id", f"TICKET-{i+1:03d}")
            message = ticket.get("message", "")
            
            print(f"\n{'='*80}")
            print(f"BATCH PROGRESS: {i+1}/{len(tickets)} tickets")
            print(f"{'='*80}")
            
            try:
                processed = self.process_ticket(ticket_id, message)
                results.append(processed)
            except Exception as e:
                logger.error(f"Failed to process ticket {ticket_id}: {e}")
                continue
        
        return results
    
    def get_statistics(self) -> ProcessingStatistics:
        """Generate statistics for the processing run."""
        if not self.processing_stats:
            return ProcessingStatistics(
                total_tickets=0,
                successful_tickets=0,
                failed_tickets=0,
                average_processing_time_ms=0.0
            )
        
        total = len(self.processing_stats)
        avg_time = sum(s["processing_time_ms"] for s in self.processing_stats) / total
        
        route_counts = {}
        for stat in self.processing_stats:
            route = stat["route"]
            route_counts[route] = route_counts.get(route, 0) + 1
        
        return ProcessingStatistics(
            total_tickets=total,
            successful_tickets=total,
            failed_tickets=0,
            average_processing_time_ms=avg_time,
            route_distribution=route_counts
        )


def load_tickets(file_path: str) -> List[Dict[str, str]]:
    """Load tickets from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, dict) and "tickets" in data:
        return data["tickets"]
    elif isinstance(data, list):
        return data
    else:
        raise ValueError("Invalid ticket data format")


def save_results(results: List[ProcessedTicket], output_path: str) -> None:
    """Save processing results to JSON file."""
    output_data = []
    
    for ticket in results:
        output_data.append({
            "ticket_id": ticket.ticket_id,
            "original_message": ticket.original_message,
            "cleaned_message": ticket.preprocessed.cleaned_message,
            "category": ticket.classification.category.value if ticket.classification.category else "unknown",
            "urgency": ticket.classification.urgency.value if ticket.classification.urgency else "unknown",
            "sentiment": ticket.parallel_analysis.sentiment.sentiment.value,
            "priority_score": ticket.parallel_analysis.priority_score,
            "route_taken": ticket.route_taken,
            "initial_response": ticket.initial_response.content,
            "reflection_critique": ticket.reflection.critique[:500] if ticket.reflection.critique else "",
            "final_response": ticket.final_response,
            "changes_made": ticket.reflection.changes_made,
            "reflection_iterations": ticket.reflection.iteration_count,
            "processing_time_ms": ticket.total_processing_time_ms
        })
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n[Results saved to {output_path}]")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Customer Support Ticket Processor')
    parser.add_argument('--input', '-i', default='src/data/sample_tickets.json',
                        help='Path to input JSON file with tickets')
    parser.add_argument('--output', '-o', default='output/results.json',
                        help='Path to output JSON file for results')
    parser.add_argument('--model', '-m', default='gpt-3.5-turbo',
                        help='OpenAI model to use')
    parser.add_argument('--limit', '-l', type=int, default=None,
                        help='Limit number of tickets to process')
    
    args = parser.parse_args()
    
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("CUSTOMER SUPPORT TICKET PROCESSOR")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    
    tickets = load_tickets(args.input)
    if args.limit:
        tickets = tickets[:args.limit]
    
    print(f"\nLoaded {len(tickets)} tickets for processing")
    
    processor = TicketProcessor(model=args.model)
    
    start_time = time.time()
    results = processor.process_batch(tickets)
    total_time = time.time() - start_time
    
    save_results(results, args.output)
    
    stats = processor.get_statistics()
    
    print("\n" + "="*80)
    print("PROCESSING SUMMARY")
    print("="*80)
    print(f"Total tickets: {stats.total_tickets}")
    print(f"Successful: {stats.successful_tickets}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average per ticket: {stats.average_processing_time_ms:.2f}ms")
    
    if stats.route_distribution:
        print("\nRoute Distribution:")
        for route, count in stats.route_distribution.items():
            print(f"  - {route}: {count} tickets ({count/stats.total_tickets*100:.1f}%)")


if __name__ == "__main__":
    main()
