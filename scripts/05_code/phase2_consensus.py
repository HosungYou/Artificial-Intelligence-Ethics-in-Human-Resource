#!/usr/bin/env python3
"""
Phase 2: Multi-Model Consensus Coding
Uses Claude, GPT-4o, and Groq/Llama to achieve consensus on extracted data.
"""

import os
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ConsensusStatus(str, Enum):
    """Consensus status for a field."""
    UNANIMOUS = "unanimous"  # 3/3 agree
    MAJORITY = "majority"    # 2/3 agree
    DISCORDANT = "discordant"  # No majority


@dataclass
class ModelExtraction:
    """Extraction result from a single model."""
    model_name: str
    provider: str
    value: Any
    confidence: float
    extraction_time: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class FieldConsensus:
    """Consensus result for a single field."""
    field_name: str
    consensus_value: Any
    consensus_status: ConsensusStatus
    model_extractions: List[ModelExtraction]
    agreement_score: float  # 0.0-1.0
    needs_human_review: bool = False


@dataclass
class Phase2ConsensusResult:
    """Result of Phase 2 consensus coding for a paper."""
    paper_id: str
    title: str
    consensus_date: str

    # Consensus for each field
    hr_function: FieldConsensus = None
    ai_technology: FieldConsensus = None
    ethical_issues: Dict[str, FieldConsensus] = field(default_factory=dict)
    theoretical_framework: FieldConsensus = None
    key_findings: FieldConsensus = None

    # Aggregate metrics
    overall_agreement: float = 0.0
    unanimous_fields: int = 0
    majority_fields: int = 0
    discordant_fields: int = 0
    needs_phase5_review: bool = False
    discordant_field_names: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'paper_id': self.paper_id,
            'title': self.title,
            'consensus_date': self.consensus_date,
            'hr_function': self._field_to_dict(self.hr_function),
            'ai_technology': self._field_to_dict(self.ai_technology),
            'ethical_issues': {k: self._field_to_dict(v) for k, v in self.ethical_issues.items()},
            'theoretical_framework': self._field_to_dict(self.theoretical_framework),
            'key_findings': self._field_to_dict(self.key_findings),
            'overall_agreement': self.overall_agreement,
            'unanimous_fields': self.unanimous_fields,
            'majority_fields': self.majority_fields,
            'discordant_fields': self.discordant_fields,
            'needs_phase5_review': self.needs_phase5_review,
            'discordant_field_names': self.discordant_field_names
        }

    def _field_to_dict(self, field_consensus: Optional[FieldConsensus]) -> Optional[Dict]:
        if not field_consensus:
            return None
        return {
            'field_name': field_consensus.field_name,
            'consensus_value': field_consensus.consensus_value,
            'consensus_status': field_consensus.consensus_status.value,
            'model_extractions': [asdict(m) for m in field_consensus.model_extractions],
            'agreement_score': field_consensus.agreement_score,
            'needs_human_review': field_consensus.needs_human_review
        }


class MultiModelExtractor:
    """Manages extraction across multiple LLM providers."""

    def __init__(self):
        self.models = {}
        self._init_claude()
        self._init_openai()
        self._init_groq()

    def _init_claude(self):
        """Initialize Claude/Anthropic client."""
        try:
            import anthropic
            self.models['claude'] = {
                'client': anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY")),
                'model': 'claude-3-5-sonnet-20241022',
                'provider': 'anthropic',
                'available': True
            }
            logger.info("Claude initialized")
        except Exception as e:
            logger.warning(f"Claude not available: {e}")
            self.models['claude'] = {'available': False}

    def _init_openai(self):
        """Initialize OpenAI client."""
        try:
            import openai
            self.models['openai'] = {
                'client': openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY")),
                'model': 'gpt-4o',
                'provider': 'openai',
                'available': True
            }
            logger.info("OpenAI initialized")
        except Exception as e:
            logger.warning(f"OpenAI not available: {e}")
            self.models['openai'] = {'available': False}

    def _init_groq(self):
        """Initialize Groq client."""
        try:
            from groq import Groq
            self.models['groq'] = {
                'client': Groq(api_key=os.getenv("GROQ_API_KEY")),
                'model': 'llama-3.3-70b-versatile',
                'provider': 'groq',
                'available': True
            }
            logger.info("Groq initialized")
        except Exception as e:
            logger.warning(f"Groq not available: {e}")
            self.models['groq'] = {'available': False}

    def _build_extraction_prompt(self, phase1_result: Dict) -> str:
        """Build prompt for verification extraction."""
        return f"""Review and verify the following coding of an academic paper on AI ethics in HR.

PAPER: {phase1_result.get('title', 'Unknown')}

INITIAL CODING (Phase 1):
{json.dumps(phase1_result, indent=2)}

VERIFICATION TASK:
Based on the initial coding and evidence provided, verify or correct each field.
Provide your own assessment with confidence scores.

RESPOND IN JSON FORMAT matching the Phase 1 structure:
{{
    "hr_function": {{"primary": "string", "confidence": 0.0-1.0}},
    "ai_technology": {{"types": ["list"], "confidence": 0.0-1.0}},
    "ethical_issues": {{
        "fairness_bias": {{"mentioned": bool, "severity": "string", "confidence": 0.0-1.0}},
        "transparency": {{"mentioned": bool, "severity": "string", "confidence": 0.0-1.0}},
        "accountability": {{"mentioned": bool, "severity": "string", "confidence": 0.0-1.0}},
        "privacy": {{"mentioned": bool, "severity": "string", "confidence": 0.0-1.0}},
        "autonomy": {{"mentioned": bool, "severity": "string", "confidence": 0.0-1.0}},
        "wellbeing": {{"mentioned": bool, "severity": "string", "confidence": 0.0-1.0}}
    }},
    "theoretical_framework": {{"applied": bool, "theory_name": "string or null", "confidence": 0.0-1.0}},
    "key_findings": {{"summary": "string", "confidence": 0.0-1.0}}
}}

Be precise and conservative with confidence scores."""

    def extract_with_model(self, model_key: str, phase1_result: Dict) -> Optional[Dict]:
        """Extract using a specific model."""
        if not self.models.get(model_key, {}).get('available'):
            return None

        config = self.models[model_key]
        prompt = self._build_extraction_prompt(phase1_result)

        try:
            if config['provider'] == 'anthropic':
                response = config['client'].messages.create(
                    model=config['model'],
                    max_tokens=2000,
                    temperature=0.1,
                    messages=[{"role": "user", "content": prompt}]
                )
                content = response.content[0].text

            elif config['provider'] == 'openai':
                response = config['client'].chat.completions.create(
                    model=config['model'],
                    max_tokens=2000,
                    temperature=0.1,
                    messages=[{"role": "user", "content": prompt}]
                )
                content = response.choices[0].message.content

            elif config['provider'] == 'groq':
                response = config['client'].chat.completions.create(
                    model=config['model'],
                    max_tokens=2000,
                    temperature=0.1,
                    messages=[{"role": "user", "content": prompt}]
                )
                content = response.choices[0].message.content

            # Parse JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            return json.loads(content)

        except Exception as e:
            logger.error(f"Error with {model_key}: {e}")
            return None

    def extract_all(self, phase1_result: Dict) -> Dict[str, Dict]:
        """Extract with all available models."""
        extractions = {}

        for model_key in ['claude', 'openai', 'groq']:
            if self.models.get(model_key, {}).get('available'):
                logger.info(f"Extracting with {model_key}...")
                result = self.extract_with_model(model_key, phase1_result)
                if result:
                    extractions[model_key] = result
                time.sleep(0.5)  # Rate limiting

        return extractions


class ConsensusCalculator:
    """Calculate consensus from multiple model extractions."""

    CRITICAL_FIELDS = [
        'hr_function.primary',
        'ethical_issues.fairness_bias.mentioned',
        'ethical_issues.transparency.mentioned'
    ]

    def calculate_field_consensus(
        self,
        field_name: str,
        extractions: Dict[str, Any]
    ) -> FieldConsensus:
        """Calculate consensus for a single field."""

        model_results = []
        values = []

        for model_name, extraction in extractions.items():
            value = self._get_nested_value(extraction, field_name)
            confidence = self._get_nested_value(extraction, f"{field_name.rsplit('.', 1)[0]}.confidence", 0.5)

            model_results.append(ModelExtraction(
                model_name=model_name,
                provider=model_name,
                value=value,
                confidence=confidence if isinstance(confidence, float) else 0.5
            ))
            values.append(value)

        # Calculate consensus
        consensus_value, status, agreement = self._find_consensus(values)

        # Check if field requires unanimous agreement
        is_critical = field_name in self.CRITICAL_FIELDS
        needs_review = status == ConsensusStatus.DISCORDANT or (
            is_critical and status != ConsensusStatus.UNANIMOUS
        )

        return FieldConsensus(
            field_name=field_name,
            consensus_value=consensus_value,
            consensus_status=status,
            model_extractions=model_results,
            agreement_score=agreement,
            needs_human_review=needs_review
        )

    def _get_nested_value(self, data: Dict, path: str, default=None) -> Any:
        """Get value from nested dict using dot notation."""
        keys = path.split('.')
        current = data

        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default

        return current

    def _find_consensus(self, values: List[Any]) -> Tuple[Any, ConsensusStatus, float]:
        """Find consensus value from list of values."""
        if not values:
            return None, ConsensusStatus.DISCORDANT, 0.0

        # Count occurrences
        value_counts = {}
        for v in values:
            # Convert to hashable representation
            v_key = json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else str(v)
            value_counts[v_key] = value_counts.get(v_key, 0) + 1

        # Find most common
        max_count = max(value_counts.values())
        total = len(values)

        # Get the actual value (not the key)
        consensus_key = [k for k, v in value_counts.items() if v == max_count][0]
        consensus_value = values[[json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else str(v)
                                  for v in values].index(consensus_key)]

        # Determine status
        if max_count == total:
            status = ConsensusStatus.UNANIMOUS
        elif max_count >= 2:  # Majority (2/3 or better)
            status = ConsensusStatus.MAJORITY
        else:
            status = ConsensusStatus.DISCORDANT

        agreement = max_count / total

        return consensus_value, status, agreement


class Phase2ConsensusCoder:
    """Orchestrates Phase 2 consensus coding."""

    ETHICS_PRINCIPLES = [
        'fairness_bias', 'transparency', 'accountability',
        'privacy', 'autonomy', 'wellbeing'
    ]

    def __init__(self):
        self.extractor = MultiModelExtractor()
        self.calculator = ConsensusCalculator()

    def process_paper(self, phase1_result: Dict) -> Phase2ConsensusResult:
        """Process a paper through multi-model consensus."""
        paper_id = phase1_result.get('paper_id', 'unknown')

        # Get extractions from all models
        model_extractions = self.extractor.extract_all(phase1_result)

        if len(model_extractions) < 2:
            logger.warning(f"Only {len(model_extractions)} models available for {paper_id}")

        # Create result
        result = Phase2ConsensusResult(
            paper_id=paper_id,
            title=phase1_result.get('title', ''),
            consensus_date=datetime.now().isoformat()
        )

        # Calculate consensus for each field
        result.hr_function = self.calculator.calculate_field_consensus(
            'hr_function.primary', model_extractions
        )

        result.ai_technology = self.calculator.calculate_field_consensus(
            'ai_technology.types', model_extractions
        )

        result.theoretical_framework = self.calculator.calculate_field_consensus(
            'theoretical_framework.applied', model_extractions
        )

        result.key_findings = self.calculator.calculate_field_consensus(
            'key_findings.summary', model_extractions
        )

        # Ethics principles
        for principle in self.ETHICS_PRINCIPLES:
            result.ethical_issues[principle] = self.calculator.calculate_field_consensus(
                f'ethical_issues.{principle}.mentioned', model_extractions
            )

        # Calculate aggregate metrics
        all_fields = [
            result.hr_function, result.ai_technology,
            result.theoretical_framework, result.key_findings
        ] + list(result.ethical_issues.values())

        all_fields = [f for f in all_fields if f is not None]

        result.unanimous_fields = sum(
            1 for f in all_fields if f.consensus_status == ConsensusStatus.UNANIMOUS
        )
        result.majority_fields = sum(
            1 for f in all_fields if f.consensus_status == ConsensusStatus.MAJORITY
        )
        result.discordant_fields = sum(
            1 for f in all_fields if f.consensus_status == ConsensusStatus.DISCORDANT
        )

        result.overall_agreement = sum(f.agreement_score for f in all_fields) / len(all_fields) if all_fields else 0

        # Determine if needs Phase 5 review
        result.discordant_field_names = [
            f.field_name for f in all_fields if f.needs_human_review
        ]
        result.needs_phase5_review = len(result.discordant_field_names) > 0

        return result

    def process_batch(
        self,
        phase1_results: List[Dict],
        progress_callback=None
    ) -> List[Phase2ConsensusResult]:
        """Process a batch of papers through consensus."""
        results = []

        for i, phase1_result in enumerate(phase1_results):
            try:
                result = self.process_paper(phase1_result)
                results.append(result)

                if progress_callback:
                    progress_callback(i + 1, len(phase1_results), result)

            except Exception as e:
                logger.error(f"Failed to process {phase1_result.get('paper_id', 'unknown')}: {e}")

        return results


def load_phase1_results(input_path: str) -> List[Dict]:
    """Load Phase 1 results."""
    with open(input_path) as f:
        return json.load(f)


def save_results(
    results: List[Phase2ConsensusResult],
    output_dir: str
):
    """Save Phase 2 consensus results."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save individual results
    for result in results:
        result_file = output_path / f"{result.paper_id}_consensus.json"
        with open(result_file, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

    # Save papers needing Phase 5 review
    phase5_queue = [r.to_dict() for r in results if r.needs_phase5_review]
    phase5_file = output_path / "phase5_queue.json"
    with open(phase5_file, "w") as f:
        json.dump(phase5_queue, f, indent=2)

    # Save summary
    summary = {
        "phase": 2,
        "consensus_date": datetime.now().isoformat(),
        "total_papers": len(results),
        "avg_agreement": sum(r.overall_agreement for r in results) / len(results) if results else 0,
        "papers_needing_review": len(phase5_queue),
        "unanimous_rate": sum(r.unanimous_fields for r in results) / (len(results) * 10) if results else 0,
        "discordant_rate": sum(r.discordant_fields for r in results) / (len(results) * 10) if results else 0
    }

    summary_file = output_path / "phase2_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    # Save combined results
    combined = [r.to_dict() for r in results]
    combined_file = output_path / "all_phase2_results.json"
    with open(combined_file, "w") as f:
        json.dump(combined, f, indent=2)

    logger.info(f"Saved {len(results)} Phase 2 results to {output_dir}")
    logger.info(f"{len(phase5_queue)} papers queued for Phase 5 review")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Phase 2: Multi-Model Consensus Coding"
    )
    parser.add_argument(
        "--input", "-i",
        default="./data/05_coded/phase1_raw/all_phase1_results.json",
        help="Input file with Phase 1 results"
    )
    parser.add_argument(
        "--output", "-o",
        default="./data/05_coded/phase2_consensus",
        help="Output directory for Phase 2 results"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of papers (for testing)"
    )

    args = parser.parse_args()

    # Load Phase 1 results
    logger.info(f"Loading Phase 1 results from {args.input}")
    phase1_results = load_phase1_results(args.input)

    if args.limit:
        phase1_results = phase1_results[:args.limit]

    logger.info(f"Processing {len(phase1_results)} papers through consensus")

    # Initialize coder
    coder = Phase2ConsensusCoder()

    # Progress callback
    def progress(current, total, result):
        status = "needs review" if result.needs_phase5_review else "consensus reached"
        logger.info(f"[{current}/{total}] {result.paper_id} - {result.overall_agreement:.0%} agreement - {status}")

    # Process papers
    results = coder.process_batch(phase1_results, progress_callback=progress)

    # Save results
    save_results(results, args.output)

    # Print summary
    avg_agreement = sum(r.overall_agreement for r in results) / len(results) if results else 0
    needs_review = sum(1 for r in results if r.needs_phase5_review)

    print("\n" + "="*60)
    print("PHASE 2: MULTI-MODEL CONSENSUS COMPLETE")
    print("="*60)
    print(f"Papers processed:       {len(results)}")
    print(f"Average agreement:      {avg_agreement:.1%}")
    print(f"Papers needing review:  {needs_review}")
    print(f"Consensus rate:         {(len(results) - needs_review) / len(results):.1%}")
    print("="*60)
    print(f"Results saved to: {args.output}")
    print("="*60)


if __name__ == "__main__":
    main()
