#!/usr/bin/env python3
"""
Stage 3: AI-Assisted Screening Script for AI-Ethics-HR Systematic Review
Uses Groq LLM for cost-effective title/abstract screening.
"""

import os
import json
import logging
import argparse
import time
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ScreeningDecision(str, Enum):
    """Screening decision options."""
    INCLUDE = "include"
    EXCLUDE = "exclude"
    UNCERTAIN = "uncertain"


@dataclass
class ScreeningResult:
    """Result of screening a single paper."""
    paper_id: str
    title: str
    decision: ScreeningDecision
    confidence: float
    reasons: List[str]
    exclusion_criteria_matched: List[str] = field(default_factory=list)
    screener: str = "ai"
    screening_date: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ScreeningStats:
    """Statistics from screening process."""
    total_screened: int = 0
    included: int = 0
    excluded: int = 0
    uncertain: int = 0
    exclusion_reasons: Dict[str, int] = field(default_factory=dict)


class GroqScreener:
    """Screen papers using Groq API with Llama model."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.1
    ):
        try:
            from groq import Groq
            self.client = Groq(api_key=api_key or os.getenv("GROQ_API_KEY"))
            self.model = model
            self.temperature = temperature
            self.available = True
        except ImportError:
            logger.warning("Groq not installed. Install with: pip install groq")
            self.available = False
        except Exception as e:
            logger.error(f"Failed to initialize Groq client: {e}")
            self.available = False

    def _build_screening_prompt(self, paper: Dict) -> str:
        """Build the screening prompt for a paper."""
        title = paper.get('title', 'N/A')
        abstract = paper.get('abstract', 'N/A') or 'Abstract not available'

        prompt = f"""You are a research assistant screening papers for a systematic literature review on AI Ethics in Human Resource Management.

PAPER TO SCREEN:
Title: {title}

Abstract: {abstract}

INCLUSION CRITERIA (paper must meet ALL):
1. Focus on AI/ML applications in HR contexts (recruitment, selection, performance management, learning & development, people analytics, employee relations)
2. Addresses ethical implications or concerns (bias, fairness, transparency, accountability, privacy, autonomy, wellbeing)
3. Academic/scholarly source (not just news or opinion)

EXCLUSION CRITERIA (paper is excluded if ANY match):
1. Pure technical papers about AI/ML without HR application or ethical discussion
2. General business AI papers without specific HR focus
3. HR software reviews without ethical analysis
4. Non-research papers (news articles, blog posts, editorials without empirical/theoretical contribution)
5. Papers about AI ethics in other domains (healthcare, autonomous vehicles) without HR relevance
6. Papers about HR practices without AI/technology component

RESPOND IN JSON FORMAT:
{{
    "decision": "include" | "exclude" | "uncertain",
    "confidence": 0.0-1.0,
    "reasons": ["list of reasons for decision"],
    "exclusion_criteria_matched": ["list of matched exclusion criteria, if any"]
}}

Be conservative - if uncertain, mark as "uncertain" rather than incorrectly excluding relevant papers."""

        return prompt

    def screen_paper(self, paper: Dict) -> ScreeningResult:
        """Screen a single paper using LLM."""
        if not self.available:
            return ScreeningResult(
                paper_id=paper.get('source_id', 'unknown'),
                title=paper.get('title', ''),
                decision=ScreeningDecision.UNCERTAIN,
                confidence=0.0,
                reasons=["Groq API not available"],
                screener="fallback"
            )

        prompt = self._build_screening_prompt(paper)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a systematic review screening assistant. Always respond with valid JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.temperature,
                max_tokens=500
            )

            # Parse response
            content = response.choices[0].message.content.strip()

            # Extract JSON from response
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            result_data = json.loads(content)

            return ScreeningResult(
                paper_id=paper.get('source_id', 'unknown'),
                title=paper.get('title', ''),
                decision=ScreeningDecision(result_data.get('decision', 'uncertain')),
                confidence=float(result_data.get('confidence', 0.5)),
                reasons=result_data.get('reasons', []),
                exclusion_criteria_matched=result_data.get('exclusion_criteria_matched', []),
                screener=f"ai:{self.model}"
            )

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response: {e}")
            return ScreeningResult(
                paper_id=paper.get('source_id', 'unknown'),
                title=paper.get('title', ''),
                decision=ScreeningDecision.UNCERTAIN,
                confidence=0.0,
                reasons=[f"Parse error: {str(e)}"],
                screener="ai:error"
            )
        except Exception as e:
            logger.error(f"Screening error: {e}")
            return ScreeningResult(
                paper_id=paper.get('source_id', 'unknown'),
                title=paper.get('title', ''),
                decision=ScreeningDecision.UNCERTAIN,
                confidence=0.0,
                reasons=[f"API error: {str(e)}"],
                screener="ai:error"
            )

    def screen_batch(
        self,
        papers: List[Dict],
        delay: float = 0.5,
        progress_callback=None
    ) -> List[ScreeningResult]:
        """Screen a batch of papers with rate limiting."""
        results = []

        for i, paper in enumerate(papers):
            result = self.screen_paper(paper)
            results.append(result)

            if progress_callback:
                progress_callback(i + 1, len(papers), result)

            # Rate limiting
            if i < len(papers) - 1:
                time.sleep(delay)

        return results


class ScreeningPipeline:
    """Orchestrates the screening process."""

    def __init__(
        self,
        screener: GroqScreener,
        human_verification_rate: float = 0.20,
        uncertain_threshold: float = 0.70
    ):
        self.screener = screener
        self.human_verification_rate = human_verification_rate
        self.uncertain_threshold = uncertain_threshold
        self.stats = ScreeningStats()

    def screen_papers(
        self,
        papers: List[Dict],
        batch_delay: float = 0.5
    ) -> Tuple[List[Dict], List[ScreeningResult]]:
        """
        Screen all papers and return included papers with results.

        Args:
            papers: List of paper dictionaries
            batch_delay: Delay between API calls

        Returns:
            Tuple of (included papers, all screening results)
        """
        def progress_callback(current, total, result):
            if current % 10 == 0 or current == total:
                logger.info(f"Screened {current}/{total} papers")

        # Screen all papers
        logger.info(f"Starting AI screening of {len(papers)} papers...")
        results = self.screener.screen_batch(
            papers, delay=batch_delay, progress_callback=progress_callback
        )

        # Map results to papers
        result_map = {r.paper_id: r for r in results}

        # Classify papers
        included_papers = []
        for paper in papers:
            paper_id = paper.get('source_id', 'unknown')
            result = result_map.get(paper_id)

            if result:
                paper['screening_result'] = asdict(result)

                if result.decision == ScreeningDecision.INCLUDE:
                    included_papers.append(paper)
                    self.stats.included += 1
                elif result.decision == ScreeningDecision.EXCLUDE:
                    self.stats.excluded += 1
                    # Track exclusion reasons
                    for reason in result.exclusion_criteria_matched:
                        self.stats.exclusion_reasons[reason] = \
                            self.stats.exclusion_reasons.get(reason, 0) + 1
                else:  # Uncertain
                    # Include uncertain papers for human review
                    paper['needs_human_review'] = True
                    included_papers.append(paper)
                    self.stats.uncertain += 1

        self.stats.total_screened = len(papers)

        return included_papers, results

    def select_human_verification_sample(
        self,
        papers: List[Dict],
        results: List[ScreeningResult],
        minimum_sample: int = 30
    ) -> List[Dict]:
        """
        Select papers for human verification using stratified sampling.

        Stratifies by:
        - Decision (include/exclude/uncertain)
        - Confidence level
        """
        sample_size = max(
            minimum_sample,
            int(len(papers) * self.human_verification_rate)
        )

        # Stratify by decision
        included = [r for r in results if r.decision == ScreeningDecision.INCLUDE]
        excluded = [r for r in results if r.decision == ScreeningDecision.EXCLUDE]
        uncertain = [r for r in results if r.decision == ScreeningDecision.UNCERTAIN]

        sample = []

        # Allocate proportionally with minimum representation
        n_include = max(5, int(sample_size * len(included) / len(results)))
        n_exclude = max(5, int(sample_size * len(excluded) / len(results)))
        n_uncertain = sample_size - n_include - n_exclude

        # Over-sample low confidence papers
        def sample_by_confidence(group: List[ScreeningResult], n: int) -> List[ScreeningResult]:
            if not group or n <= 0:
                return []
            if len(group) <= n:
                return group

            # Sort by confidence (ascending) to prioritize low confidence
            sorted_group = sorted(group, key=lambda r: r.confidence)

            # Take more from low confidence
            n_low = int(n * 0.6)  # 60% from bottom half
            n_high = n - n_low

            mid = len(sorted_group) // 2
            low_conf = sorted_group[:mid] if mid > 0 else sorted_group
            high_conf = sorted_group[mid:] if mid < len(sorted_group) else []

            selected = []
            if low_conf:
                selected.extend(random.sample(low_conf, min(n_low, len(low_conf))))
            if high_conf:
                remaining = n - len(selected)
                selected.extend(random.sample(high_conf, min(remaining, len(high_conf))))

            return selected[:n]

        sample.extend(sample_by_confidence(included, n_include))
        sample.extend(sample_by_confidence(excluded, n_exclude))
        sample.extend(sample_by_confidence(uncertain, n_uncertain))

        # Map back to papers
        sample_ids = {r.paper_id for r in sample}
        sample_papers = [p for p in papers if p.get('source_id') in sample_ids]

        logger.info(f"Selected {len(sample_papers)} papers for human verification")

        return sample_papers


def load_papers(input_path: str) -> List[Dict]:
    """Load papers from JSON file."""
    with open(input_path) as f:
        return json.load(f)


def save_screening_results(
    included_papers: List[Dict],
    all_results: List[ScreeningResult],
    verification_sample: List[Dict],
    stats: ScreeningStats,
    output_dir: str
):
    """Save screening results and statistics."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save included papers (for full-text retrieval)
    included_file = output_path / "screened_included.json"
    with open(included_file, "w") as f:
        json.dump(included_papers, f, indent=2)

    # Save all screening results
    results_file = output_path / "all_screening_results.json"
    with open(results_file, "w") as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)

    # Save human verification sample
    verification_file = output_path / "human_verification_sample.json"
    with open(verification_file, "w") as f:
        json.dump(verification_sample, f, indent=2)

    # Create human verification spreadsheet template
    _create_verification_template(verification_sample, output_path)

    # Save PRISMA screening summary
    summary = {
        "screening_date": datetime.now().isoformat(),
        "total_screened": stats.total_screened,
        "screening_results": {
            "included": stats.included,
            "excluded": stats.excluded,
            "uncertain": stats.uncertain
        },
        "exclusion_reasons": stats.exclusion_reasons,
        "human_verification": {
            "sample_size": len(verification_sample),
            "verification_file": "human_verification_sample.json"
        }
    }

    summary_file = output_path / "prisma_screening_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def _create_verification_template(papers: List[Dict], output_path: Path):
    """Create a template for human verification."""
    try:
        import csv

        csv_file = output_path / "human_verification_template.csv"

        fieldnames = [
            'paper_id', 'title', 'abstract_preview',
            'ai_decision', 'ai_confidence', 'ai_reasons',
            'human_decision', 'human_notes', 'agreement'
        ]

        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for paper in papers:
                screening = paper.get('screening_result', {})
                abstract = paper.get('abstract', '')[:300] + '...' if paper.get('abstract') else 'N/A'

                writer.writerow({
                    'paper_id': paper.get('source_id', ''),
                    'title': paper.get('title', ''),
                    'abstract_preview': abstract,
                    'ai_decision': screening.get('decision', ''),
                    'ai_confidence': screening.get('confidence', ''),
                    'ai_reasons': '; '.join(screening.get('reasons', [])),
                    'human_decision': '',
                    'human_notes': '',
                    'agreement': ''
                })

        logger.info(f"Created human verification template: {csv_file}")

    except Exception as e:
        logger.error(f"Failed to create verification template: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="AI-assisted screening for AI Ethics in HR systematic review"
    )
    parser.add_argument(
        "--input", "-i",
        default="./data/02_deduplicated/deduplicated_papers.json",
        help="Input file with deduplicated papers"
    )
    parser.add_argument(
        "--output", "-o",
        default="./data/03_screened",
        help="Output directory for screening results"
    )
    parser.add_argument(
        "--model",
        default="llama-3.3-70b-versatile",
        help="Groq model to use for screening"
    )
    parser.add_argument(
        "--verification-rate",
        type=float,
        default=0.20,
        help="Human verification sample rate (0-1)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay between API calls in seconds"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of papers to screen (for testing)"
    )

    args = parser.parse_args()

    # Load papers
    logger.info(f"Loading papers from {args.input}")
    papers = load_papers(args.input)
    logger.info(f"Loaded {len(papers)} papers")

    if args.limit:
        papers = papers[:args.limit]
        logger.info(f"Limited to {len(papers)} papers for testing")

    # Initialize screener and pipeline
    screener = GroqScreener(model=args.model)
    if not screener.available:
        logger.error("Groq API not available. Please set GROQ_API_KEY environment variable.")
        return

    pipeline = ScreeningPipeline(
        screener=screener,
        human_verification_rate=args.verification_rate
    )

    # Screen papers
    included_papers, all_results = pipeline.screen_papers(
        papers, batch_delay=args.delay
    )

    # Select human verification sample
    verification_sample = pipeline.select_human_verification_sample(
        papers, all_results
    )

    # Save results
    summary = save_screening_results(
        included_papers,
        all_results,
        verification_sample,
        pipeline.stats,
        args.output
    )

    # Print summary
    stats = pipeline.stats
    print("\n" + "="*60)
    print("STAGE 3: SCREENING COMPLETE")
    print("="*60)
    print(f"Total papers screened:  {stats.total_screened}")
    print(f"Included:               {stats.included}")
    print(f"Excluded:               {stats.excluded}")
    print(f"Uncertain (for review): {stats.uncertain}")
    print("-"*60)
    print("Exclusion reasons:")
    for reason, count in sorted(stats.exclusion_reasons.items(),
                                 key=lambda x: x[1], reverse=True):
        print(f"  - {reason}: {count}")
    print("-"*60)
    print(f"Human verification sample: {len(verification_sample)} papers")
    print("="*60)
    print(f"Results saved to: {args.output}")
    print("="*60)


if __name__ == "__main__":
    main()
