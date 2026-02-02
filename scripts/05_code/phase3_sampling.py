#!/usr/bin/env python3
"""
Phase 3: Human Verification Sampling
Stratified sampling for human gold standard creation.
"""

import os
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import random
import csv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class SamplingConfig:
    """Configuration for stratified sampling."""
    overall_rate: float = 0.20
    minimum_papers: int = 30
    maximum_papers: int = 50
    confidence_oversample_low: float = 0.60
    random_seed: int = 42


class StratifiedSampler:
    """Stratified sampling for human verification."""

    ETHICS_PRINCIPLES = ['fairness_bias', 'transparency', 'accountability',
                         'privacy', 'autonomy', 'wellbeing']
    HR_FUNCTIONS = ['recruitment', 'selection', 'performance_management',
                    'learning_development', 'people_analytics', 'employee_relations']

    def __init__(self, config: SamplingConfig):
        self.config = config
        random.seed(config.random_seed)

    def sample(self, papers: List[Dict]) -> List[Dict]:
        """Perform stratified sampling."""
        target_size = max(
            self.config.minimum_papers,
            min(self.config.maximum_papers, int(len(papers) * self.config.overall_rate))
        )

        # Initialize sample with required coverage
        sample = []
        remaining = list(papers)

        # 1. Ensure ethics principle coverage (1 per principle)
        for principle in self.ETHICS_PRINCIPLES:
            paper = self._find_paper_with_ethics(remaining, principle)
            if paper and paper not in sample:
                sample.append(paper)
                remaining.remove(paper)

        # 2. Ensure HR function coverage
        for function in self.HR_FUNCTIONS:
            paper = self._find_paper_with_hr_function(remaining, function)
            if paper and paper not in sample:
                sample.append(paper)
                remaining.remove(paper)

        # 3. Oversample low confidence
        remaining_needed = target_size - len(sample)
        if remaining_needed > 0:
            sorted_by_conf = sorted(remaining, key=lambda p: self._get_confidence(p))
            n_low = int(remaining_needed * self.config.confidence_oversample_low)
            n_high = remaining_needed - n_low

            mid = len(sorted_by_conf) // 2
            low_conf = sorted_by_conf[:mid]
            high_conf = sorted_by_conf[mid:]

            sample.extend(random.sample(low_conf, min(n_low, len(low_conf))))
            sample.extend(random.sample(high_conf, min(n_high, len(high_conf))))

        return sample[:target_size]

    def _find_paper_with_ethics(self, papers: List[Dict], principle: str) -> Optional[Dict]:
        for paper in papers:
            ethics = paper.get('ethical_issues', {})
            if ethics.get(principle, {}).get('consensus_value') == True:
                return paper
        return papers[0] if papers else None

    def _find_paper_with_hr_function(self, papers: List[Dict], function: str) -> Optional[Dict]:
        for paper in papers:
            hr = paper.get('hr_function', {})
            if hr.get('consensus_value') == function:
                return paper
        return None

    def _get_confidence(self, paper: Dict) -> float:
        return paper.get('overall_agreement', 0.5)


def create_human_template(papers: List[Dict], output_path: Path):
    """Create Excel-like template for human coding."""
    csv_file = output_path / "human_gold_standard_template.csv"

    fieldnames = [
        'paper_id', 'title', 'ai_hr_function', 'ai_ethics_mentioned',
        'human_hr_function', 'human_fairness_mentioned', 'human_transparency_mentioned',
        'human_accountability_mentioned', 'human_privacy_mentioned',
        'human_autonomy_mentioned', 'human_wellbeing_mentioned',
        'human_theory_applied', 'human_notes', 'coding_date'
    ]

    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for paper in papers:
            ethics = paper.get('ethical_issues', {})
            writer.writerow({
                'paper_id': paper.get('paper_id', ''),
                'title': paper.get('title', ''),
                'ai_hr_function': paper.get('hr_function', {}).get('consensus_value', ''),
                'ai_ethics_mentioned': '; '.join([
                    p for p in ['fairness_bias', 'transparency', 'accountability',
                               'privacy', 'autonomy', 'wellbeing']
                    if ethics.get(p, {}).get('consensus_value') == True
                ]),
                'human_hr_function': '',
                'human_fairness_mentioned': '',
                'human_transparency_mentioned': '',
                'human_accountability_mentioned': '',
                'human_privacy_mentioned': '',
                'human_autonomy_mentioned': '',
                'human_wellbeing_mentioned': '',
                'human_theory_applied': '',
                'human_notes': '',
                'coding_date': ''
            })

    logger.info(f"Created human coding template: {csv_file}")


def main():
    parser = argparse.ArgumentParser(description="Phase 3: Human Verification Sampling")
    parser.add_argument("--input", "-i", default="./data/05_coded/phase2_consensus/all_phase2_results.json")
    parser.add_argument("--output", "-o", default="./data/05_coded/phase3_human")
    parser.add_argument("--rate", type=float, default=0.20)
    parser.add_argument("--min-papers", type=int, default=30)
    args = parser.parse_args()

    # Load Phase 2 results
    with open(args.input) as f:
        papers = json.load(f)

    # Sample
    config = SamplingConfig(overall_rate=args.rate, minimum_papers=args.min_papers)
    sampler = StratifiedSampler(config)
    sample = sampler.sample(papers)

    # Save
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / "human_verification_sample.json", "w") as f:
        json.dump(sample, f, indent=2)

    create_human_template(sample, output_path)

    print(f"\nPhase 3: Selected {len(sample)} papers for human verification")
    print(f"Template saved to: {args.output}")


if __name__ == "__main__":
    main()
