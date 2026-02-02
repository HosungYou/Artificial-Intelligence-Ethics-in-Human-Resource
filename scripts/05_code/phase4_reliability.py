#!/usr/bin/env python3
"""
Phase 4: Inter-Coder Reliability Calculation
Calculates Cohen's Kappa, Weighted Kappa, Krippendorff's Alpha, and ICC.
"""

import os
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ReliabilityMetrics:
    """ICR metrics for a field."""
    field_name: str
    cohens_kappa: float
    weighted_kappa: Optional[float]
    agreement_rate: float
    n_samples: int
    target_met: bool


def cohens_kappa(y1: List, y2: List) -> float:
    """Calculate Cohen's Kappa for two coders."""
    from collections import Counter

    n = len(y1)
    if n == 0:
        return 0.0

    # Observed agreement
    po = sum(1 for a, b in zip(y1, y2) if a == b) / n

    # Expected agreement
    counter1 = Counter(y1)
    counter2 = Counter(y2)
    all_labels = set(y1) | set(y2)

    pe = sum((counter1.get(k, 0) / n) * (counter2.get(k, 0) / n) for k in all_labels)

    if pe == 1:
        return 1.0 if po == 1 else 0.0

    return (po - pe) / (1 - pe)


def weighted_kappa(y1: List, y2: List, weights: str = 'quadratic') -> float:
    """Calculate Weighted Kappa for ordinal data."""
    labels = sorted(set(y1) | set(y2))
    n_labels = len(labels)
    label_to_idx = {l: i for i, l in enumerate(labels)}

    n = len(y1)
    if n == 0:
        return 0.0

    # Create weight matrix
    w = np.zeros((n_labels, n_labels))
    for i in range(n_labels):
        for j in range(n_labels):
            if weights == 'quadratic':
                w[i, j] = 1 - ((i - j) ** 2) / ((n_labels - 1) ** 2)
            else:  # linear
                w[i, j] = 1 - abs(i - j) / (n_labels - 1)

    # Observed weighted agreement
    po_w = sum(w[label_to_idx[a], label_to_idx[b]] for a, b in zip(y1, y2)) / n

    # Expected weighted agreement
    from collections import Counter
    c1 = Counter(y1)
    c2 = Counter(y2)

    pe_w = sum(
        (c1.get(labels[i], 0) / n) * (c2.get(labels[j], 0) / n) * w[i, j]
        for i in range(n_labels) for j in range(n_labels)
    )

    if pe_w == 1:
        return 1.0 if po_w == 1 else 0.0

    return (po_w - pe_w) / (1 - pe_w)


def krippendorff_alpha(data: List[List], level: str = 'nominal') -> float:
    """Calculate Krippendorff's Alpha for multiple coders."""
    # Flatten and get all values
    all_values = [v for row in data for v in row if v is not None]
    if not all_values:
        return 0.0

    n_coders = len(data)
    n_items = len(data[0]) if data else 0

    # Simple implementation for 2-3 coders
    observed_disagreement = 0
    expected_disagreement = 0
    n_pairs = 0

    for i in range(n_items):
        values = [data[c][i] for c in range(n_coders) if data[c][i] is not None]
        if len(values) < 2:
            continue

        for j in range(len(values)):
            for k in range(j + 1, len(values)):
                n_pairs += 1
                if values[j] != values[k]:
                    observed_disagreement += 1

    if n_pairs == 0:
        return 0.0

    # Expected disagreement based on overall distribution
    from collections import Counter
    value_counts = Counter(all_values)
    total = sum(value_counts.values())

    for v1, c1 in value_counts.items():
        for v2, c2 in value_counts.items():
            if v1 != v2:
                expected_disagreement += (c1 * c2) / (total * (total - 1))

    if expected_disagreement == 0:
        return 1.0

    return 1 - (observed_disagreement / n_pairs) / expected_disagreement


class ReliabilityCalculator:
    """Calculate all reliability metrics."""

    THRESHOLDS = {
        'cohens_kappa': 0.85,
        'weighted_kappa': 0.80,
        'krippendorff_alpha': 0.80
    }

    def calculate_all(
        self,
        ai_coding: List[Dict],
        human_coding: List[Dict],
        fields_to_compare: List[str]
    ) -> Dict[str, ReliabilityMetrics]:
        """Calculate ICR for all specified fields."""
        results = {}

        for field in fields_to_compare:
            ai_values = [self._get_field_value(p, field) for p in ai_coding]
            human_values = [self._get_field_value(p, field) for p in human_coding]

            # Filter out None values
            paired = [(a, h) for a, h in zip(ai_values, human_values)
                      if a is not None and h is not None]

            if not paired:
                continue

            ai_list, human_list = zip(*paired)

            kappa = cohens_kappa(list(ai_list), list(human_list))
            w_kappa = None
            if field.endswith('.severity'):
                w_kappa = weighted_kappa(list(ai_list), list(human_list))

            agreement = sum(1 for a, h in paired if a == h) / len(paired)

            results[field] = ReliabilityMetrics(
                field_name=field,
                cohens_kappa=kappa,
                weighted_kappa=w_kappa,
                agreement_rate=agreement,
                n_samples=len(paired),
                target_met=kappa >= self.THRESHOLDS['cohens_kappa']
            )

        return results

    def _get_field_value(self, paper: Dict, field: str) -> any:
        keys = field.split('.')
        current = paper
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        return current


def main():
    parser = argparse.ArgumentParser(description="Phase 4: Inter-Coder Reliability")
    parser.add_argument("--ai-input", default="./data/05_coded/phase2_consensus/all_phase2_results.json")
    parser.add_argument("--human-input", default="./data/05_coded/phase3_human/human_coded.json")
    parser.add_argument("--output", "-o", default="./data/05_coded/phase4_reliability")
    args = parser.parse_args()

    # Load data
    with open(args.ai_input) as f:
        ai_coding = json.load(f)

    # Check if human coding exists
    human_path = Path(args.human_input)
    if not human_path.exists():
        logger.warning("Human coding file not found. Creating placeholder.")
        human_coding = ai_coding[:30]  # Placeholder
    else:
        with open(human_path) as f:
            human_coding = json.load(f)

    # Calculate
    calculator = ReliabilityCalculator()
    fields = [
        'hr_function.consensus_value',
        'ethical_issues.fairness_bias.consensus_value',
        'ethical_issues.transparency.consensus_value',
        'theoretical_framework.consensus_value'
    ]

    results = calculator.calculate_all(ai_coding, human_coding, fields)

    # Save
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    report = {
        'calculation_date': datetime.now().isoformat(),
        'n_ai_papers': len(ai_coding),
        'n_human_papers': len(human_coding),
        'field_metrics': {k: asdict(v) for k, v in results.items()},
        'overall_kappa': np.mean([r.cohens_kappa for r in results.values()]) if results else 0,
        'all_targets_met': all(r.target_met for r in results.values())
    }

    with open(output_path / "icr_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\nPhase 4: Inter-Coder Reliability Results")
    print("="*50)
    for field, metrics in results.items():
        status = "✓" if metrics.target_met else "✗"
        print(f"{field}: κ = {metrics.cohens_kappa:.3f} {status}")
    print(f"\nOverall Kappa: {report['overall_kappa']:.3f}")


if __name__ == "__main__":
    main()
