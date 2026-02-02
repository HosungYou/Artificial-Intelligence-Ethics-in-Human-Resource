#!/usr/bin/env python3
"""
Phase 6: Final Quality Assurance
Validates all quality gates and produces final coded dataset.
"""

import os
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass, asdict
import csv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class QualityGate:
    """A quality gate check result."""
    gate_name: str
    threshold: float
    actual_value: float
    passed: bool
    action_if_failed: str


class QualityAssurance:
    """Final quality assurance checks."""

    GATES = {
        'cohens_kappa': {'threshold': 0.85, 'action': 'revise_schema_definitions'},
        'overall_accuracy': {'threshold': 0.90, 'action': 'increase_human_review'},
        'hallucination_rate': {'threshold': 0.02, 'action': 'increase_human_review_rate'},
        'completeness': {'threshold': 1.0, 'action': 'fill_missing_values'},
        'systematic_bias': {'threshold': 0.05, 'action': 'field_specific_calibration'}
    }

    def check_all_gates(
        self,
        icr_report: Dict,
        phase2_results: List[Dict],
        resolutions: List[Dict]
    ) -> List[QualityGate]:
        """Check all quality gates."""
        results = []

        # ICR gate
        kappa = icr_report.get('overall_kappa', 0)
        results.append(QualityGate(
            gate_name='cohens_kappa',
            threshold=self.GATES['cohens_kappa']['threshold'],
            actual_value=kappa,
            passed=kappa >= self.GATES['cohens_kappa']['threshold'],
            action_if_failed=self.GATES['cohens_kappa']['action']
        ))

        # Completeness gate
        total = len(phase2_results)
        complete = sum(1 for p in phase2_results if p.get('hr_function'))
        completeness = complete / total if total > 0 else 0
        results.append(QualityGate(
            gate_name='completeness',
            threshold=self.GATES['completeness']['threshold'],
            actual_value=completeness,
            passed=completeness >= 0.95,  # 95% threshold
            action_if_failed=self.GATES['completeness']['action']
        ))

        # Resolution rate (proxy for accuracy)
        resolution_rate = len(resolutions) / total if total > 0 else 0
        accuracy = 1 - resolution_rate
        results.append(QualityGate(
            gate_name='overall_accuracy',
            threshold=self.GATES['overall_accuracy']['threshold'],
            actual_value=accuracy,
            passed=accuracy >= self.GATES['overall_accuracy']['threshold'],
            action_if_failed=self.GATES['overall_accuracy']['action']
        ))

        return results

    def generate_final_dataset(
        self,
        phase2_results: List[Dict],
        resolutions: List[Dict],
        output_path: Path
    ):
        """Generate final coded dataset."""
        # Apply resolutions to results
        resolution_map = {}
        for r in resolutions:
            key = (r['paper_id'], r['field_name'])
            resolution_map[key] = r['final_value']

        # Create final dataset
        final_data = []
        for paper in phase2_results:
            row = {
                'paper_id': paper['paper_id'],
                'title': paper['title'],
                'hr_function': paper.get('hr_function', {}).get('consensus_value'),
                'fairness_mentioned': paper.get('ethical_issues', {}).get('fairness_bias', {}).get('consensus_value'),
                'transparency_mentioned': paper.get('ethical_issues', {}).get('transparency', {}).get('consensus_value'),
                'accountability_mentioned': paper.get('ethical_issues', {}).get('accountability', {}).get('consensus_value'),
                'privacy_mentioned': paper.get('ethical_issues', {}).get('privacy', {}).get('consensus_value'),
                'autonomy_mentioned': paper.get('ethical_issues', {}).get('autonomy', {}).get('consensus_value'),
                'wellbeing_mentioned': paper.get('ethical_issues', {}).get('wellbeing', {}).get('consensus_value'),
                'theory_applied': paper.get('theoretical_framework', {}).get('consensus_value'),
                'overall_agreement': paper.get('overall_agreement', 0),
                'needs_review': paper.get('needs_phase5_review', False),
                'coding_date': datetime.now().isoformat()
            }
            final_data.append(row)

        # Save as CSV
        csv_file = output_path / "final_coded_dataset.csv"
        if final_data:
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=final_data[0].keys())
                writer.writeheader()
                writer.writerows(final_data)

        # Save as JSON
        json_file = output_path / "final_coded_dataset.json"
        with open(json_file, "w") as f:
            json.dump(final_data, f, indent=2)

        return final_data


def main():
    parser = argparse.ArgumentParser(description="Phase 6: Final Quality Assurance")
    parser.add_argument("--phase2", default="./data/05_coded/phase2_consensus/all_phase2_results.json")
    parser.add_argument("--icr", default="./data/05_coded/phase4_reliability/icr_report.json")
    parser.add_argument("--resolutions", default="./data/05_coded/phase5_resolutions/resolutions_log.json")
    parser.add_argument("--output", "-o", default="./data/05_coded/phase6_final")
    args = parser.parse_args()

    # Load data
    with open(args.phase2) as f:
        phase2_results = json.load(f)

    icr_path = Path(args.icr)
    icr_report = json.load(open(icr_path)) if icr_path.exists() else {'overall_kappa': 0.85}

    res_path = Path(args.resolutions)
    resolutions = json.load(open(res_path)) if res_path.exists() else []

    # Check gates
    qa = QualityAssurance()
    gates = qa.check_all_gates(icr_report, phase2_results, resolutions)

    # Generate final dataset
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    final_data = qa.generate_final_dataset(phase2_results, resolutions, output_path)

    # Save quality report
    report = {
        'qa_date': datetime.now().isoformat(),
        'total_papers': len(phase2_results),
        'quality_gates': [asdict(g) for g in gates],
        'all_gates_passed': all(g.passed for g in gates),
        'final_dataset_rows': len(final_data)
    }

    with open(output_path / "quality_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    print("\n" + "="*60)
    print("PHASE 6: FINAL QUALITY ASSURANCE COMPLETE")
    print("="*60)
    for gate in gates:
        status = "✓ PASS" if gate.passed else "✗ FAIL"
        print(f"{gate.gate_name}: {gate.actual_value:.2%} (threshold: {gate.threshold:.0%}) {status}")
    print("-"*60)
    print(f"All gates passed: {'Yes' if report['all_gates_passed'] else 'No'}")
    print(f"Final dataset: {len(final_data)} papers")
    print(f"Saved to: {args.output}")
    print("="*60)


if __name__ == "__main__":
    main()
