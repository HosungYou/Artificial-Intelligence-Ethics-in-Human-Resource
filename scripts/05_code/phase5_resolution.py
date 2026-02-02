#!/usr/bin/env python3
"""
Phase 5: Discrepancy Resolution Protocol
Resolves disagreements between AI models and human coding.
"""

import os
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass, asdict
from enum import Enum

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ResolutionMethod(str, Enum):
    RE_EXTRACT = "re_extract_with_context"
    HUMAN_DECISION = "human_final_decision"
    SCHEMA_UPDATE = "update_codebook_clarification"
    DOCUMENT = "document_and_decide"


@dataclass
class DiscrepancyResolution:
    """Record of a resolved discrepancy."""
    paper_id: str
    field_name: str
    original_values: Dict[str, any]
    final_value: any
    resolution_method: str
    rationale: str
    resolver: str
    timestamp: str


class DiscrepancyResolver:
    """Resolve discrepancies in coding."""

    def __init__(self):
        self.resolutions = []

    def resolve(
        self,
        paper_id: str,
        field_name: str,
        model_values: Dict[str, any],
        human_value: any = None
    ) -> DiscrepancyResolution:
        """Resolve a single discrepancy."""

        # Determine resolution method
        if human_value is not None:
            # Human value takes precedence
            method = ResolutionMethod.HUMAN_DECISION
            final_value = human_value
            rationale = "Human gold standard value used"
        else:
            # Use majority if available
            values = list(model_values.values())
            from collections import Counter
            counts = Counter(str(v) for v in values)
            most_common = counts.most_common(1)[0]

            if most_common[1] >= 2:
                method = ResolutionMethod.DOCUMENT
                final_value = [v for v in values if str(v) == most_common[0]][0]
                rationale = f"Majority vote ({most_common[1]}/3 models agreed)"
            else:
                method = ResolutionMethod.RE_EXTRACT
                final_value = values[0]  # Use Claude's value as default
                rationale = "No majority - defaulting to Claude extraction"

        resolution = DiscrepancyResolution(
            paper_id=paper_id,
            field_name=field_name,
            original_values=model_values,
            final_value=final_value,
            resolution_method=method.value,
            rationale=rationale,
            resolver="system",
            timestamp=datetime.now().isoformat()
        )

        self.resolutions.append(resolution)
        return resolution

    def resolve_batch(self, phase5_queue: List[Dict]) -> List[DiscrepancyResolution]:
        """Resolve all papers in Phase 5 queue."""
        for paper in phase5_queue:
            for field_name in paper.get('discordant_field_names', []):
                # Get model values for this field
                model_values = {}
                for field_consensus_key in ['hr_function', 'ai_technology',
                                            'theoretical_framework', 'key_findings']:
                    if field_name.startswith(field_consensus_key):
                        consensus = paper.get(field_consensus_key, {})
                        for extraction in consensus.get('model_extractions', []):
                            model_values[extraction['model_name']] = extraction['value']

                # Check ethics fields
                if field_name.startswith('ethical_issues.'):
                    principle = field_name.split('.')[1]
                    ethics = paper.get('ethical_issues', {}).get(principle, {})
                    for extraction in ethics.get('model_extractions', []):
                        model_values[extraction['model_name']] = extraction['value']

                if model_values:
                    self.resolve(paper['paper_id'], field_name, model_values)

        return self.resolutions


def main():
    parser = argparse.ArgumentParser(description="Phase 5: Discrepancy Resolution")
    parser.add_argument("--input", "-i", default="./data/05_coded/phase2_consensus/phase5_queue.json")
    parser.add_argument("--output", "-o", default="./data/05_coded/phase5_resolutions")
    args = parser.parse_args()

    # Load Phase 5 queue
    with open(args.input) as f:
        phase5_queue = json.load(f)

    # Resolve
    resolver = DiscrepancyResolver()
    resolutions = resolver.resolve_batch(phase5_queue)

    # Save
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / "resolutions_log.json", "w") as f:
        json.dump([asdict(r) for r in resolutions], f, indent=2)

    summary = {
        'resolution_date': datetime.now().isoformat(),
        'total_discrepancies': len(resolutions),
        'by_method': {}
    }
    for r in resolutions:
        summary['by_method'][r.resolution_method] = summary['by_method'].get(r.resolution_method, 0) + 1

    with open(output_path / "phase5_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nPhase 5: Resolved {len(resolutions)} discrepancies")
    for method, count in summary['by_method'].items():
        print(f"  - {method}: {count}")


if __name__ == "__main__":
    main()
