#!/usr/bin/env python3
"""
Audit trail logging for systematic review coding pipeline.
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field
import hashlib


@dataclass
class AuditEntry:
    """A single audit log entry."""
    timestamp: str
    phase: str
    action: str
    paper_id: Optional[str]
    field_name: Optional[str]
    old_value: Optional[Any]
    new_value: Optional[Any]
    reason: str
    actor: str  # 'ai:claude', 'ai:gpt4', 'human:coder1', 'system'
    confidence: Optional[float] = None
    metadata: Dict = field(default_factory=dict)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(asdict(self), default=str)

    @classmethod
    def from_json(cls, json_str: str) -> 'AuditEntry':
        """Create from JSON string."""
        data = json.loads(json_str)
        return cls(**data)


class AuditLogger:
    """Manages audit trail for the coding pipeline."""

    def __init__(
        self,
        log_file: str,
        auto_flush: bool = True
    ):
        """
        Initialize audit logger.

        Args:
            log_file: Path to JSONL audit log file
            auto_flush: Whether to flush after each write
        """
        self.log_file = Path(log_file)
        self.auto_flush = auto_flush
        self._buffer: List[AuditEntry] = []

        # Ensure directory exists
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        phase: str,
        action: str,
        paper_id: Optional[str] = None,
        field_name: Optional[str] = None,
        old_value: Optional[Any] = None,
        new_value: Optional[Any] = None,
        reason: str = "",
        actor: str = "system",
        confidence: Optional[float] = None,
        metadata: Optional[Dict] = None
    ) -> AuditEntry:
        """
        Log an audit entry.

        Args:
            phase: Pipeline phase (e.g., 'phase1', 'phase2')
            action: Action performed (e.g., 'extract', 'consensus', 'resolve')
            paper_id: ID of paper being processed
            field_name: Name of field being modified
            old_value: Previous value (if applicable)
            new_value: New value
            reason: Explanation for the action
            actor: Who/what performed the action
            confidence: Confidence score if applicable
            metadata: Additional context

        Returns:
            The created AuditEntry
        """
        entry = AuditEntry(
            timestamp=datetime.now().isoformat(),
            phase=phase,
            action=action,
            paper_id=paper_id,
            field_name=field_name,
            old_value=old_value,
            new_value=new_value,
            reason=reason,
            actor=actor,
            confidence=confidence,
            metadata=metadata or {}
        )

        self._buffer.append(entry)

        if self.auto_flush:
            self.flush()

        return entry

    def log_extraction(
        self,
        paper_id: str,
        model: str,
        extractions: Dict[str, Any],
        confidence_scores: Dict[str, float]
    ):
        """Log an extraction event."""
        self.log(
            phase='extraction',
            action='extract',
            paper_id=paper_id,
            reason=f"Extraction by {model}",
            actor=f"ai:{model}",
            metadata={
                'extractions': extractions,
                'confidences': confidence_scores
            }
        )

    def log_consensus(
        self,
        paper_id: str,
        field_name: str,
        model_values: Dict[str, Any],
        consensus_value: Any,
        status: str
    ):
        """Log a consensus decision."""
        self.log(
            phase='consensus',
            action='consensus',
            paper_id=paper_id,
            field_name=field_name,
            new_value=consensus_value,
            reason=f"Consensus status: {status}",
            actor="system",
            metadata={
                'model_values': model_values,
                'consensus_status': status
            }
        )

    def log_resolution(
        self,
        paper_id: str,
        field_name: str,
        original_values: Dict[str, Any],
        resolved_value: Any,
        resolution_method: str,
        rationale: str,
        resolver: str
    ):
        """Log a discrepancy resolution."""
        self.log(
            phase='resolution',
            action='resolve',
            paper_id=paper_id,
            field_name=field_name,
            old_value=original_values,
            new_value=resolved_value,
            reason=rationale,
            actor=resolver,
            metadata={
                'resolution_method': resolution_method
            }
        )

    def log_human_coding(
        self,
        paper_id: str,
        field_name: str,
        ai_value: Any,
        human_value: Any,
        coder_id: str
    ):
        """Log human coding for verification."""
        self.log(
            phase='human_verification',
            action='code',
            paper_id=paper_id,
            field_name=field_name,
            old_value=ai_value,
            new_value=human_value,
            reason="Human gold standard coding",
            actor=f"human:{coder_id}"
        )

    def flush(self):
        """Write buffered entries to file."""
        if not self._buffer:
            return

        with open(self.log_file, 'a') as f:
            for entry in self._buffer:
                f.write(entry.to_json() + '\n')

        self._buffer = []

    def get_session_entries(
        self,
        since: Optional[str] = None,
        phase: Optional[str] = None,
        paper_id: Optional[str] = None
    ) -> List[AuditEntry]:
        """
        Retrieve entries matching criteria.

        Args:
            since: ISO timestamp to filter from
            phase: Filter by phase
            paper_id: Filter by paper

        Returns:
            List of matching AuditEntry objects
        """
        # Flush buffer first
        self.flush()

        entries = []
        if self.log_file.exists():
            with open(self.log_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    entry = AuditEntry.from_json(line)

                    # Apply filters
                    if since and entry.timestamp < since:
                        continue
                    if phase and entry.phase != phase:
                        continue
                    if paper_id and entry.paper_id != paper_id:
                        continue

                    entries.append(entry)

        return entries

    def get_paper_history(self, paper_id: str) -> List[AuditEntry]:
        """Get complete audit trail for a paper."""
        return self.get_session_entries(paper_id=paper_id)

    def generate_summary(self) -> Dict:
        """Generate summary statistics from audit log."""
        entries = self.get_session_entries()

        summary = {
            'total_entries': len(entries),
            'by_phase': {},
            'by_action': {},
            'by_actor': {},
            'papers_processed': set(),
            'resolutions_count': 0
        }

        for entry in entries:
            # Count by phase
            summary['by_phase'][entry.phase] = summary['by_phase'].get(entry.phase, 0) + 1

            # Count by action
            summary['by_action'][entry.action] = summary['by_action'].get(entry.action, 0) + 1

            # Count by actor
            summary['by_actor'][entry.actor] = summary['by_actor'].get(entry.actor, 0) + 1

            # Track papers
            if entry.paper_id:
                summary['papers_processed'].add(entry.paper_id)

            # Count resolutions
            if entry.action == 'resolve':
                summary['resolutions_count'] += 1

        summary['papers_processed'] = len(summary['papers_processed'])

        return summary


def load_audit_trail(log_file: str) -> List[AuditEntry]:
    """
    Load all entries from an audit log file.

    Args:
        log_file: Path to JSONL audit log

    Returns:
        List of AuditEntry objects
    """
    entries = []
    log_path = Path(log_file)

    if log_path.exists():
        with open(log_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(AuditEntry.from_json(line))

    return entries


def compute_data_hash(data: Any) -> str:
    """
    Compute hash of data for integrity verification.

    Args:
        data: Data to hash (will be JSON serialized)

    Returns:
        SHA-256 hash string
    """
    json_str = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(json_str.encode()).hexdigest()
