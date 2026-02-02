#!/usr/bin/env python3
"""
Confidence calculation and calibration utilities.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class CalibrationResult:
    """Result of confidence calibration."""
    original_confidence: List[float]
    calibrated_confidence: List[float]
    calibration_error: float
    reliability_diagram_data: Dict


class ConfidenceCalculator:
    """Calculate and aggregate confidence scores."""

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initialize calculator with optional field weights.

        Args:
            weights: Dict mapping field names to importance weights
        """
        self.weights = weights or {}

    def aggregate(
        self,
        field_confidences: Dict[str, float],
        method: str = 'weighted_mean'
    ) -> float:
        """
        Aggregate multiple field confidences into single score.

        Args:
            field_confidences: Dict of field_name -> confidence
            method: 'mean', 'weighted_mean', 'min', or 'harmonic'

        Returns:
            Aggregated confidence score
        """
        if not field_confidences:
            return 0.0

        confs = list(field_confidences.values())

        if method == 'mean':
            return np.mean(confs)

        elif method == 'weighted_mean':
            weights = [self.weights.get(f, 1.0) for f in field_confidences.keys()]
            total_weight = sum(weights)
            if total_weight == 0:
                return np.mean(confs)
            return sum(c * w for c, w in zip(confs, weights)) / total_weight

        elif method == 'min':
            return min(confs)

        elif method == 'harmonic':
            # Harmonic mean (penalizes low values more)
            if 0 in confs:
                return 0.0
            return len(confs) / sum(1/c for c in confs)

        else:
            raise ValueError(f"Unknown aggregation method: {method}")

    def from_evidence(
        self,
        evidence_chunks: List[str],
        extraction_value: any,
        relevance_scores: Optional[List[float]] = None
    ) -> float:
        """
        Calculate confidence based on evidence quality.

        Args:
            evidence_chunks: List of supporting text chunks
            extraction_value: The extracted value
            relevance_scores: Optional retrieval similarity scores

        Returns:
            Confidence score (0-1)
        """
        # Base confidence on evidence presence
        if not evidence_chunks:
            return 0.3  # Low confidence without evidence

        # Adjust by number of supporting chunks
        chunk_factor = min(1.0, len(evidence_chunks) / 3)

        # Adjust by relevance scores if available
        relevance_factor = 1.0
        if relevance_scores:
            relevance_factor = np.mean(relevance_scores)

        # Adjust by value specificity
        specificity_factor = 0.8
        if extraction_value and str(extraction_value) not in ['unknown', 'not_specified', 'N/A']:
            specificity_factor = 1.0

        confidence = 0.5 + 0.5 * (chunk_factor * relevance_factor * specificity_factor)
        return min(1.0, max(0.0, confidence))


class IsotonicCalibrator:
    """
    Isotonic regression calibrator for confidence scores.
    Maps raw confidences to calibrated probabilities.
    """

    def __init__(self):
        self._calibration_map = None
        self._is_fitted = False

    def fit(
        self,
        confidences: List[float],
        actual_correct: List[bool]
    ) -> 'IsotonicCalibrator':
        """
        Fit calibrator using validation data.

        Args:
            confidences: Predicted confidence scores
            actual_correct: Whether prediction was actually correct

        Returns:
            Self for chaining
        """
        try:
            from sklearn.isotonic import IsotonicRegression

            ir = IsotonicRegression(out_of_bounds='clip')
            ir.fit(confidences, [int(c) for c in actual_correct])
            self._calibration_map = ir
            self._is_fitted = True

        except ImportError:
            # Fallback: simple binning calibration
            self._calibration_map = self._fit_binned(confidences, actual_correct)
            self._is_fitted = True

        return self

    def _fit_binned(
        self,
        confidences: List[float],
        actual_correct: List[bool],
        n_bins: int = 10
    ) -> Dict:
        """Fallback binned calibration."""
        bins = np.linspace(0, 1, n_bins + 1)
        calibration_map = {}

        for i in range(n_bins):
            bin_mask = [(bins[i] <= c < bins[i+1]) for c in confidences]
            bin_confs = [c for c, m in zip(confidences, bin_mask) if m]
            bin_correct = [c for c, m in zip(actual_correct, bin_mask) if m]

            if bin_correct:
                calibration_map[(bins[i], bins[i+1])] = np.mean(bin_correct)
            else:
                calibration_map[(bins[i], bins[i+1])] = (bins[i] + bins[i+1]) / 2

        return calibration_map

    def calibrate(self, confidence: float) -> float:
        """
        Calibrate a single confidence score.

        Args:
            confidence: Raw confidence (0-1)

        Returns:
            Calibrated confidence
        """
        if not self._is_fitted:
            return confidence

        if hasattr(self._calibration_map, 'predict'):
            # sklearn IsotonicRegression
            return float(self._calibration_map.predict([confidence])[0])
        else:
            # Binned fallback
            for (lo, hi), cal_conf in self._calibration_map.items():
                if lo <= confidence < hi:
                    return cal_conf
            return confidence

    def calibrate_batch(self, confidences: List[float]) -> List[float]:
        """Calibrate a batch of confidence scores."""
        return [self.calibrate(c) for c in confidences]

    def calculate_calibration_error(
        self,
        confidences: List[float],
        actual_correct: List[bool],
        n_bins: int = 10
    ) -> float:
        """
        Calculate Expected Calibration Error (ECE).

        Args:
            confidences: Confidence scores
            actual_correct: Actual correctness
            n_bins: Number of bins for calibration

        Returns:
            ECE score (lower is better)
        """
        bins = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        total = len(confidences)

        for i in range(n_bins):
            bin_mask = [(bins[i] <= c < bins[i+1]) for c in confidences]
            bin_confs = [c for c, m in zip(confidences, bin_mask) if m]
            bin_correct = [c for c, m in zip(actual_correct, bin_mask) if m]

            if bin_correct:
                avg_conf = np.mean(bin_confs)
                accuracy = np.mean(bin_correct)
                ece += (len(bin_correct) / total) * abs(avg_conf - accuracy)

        return ece


def calibrate_confidence(
    raw_confidences: List[float],
    validation_correct: List[bool]
) -> CalibrationResult:
    """
    Convenience function to calibrate confidences.

    Args:
        raw_confidences: Original confidence scores
        validation_correct: Whether predictions were correct

    Returns:
        CalibrationResult with calibrated values and metrics
    """
    calibrator = IsotonicCalibrator()
    calibrator.fit(raw_confidences, validation_correct)

    calibrated = calibrator.calibrate_batch(raw_confidences)
    ece_before = calibrator.calculate_calibration_error(
        raw_confidences, validation_correct
    )
    ece_after = calibrator.calculate_calibration_error(
        calibrated, validation_correct
    )

    # Reliability diagram data
    n_bins = 10
    bins = np.linspace(0, 1, n_bins + 1)
    diagram_data = {
        'bins': bins.tolist(),
        'accuracy_per_bin': [],
        'confidence_per_bin': [],
        'count_per_bin': []
    }

    for i in range(n_bins):
        bin_mask = [(bins[i] <= c < bins[i+1]) for c in calibrated]
        bin_confs = [c for c, m in zip(calibrated, bin_mask) if m]
        bin_correct = [c for c, m in zip(validation_correct, bin_mask) if m]

        diagram_data['confidence_per_bin'].append(
            np.mean(bin_confs) if bin_confs else None
        )
        diagram_data['accuracy_per_bin'].append(
            np.mean(bin_correct) if bin_correct else None
        )
        diagram_data['count_per_bin'].append(len(bin_correct))

    return CalibrationResult(
        original_confidence=raw_confidences,
        calibrated_confidence=calibrated,
        calibration_error=ece_after,
        reliability_diagram_data=diagram_data
    )
