#!/usr/bin/env python3
"""
Statistical metrics for inter-coder reliability.
Implements Cohen's Kappa, Weighted Kappa, Krippendorff's Alpha, and ICC.
"""

import numpy as np
from typing import List, Tuple, Optional, Union
from collections import Counter


def cohens_kappa(y1: List, y2: List) -> float:
    """
    Calculate Cohen's Kappa coefficient for two raters.

    Args:
        y1: List of ratings from rater 1
        y2: List of ratings from rater 2

    Returns:
        Kappa coefficient (-1 to 1)
    """
    if len(y1) != len(y2):
        raise ValueError("Rating lists must have equal length")

    n = len(y1)
    if n == 0:
        return 0.0

    # Observed agreement
    po = sum(1 for a, b in zip(y1, y2) if a == b) / n

    # Expected agreement by chance
    labels = list(set(y1) | set(y2))
    c1 = Counter(y1)
    c2 = Counter(y2)

    pe = sum((c1.get(k, 0) / n) * (c2.get(k, 0) / n) for k in labels)

    # Handle edge case where pe = 1
    if pe == 1.0:
        return 1.0 if po == 1.0 else 0.0

    kappa = (po - pe) / (1 - pe)
    return kappa


def weighted_kappa(
    y1: List,
    y2: List,
    weights: str = 'quadratic'
) -> float:
    """
    Calculate Weighted Kappa for ordinal data.

    Args:
        y1: List of ordinal ratings from rater 1
        y2: List of ordinal ratings from rater 2
        weights: 'linear' or 'quadratic' weighting scheme

    Returns:
        Weighted Kappa coefficient
    """
    if len(y1) != len(y2):
        raise ValueError("Rating lists must have equal length")

    # Get unique sorted labels (assuming ordinal)
    labels = sorted(set(y1) | set(y2))
    n_labels = len(labels)
    label_to_idx = {label: i for i, label in enumerate(labels)}

    n = len(y1)
    if n == 0:
        return 0.0

    # Build weight matrix
    W = np.zeros((n_labels, n_labels))
    for i in range(n_labels):
        for j in range(n_labels):
            if weights == 'quadratic':
                W[i, j] = 1 - ((i - j) ** 2) / ((n_labels - 1) ** 2) if n_labels > 1 else 1
            elif weights == 'linear':
                W[i, j] = 1 - abs(i - j) / (n_labels - 1) if n_labels > 1 else 1
            else:
                raise ValueError(f"Unknown weight scheme: {weights}")

    # Build observed and expected matrices
    O = np.zeros((n_labels, n_labels))
    for a, b in zip(y1, y2):
        O[label_to_idx[a], label_to_idx[b]] += 1
    O = O / n

    # Expected frequencies
    row_marginals = O.sum(axis=1)
    col_marginals = O.sum(axis=0)
    E = np.outer(row_marginals, col_marginals)

    # Weighted agreement
    po_w = np.sum(W * O)
    pe_w = np.sum(W * E)

    if pe_w == 1.0:
        return 1.0 if po_w == 1.0 else 0.0

    return (po_w - pe_w) / (1 - pe_w)


def krippendorff_alpha(
    data: List[List],
    level: str = 'nominal'
) -> float:
    """
    Calculate Krippendorff's Alpha for multiple coders.

    Args:
        data: List of lists where each inner list is one coder's ratings
              data[coder][item] = rating (None for missing)
        level: 'nominal', 'ordinal', 'interval', or 'ratio'

    Returns:
        Alpha coefficient
    """
    # Build units matrix: units[item] = [ratings from different coders]
    n_coders = len(data)
    n_items = len(data[0]) if data else 0

    if n_items == 0 or n_coders < 2:
        return 0.0

    # Collect all non-missing values
    all_values = []
    units = []
    for i in range(n_items):
        unit_values = [data[c][i] for c in range(n_coders) if data[c][i] is not None]
        if len(unit_values) >= 2:
            units.append(unit_values)
            all_values.extend(unit_values)

    if not units:
        return 0.0

    # Define difference function based on level
    def difference(v1, v2):
        if level == 'nominal':
            return 0 if v1 == v2 else 1
        elif level == 'ordinal':
            values = sorted(set(all_values))
            i1, i2 = values.index(v1), values.index(v2)
            return (i1 - i2) ** 2
        elif level in ('interval', 'ratio'):
            return (float(v1) - float(v2)) ** 2
        else:
            raise ValueError(f"Unknown level: {level}")

    # Calculate observed disagreement
    observed_disagreement = 0
    n_pairs_observed = 0

    for unit_values in units:
        m = len(unit_values)
        if m < 2:
            continue

        for i in range(m):
            for j in range(i + 1, m):
                observed_disagreement += difference(unit_values[i], unit_values[j])
                n_pairs_observed += 1

    if n_pairs_observed == 0:
        return 0.0

    Do = observed_disagreement / n_pairs_observed

    # Calculate expected disagreement
    n_total = len(all_values)
    value_counts = Counter(all_values)

    expected_disagreement = 0
    n_pairs_expected = 0

    for v1, c1 in value_counts.items():
        for v2, c2 in value_counts.items():
            pairs = c1 * c2 if v1 != v2 else c1 * (c1 - 1)
            expected_disagreement += difference(v1, v2) * pairs
            n_pairs_expected += pairs

    if n_pairs_expected == 0:
        return 0.0

    De = expected_disagreement / n_pairs_expected

    if De == 0:
        return 1.0

    return 1 - (Do / De)


def intraclass_correlation(
    ratings: np.ndarray,
    model: str = 'ICC(2,1)'
) -> Tuple[float, Tuple[float, float]]:
    """
    Calculate Intraclass Correlation Coefficient.

    Args:
        ratings: 2D array (n_subjects x n_raters)
        model: ICC model type ('ICC(1,1)', 'ICC(2,1)', 'ICC(3,1)')

    Returns:
        Tuple of (ICC value, (lower CI, upper CI))
    """
    n, k = ratings.shape

    # Calculate means
    subject_means = ratings.mean(axis=1)
    rater_means = ratings.mean(axis=0)
    grand_mean = ratings.mean()

    # Sum of squares
    SS_total = np.sum((ratings - grand_mean) ** 2)
    SS_rows = k * np.sum((subject_means - grand_mean) ** 2)  # Between subjects
    SS_cols = n * np.sum((rater_means - grand_mean) ** 2)    # Between raters
    SS_error = SS_total - SS_rows - SS_cols                   # Residual

    # Mean squares
    MS_rows = SS_rows / (n - 1)
    MS_cols = SS_cols / (k - 1) if k > 1 else 0
    MS_error = SS_error / ((n - 1) * (k - 1)) if (n - 1) * (k - 1) > 0 else 0

    # Calculate ICC based on model
    if model == 'ICC(1,1)':
        # One-way random
        MS_within = (SS_cols + SS_error) / (n * (k - 1))
        icc = (MS_rows - MS_within) / (MS_rows + (k - 1) * MS_within)
    elif model == 'ICC(2,1)':
        # Two-way random, single measurement
        icc = (MS_rows - MS_error) / (MS_rows + (k - 1) * MS_error +
              (k / n) * (MS_cols - MS_error))
    elif model == 'ICC(3,1)':
        # Two-way mixed, single measurement
        icc = (MS_rows - MS_error) / (MS_rows + (k - 1) * MS_error)
    else:
        raise ValueError(f"Unknown ICC model: {model}")

    # Confidence intervals (simplified approximation)
    # For more accurate CIs, use scipy.stats
    icc = max(-1, min(1, icc))  # Bound between -1 and 1

    # Rough 95% CI approximation
    se = np.sqrt((1 - icc ** 2) / (n - 2)) if n > 2 else 0.5
    ci_lower = icc - 1.96 * se
    ci_upper = icc + 1.96 * se

    return icc, (max(-1, ci_lower), min(1, ci_upper))


def agreement_rate(y1: List, y2: List) -> float:
    """
    Calculate simple agreement rate (percentage agreement).

    Args:
        y1: Ratings from rater 1
        y2: Ratings from rater 2

    Returns:
        Proportion of matching ratings (0 to 1)
    """
    if len(y1) != len(y2):
        raise ValueError("Rating lists must have equal length")

    if len(y1) == 0:
        return 0.0

    matches = sum(1 for a, b in zip(y1, y2) if a == b)
    return matches / len(y1)


def interpret_kappa(kappa: float) -> str:
    """
    Interpret Kappa value using Landis & Koch (1977) benchmarks.

    Args:
        kappa: Kappa coefficient

    Returns:
        Interpretation string
    """
    if kappa < 0:
        return "Poor (less than chance)"
    elif kappa < 0.20:
        return "Slight"
    elif kappa < 0.40:
        return "Fair"
    elif kappa < 0.60:
        return "Moderate"
    elif kappa < 0.80:
        return "Substantial"
    else:
        return "Almost perfect"
