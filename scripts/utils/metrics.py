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


# ============================================
# v1.1.0: Multi-select Agreement Metrics
# ============================================

def jaccard_similarity(set1: set, set2: set) -> float:
    """
    Calculate Jaccard similarity coefficient between two sets.

    Jaccard = |intersection| / |union|

    Args:
        set1: First set of items
        set2: Second set of items

    Returns:
        Jaccard coefficient (0 to 1)
    """
    if not set1 and not set2:
        return 1.0  # Both empty = perfect agreement

    if not set1 or not set2:
        return 0.0  # One empty, one not = no agreement

    intersection = len(set1 & set2)
    union = len(set1 | set2)

    return intersection / union if union > 0 else 0.0


def dice_coefficient(set1: set, set2: set) -> float:
    """
    Calculate Dice coefficient (Sorensen-Dice) between two sets.

    Dice = 2 * |intersection| / (|set1| + |set2|)

    Args:
        set1: First set of items
        set2: Second set of items

    Returns:
        Dice coefficient (0 to 1)
    """
    if not set1 and not set2:
        return 1.0

    if not set1 or not set2:
        return 0.0

    intersection = len(set1 & set2)
    total = len(set1) + len(set2)

    return (2 * intersection) / total if total > 0 else 0.0


def multi_select_agreement(
    y1: List[List],
    y2: List[List],
    metric: str = 'jaccard'
) -> Tuple[float, float]:
    """
    Calculate agreement for multi-select (set) fields across multiple items.

    For each item, computes the similarity between two coders' selections,
    then returns the mean and standard deviation.

    Args:
        y1: List of lists - each inner list is one coder's selections for an item
        y2: List of lists - second coder's selections (same length as y1)
        metric: 'jaccard' or 'dice'

    Returns:
        Tuple of (mean_agreement, std_agreement)
    """
    if len(y1) != len(y2):
        raise ValueError("Rating lists must have equal length")

    if len(y1) == 0:
        return 0.0, 0.0

    # Choose similarity function
    if metric == 'jaccard':
        sim_func = jaccard_similarity
    elif metric == 'dice':
        sim_func = dice_coefficient
    else:
        raise ValueError(f"Unknown metric: {metric}. Use 'jaccard' or 'dice'")

    # Calculate per-item similarity
    similarities = []
    for items1, items2 in zip(y1, y2):
        set1 = set(items1) if items1 else set()
        set2 = set(items2) if items2 else set()
        similarities.append(sim_func(set1, set2))

    mean_sim = np.mean(similarities)
    std_sim = np.std(similarities)

    return float(mean_sim), float(std_sim)


def multi_select_krippendorff(
    data: List[List[List]],
    level: str = 'nominal'
) -> float:
    """
    Calculate Krippendorff's Alpha for multi-select fields.

    Treats each possible value as a binary presence/absence variable
    and computes alpha across all variables.

    Args:
        data: List of coders, each coder has a list of items,
              each item is a list of selected values
              data[coder][item] = [selected values]
        level: 'nominal' (default) for multi-select

    Returns:
        Average alpha across all binary variables
    """
    n_coders = len(data)
    n_items = len(data[0]) if data else 0

    if n_items == 0 or n_coders < 2:
        return 0.0

    # Get all possible values across all coders and items
    all_values = set()
    for coder_data in data:
        for item in coder_data:
            if item:
                all_values.update(item)

    if not all_values:
        return 0.0

    # For each possible value, create binary ratings and compute alpha
    alphas = []
    for value in all_values:
        # Create binary matrix: was this value selected?
        binary_data = []
        for coder_data in data:
            coder_binary = []
            for item in coder_data:
                if item is None:
                    coder_binary.append(None)
                else:
                    coder_binary.append(1 if value in item else 0)
            binary_data.append(coder_binary)

        # Compute alpha for this binary variable
        alpha = krippendorff_alpha(binary_data, level='nominal')
        alphas.append(alpha)

    return float(np.mean(alphas)) if alphas else 0.0


def ordinal_distance(v1: int, v2: int, n_levels: int) -> float:
    """
    Calculate normalized ordinal distance between two values.

    Args:
        v1: First ordinal value (1 to n_levels)
        v2: Second ordinal value (1 to n_levels)
        n_levels: Total number of ordinal levels

    Returns:
        Normalized distance (0 to 1)
    """
    if n_levels <= 1:
        return 0.0
    return abs(v1 - v2) / (n_levels - 1)


def ordinal_agreement(
    y1: List[int],
    y2: List[int],
    n_levels: int
) -> Tuple[float, float]:
    """
    Calculate ordinal agreement with tolerance.

    Returns both exact match rate and "close" match rate
    (within 1 level).

    Args:
        y1: Ordinal ratings from rater 1
        y2: Ordinal ratings from rater 2
        n_levels: Number of ordinal levels

    Returns:
        Tuple of (exact_match_rate, close_match_rate)
    """
    if len(y1) != len(y2):
        raise ValueError("Rating lists must have equal length")

    if len(y1) == 0:
        return 0.0, 0.0

    exact_matches = 0
    close_matches = 0

    for a, b in zip(y1, y2):
        if a == b:
            exact_matches += 1
            close_matches += 1
        elif abs(a - b) <= 1:
            close_matches += 1

    n = len(y1)
    return exact_matches / n, close_matches / n
