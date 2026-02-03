# ADR-002: Coding Schema Extension for Stance, Solutions, and Temporal Analysis

**Date**: 2026-02-03
**Status**: Accepted
**Deciders**: Hosung You, Claude Code (AI Assistant)
**Version**: v1.1.0

## Context

The initial coding schema (v1.0.0) captured basic information about AI ethics in HR literature:
- HR functions
- AI technology types
- 6 ethical principles (fairness, transparency, accountability, privacy, autonomy, wellbeing)
- Theoretical frameworks
- Key findings

However, after reviewing the research goals and analysis requirements, we identified gaps:

1. **No stance tracking**: We couldn't analyze whether papers were optimistic, critical, or balanced toward AI
2. **No solution categorization**: Proposed solutions were captured in free-text only
3. **No temporal structure**: Couldn't track how concepts evolved over the 2015-2025 period
4. **Limited network analysis**: Needed structured data for co-occurrence analysis

## Decision Drivers

* **Research Question Support**: RQ3 asks about theoretical frameworks AND solutions proposed
* **Trend Analysis Need**: Want to show how AI ethics concerns evolved over time
* **Publication Requirements**: Target journals (HRDR, HRDQ) expect sophisticated analysis
* **Methodological Rigor**: Need reliable, codeable fields for inter-coder reliability
* **Network Visualization**: Co-occurrence networks require structured categorical data

## Considered Options

### Option 1: Minimal Extension (Stance Only)
Add only `overall_tone` field to capture paper stance.

**Pros**: Simple, fast to implement
**Cons**: Doesn't address solution categorization or temporal analysis needs

### Option 2: Comprehensive Extension (Chosen)
Add three new sections:
- Stance Classification (tone, argument basis, per-principle stance)
- Solution Taxonomy (technical, organizational, regulatory with multi-select)
- Temporal Metadata (research periods)

**Pros**: Addresses all identified gaps, enables rich analysis
**Cons**: More complex extraction, longer coding time per paper

### Option 3: Post-hoc Coding
Keep schema minimal, do post-hoc manual coding for analysis.

**Pros**: Faster initial coding
**Cons**: Inconsistent, not reproducible, defeats RAG automation purpose

## Decision

We chose **Option 2: Comprehensive Extension** because:

1. **Alignment with Research Goals**: The extended fields directly support answering all research questions
2. **One-time Investment**: Coding once with richer schema avoids re-processing papers later
3. **Analysis Capability**: Enables co-occurrence networks, evolution tracking, and solution mapping
4. **Reliability**: Structured fields can be validated with inter-coder reliability metrics

### Schema Design Decisions

#### Stance Classification

| Field | Type | Rationale |
|-------|------|-----------|
| `overall_tone` | Categorical (4 options) | Captures dominant framing without over-complicating |
| `argument_basis` | Categorical (3 options) | Distinguishes evidence vs. opinion-based claims |
| `per_principle_stance` | Ordinal (4 levels) | Allows nuanced tracking per ethical principle |

**Key Decision**: Per-principle stance uses ordinal scale (4→1: concern_high → solution_focused) to enable weighted kappa calculation.

#### Solution Taxonomy

Three categories chosen based on literature review:
- **Technical**: AI/ML-specific interventions (algorithm audit, XAI, fairness constraints)
- **Organizational**: Governance and process changes (oversight, committees, training)
- **Regulatory**: External governance (laws, standards, certification)

**Key Decision**: Multi-select within categories because papers often propose multiple solutions.

**Key Decision**: Added `empirical_validation` sub-field to track whether solutions were tested.

#### Temporal Metadata

Four research periods chosen:
| Period | Rationale |
|--------|-----------|
| 2015-2017 | Pre-GDPR, foundational AI ethics discussions |
| 2018-2020 | GDPR implementation, corporate AI ethics emergence |
| 2021-2023 | EU AI Act drafting, maturation of field |
| 2024-2025 | Generative AI disruption (ChatGPT era) |

**Key Decision**: 3-year windows balance granularity with sufficient papers per period.

### Consequences

**Positive:**
- Enables stance evolution analysis (critical → balanced trend hypothesis)
- Supports solution emergence tracking (technical → regulatory shift)
- Powers concept co-occurrence network visualization
- Maintains backward compatibility with v1.0.0 coded papers (new fields optional)
- All new fields have defined reliability metrics

**Negative:**
- Increased extraction prompt complexity (~50% longer)
- Higher per-paper API cost (more tokens)
- Requires codebook training update for human coders
- Per-principle stance may have lower reliability (conditional field)

## Implementation Notes

### Files Modified

```
codebook/
├── coding_schema.yaml          # +217 lines (new sections)
└── AI_Ethics_HR_Codebook.md    # +259 lines (coding instructions)

scripts/
├── 05_code/phase1_initial.py   # Extended prompts and processing
└── utils/metrics.py            # Added Jaccard, multi-select metrics

configs/phase_configs/
├── phase1_config.yaml          # +5 confidence thresholds
├── phase2_models.yaml          # +13 lines (consensus rules)
└── phase4_thresholds.yaml      # +41 lines (reliability targets)
```

### New Analysis Module

```
scripts/06_analyze/
├── __init__.py
├── cooccurrence_network.py     # NetworkX-based analysis
├── evolution_tracking.py       # Temporal trend analysis
└── visualizations.py           # Publication figures
```

### Reliability Targets

| New Field | Metric | Target |
|-----------|--------|--------|
| stance_overall | Cohen's κ | ≥ 0.80 |
| per_principle_stance | Weighted κ | ≥ 0.75 |
| solutions_proposed | Cohen's κ | ≥ 0.90 |
| solution_types (multi) | Krippendorff's α | ≥ 0.75 |

### Extraction Prompt Extension

Added to Phase 1 prompt:
```
6. **Stance Classification** (NEW):
   a) Overall Tone: AI_optimistic, AI_critical, balanced, neutral
   b) Argument Basis: evidence_based, opinion_based, mixed
   c) Per-Principle Stance: concern_high(4) → solution_focused(1)

7. **Solution Taxonomy** (NEW):
   - Technical: algorithm_audit, explainable_AI, ...
   - Organizational: human_oversight, ethics_committee, ...
   - Regulatory: legislation, industry_standards, ...
```

## Alternatives Not Chosen

### Free-text Solution Extraction
Considered extracting solutions as free-text and clustering post-hoc.
**Rejected**: Inconsistent results, not reproducible, harder reliability calculation.

### Binary Stance (Pro/Con)
Considered simple binary stance classification.
**Rejected**: Too simplistic, loses nuance of "balanced" papers.

### Continuous Temporal (by Year)
Considered coding by exact publication year.
**Rejected**: Too sparse for trend analysis (some years have few papers).

## Related

- [ADR-001](./ADR-001-initial-schema-design.md) - Initial v1.0.0 schema design
- [Coding Schema v1.1.0](../../codebook/coding_schema.yaml)
- [Codebook v1.1](../../codebook/AI_Ethics_HR_Codebook.md)
- [GitHub Release v1.1.0](https://github.com/HosungYou/Artificial-Intelligence-Ethics-in-Human-Resource/releases/tag/v1.1.0)

## Session Notes

### Planning Discussion (2026-02-03)

**Initial Request**: Extend coding schema with stance classification, solution taxonomy, and temporal tracking.

**Key Discussion Points**:
1. Stance classification should capture both overall tone AND per-principle nuance
2. Solution taxonomy needed three tiers (technical → organizational → regulatory)
3. Research periods should align with major AI ethics milestones (GDPR, EU AI Act, ChatGPT)
4. Multi-select fields need specialized reliability metrics (Jaccard, Krippendorff's α)

**Implementation Approach**:
1. Extend schema YAML first (source of truth)
2. Update codebook documentation
3. Modify extraction prompts
4. Add reliability metrics
5. Create analysis scripts
6. Verify all syntax

**Verification**:
- All YAML files validated: `python3 -c "import yaml; yaml.safe_load(...)"`
- All Python files validated: `python3 -m py_compile ...`
- 13 files changed, +3,054 lines

---

*ADR created following [MADR](https://adr.github.io/madr/) template*
