# ADR-001: Initial Coding Schema Design (v1.0.0)

**Date**: 2026-02-02 (Retrospective)
**Status**: Accepted (Superseded by ADR-002 for extensions)
**Deciders**: Hosung You
**Version**: v1.0.0

## Context

Starting a systematic literature review on AI ethics in Human Resource Management required a structured coding schema to extract consistent data from academic papers using RAG-enabled AI coding.

## Decision Drivers

* **Systematic Review Requirements**: Need standardized data extraction across all papers
* **PRISMA 2020 Compliance**: Schema must support reproducible, auditable coding
* **AI-Assisted Coding**: Fields must be extractable by LLMs with confidence scores
* **Inter-Coder Reliability**: Fields need clear definitions for reliability calculation
* **Research Questions**: Schema must capture data to answer RQ1-RQ4

## Considered Options

### Option 1: Minimal Schema
Only capture basic metadata and ethical issues presence (boolean).

### Option 2: Comprehensive Schema (Chosen)
Capture detailed information across 5 major categories with typed fields.

### Option 3: Free-text Extraction
Extract information as free-text and code post-hoc.

## Decision

We chose **Option 2: Comprehensive Schema** with the following structure:

### Schema Categories

1. **study_metadata**: ID, authors, year, title, journal, DOI, country, methodology
2. **hr_function**: Primary and secondary HR functions (9 categories)
3. **ai_technology**: Technology types (9 categories) and specific tools
4. **ethical_issues**: 6 principles × (mentioned, type, severity)
5. **theoretical_framework**: Applied, name, category
6. **key_findings**: Summary, recommendations, limitations
7. **quality_indicators**: Peer-reviewed, empirical, sample size, geographic scope
8. **coding_metadata**: Coder ID, date, phase, confidence scores

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| 6 ethical principles | Based on established AI ethics frameworks (Floridi, IEEE) |
| Ordinal severity (4 levels) | Enables weighted kappa calculation |
| Multi-select for types | Papers often discuss multiple sub-issues |
| Conditional fields | Reduce coding burden when not applicable |
| Confidence scores | Required for 6-phase validation pipeline |

### Consequences

**Positive:**
- Comprehensive data for research questions
- Clear field definitions for reliability
- Supports both AI and human coding
- Extensible for future needs

**Negative:**
- Complex extraction prompts
- Longer coding time per paper
- Some fields may have lower reliability

## Related

- [ADR-002](./ADR-002-coding-schema-extension-v1.1.md) - Schema extension for v1.1.0
- [Coding Schema](../../codebook/coding_schema.yaml)
- [Codebook](../../codebook/AI_Ethics_HR_Codebook.md)

---

*Retrospective ADR documenting initial design decisions*
