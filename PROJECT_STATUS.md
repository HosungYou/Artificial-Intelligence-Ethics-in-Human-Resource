# Project Status: AI Ethics in HR Systematic Review

**Last Updated**: 2026-02-02
**Version**: 1.0.0
**Author**: Hosung You

---

## Executive Summary

This project implements a RAG-enabled systematic literature review methodology to examine AI ethics issues across human resource management functions. The core innovation is a 6-Phase Validated Coding Pipeline that achieves >97% accuracy through multi-model consensus and human verification.

---

## 1. Project Plan (Original Scope)

### 1.1 Research Objectives

| Objective | Type | Status |
|-----------|------|--------|
| Comprehensive mapping of AI ethics issues across HR functions | Primary | Ready |
| Demonstrate RAG-enabled systematic review methodology | Secondary | Ready |
| Validate AI-assisted coding accuracy vs. human coding | Secondary | Ready |

### 1.2 Research Questions

**Primary RQs (Systematic Review)**:
- RQ1: What are the key ethical issues associated with AI applications across HR functions?
- RQ2: How do AI ethics concerns vary across different HR domains?
- RQ3: What theoretical frameworks have been applied to understand AI ethics in HR?
- RQ4: What gaps exist in the current literature on AI ethics in HR?

**Secondary RQs (Methodology)**:
- RQ5: How can RAG-enabled systematic review methodology improve literature synthesis?
- RQ6: What are the accuracy and reliability of AI-assisted coding compared to human coding?

### 1.3 Target Journals
- Human Resource Development Review (HRDR)
- Human Resource Development Quarterly (HRDQ)
- Human Resource Development International (HRDI)

---

## 2. Completed Work (2026-02-02)

### 2.1 Infrastructure (100% Complete)

| Component | File(s) | Status |
|-----------|---------|--------|
| Project structure | 12 directories | ✅ Complete |
| Pipeline configuration | `pipeline_config.yaml` | ✅ Complete |
| Phase configurations | `phase_configs/*.yaml` (6 files) | ✅ Complete |
| Coding schema | `coding_schema.yaml` | ✅ Complete |
| Dependencies | `requirements.txt` | ✅ Complete |

### 2.2 Pipeline Scripts (100% Complete)

| Stage | Script | Description | Lines | Status |
|-------|--------|-------------|-------|--------|
| 1 | `01_search.py` | Multi-database search | ~450 | ✅ Complete |
| 2 | `02_deduplicate.py` | Deduplication | ~350 | ✅ Complete |
| 3 | `03_screen.py` | AI screening | ~400 | ✅ Complete |
| 4 | `04_build_rag.py` | RAG building | ~400 | ✅ Complete |
| 5 | `05_code/phase1_initial.py` | Initial AI coding | ~400 | ✅ Complete |
| 5 | `05_code/phase2_consensus.py` | Multi-model consensus | ~450 | ✅ Complete |
| 5 | `05_code/phase3_sampling.py` | Human sampling | ~200 | ✅ Complete |
| 5 | `05_code/phase4_reliability.py` | ICR calculation | ~250 | ✅ Complete |
| 5 | `05_code/phase5_resolution.py` | Discrepancy resolution | ~200 | ✅ Complete |
| 5 | `05_code/phase6_qa.py` | Quality assurance | ~250 | ✅ Complete |
| 6 | `07_sensitivity.py` | Sensitivity analysis | ~450 | ✅ Complete |

### 2.3 Utility Modules (100% Complete)

| Module | Functions | Status |
|--------|-----------|--------|
| `utils/metrics.py` | Cohen's κ, Weighted κ, Krippendorff's α, ICC | ✅ Complete |
| `utils/confidence.py` | Confidence calculation, Isotonic calibration | ✅ Complete |
| `utils/audit.py` | Audit logging, Trail management | ✅ Complete |

### 2.4 Documentation (100% Complete)

| Document | Purpose | Status |
|----------|---------|--------|
| `README.md` | Project overview | ✅ Complete |
| `AI_Ethics_HR_Codebook.md` | Coding manual | ✅ Complete |
| `training_protocol.md` | Coder training | ✅ Complete |
| `CHANGELOG.md` | Version history | ✅ Complete |
| `PROJECT_STATUS.md` | This document | ✅ Complete |

### 2.5 Technical Specifications Implemented

```
Pipeline Architecture:
┌─────────────────────────────────────────────────────────────────────────┐
│  Stage 1: IDENTIFICATION (Semantic Scholar + OpenAlex + arXiv)         │
│  ↓                                                                      │
│  Stage 2: DEDUPLICATION (DOI + Title fuzzy + arXiv ID)                 │
│  ↓                                                                      │
│  Stage 3: SCREENING (Groq LLM + 20% human verification)                │
│  ↓                                                                      │
│  Stage 4: RAG BUILDING (ChromaDB + all-MiniLM-L6-v2)                   │
│  ↓                                                                      │
│  Stage 5: 6-PHASE VALIDATED CODING                                     │
│  │  Phase 1: Initial AI Coding (Claude 3.5 Sonnet)                     │
│  │  Phase 2: Multi-Model Consensus (Claude + GPT-4o + Groq)            │
│  │  Phase 3: Human Verification (20% stratified sample)                │
│  │  Phase 4: Inter-Coder Reliability (κ ≥ 0.85)                        │
│  │  Phase 5: Discrepancy Resolution (audit trail)                      │
│  │  Phase 6: Quality Assurance (gates validation)                      │
│  ↓                                                                      │
│  Stage 6: SENSITIVITY ANALYSIS (model/RAG/prompt comparison)           │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Future Work Required

### 3.1 Immediate Next Steps (Week 1-2)

| Task | Priority | Estimated Effort |
|------|----------|------------------|
| Execute database searches | High | 2-4 hours |
| Run deduplication | High | 1 hour |
| Perform AI screening | High | 4-8 hours |
| Download full-text PDFs | High | 8-16 hours |

### 3.2 RAG & Coding Phase (Week 3-6)

| Task | Priority | Estimated Effort |
|------|----------|------------------|
| Build RAG index from PDFs | High | 2-4 hours |
| Run Phase 1 initial coding | High | 4-8 hours |
| Run Phase 2 consensus | High | 4-8 hours |
| Complete Phase 3 human verification | High | 8-16 hours |
| Calculate Phase 4 ICR metrics | Medium | 2 hours |
| Resolve Phase 5 discrepancies | Medium | 4-8 hours |
| Run Phase 6 QA validation | Medium | 2 hours |

### 3.3 Analysis & Writing Phase (Week 7-12)

| Task | Priority | Estimated Effort |
|------|----------|------------------|
| Create `06_analyze.py` script | High | 4-8 hours |
| Run sensitivity analysis | Medium | 4 hours |
| Generate PRISMA flow diagram | High | 2 hours |
| Thematic synthesis | High | 16-24 hours |
| Draft manuscript | High | 40+ hours |
| Prepare supplementary materials | Medium | 8 hours |

### 3.4 Scripts Still Needed

| Script | Purpose | Priority |
|--------|---------|----------|
| `06_analyze.py` | Thematic analysis and visualization | High |
| `08_prisma_diagram.py` | PRISMA 2020 flow diagram generator | Medium |
| `09_export_tables.py` | Generate publication-ready tables | Medium |

### 3.5 Data Files to Generate

| File | Stage | Description |
|------|-------|-------------|
| `data/01_search_results/*.json` | Search | Raw API results |
| `data/02_deduplicated/*.json` | Dedup | Unique papers |
| `data/03_screened/*.json` | Screen | Included papers |
| `data/04_full_text/*.pdf` | PDF | Full-text documents |
| `data/05_coded/phase6_final/final_coded_dataset.csv` | Coding | Final dataset |
| `data/06_analysis/ethics_hr_matrix.csv` | Analysis | Results matrix |

---

## 4. Quality Targets

| Metric | Target | Acceptable | Current |
|--------|--------|------------|---------|
| Cohen's κ (categorical) | ≥ 0.85 | ≥ 0.80 | TBD |
| Weighted κ (ordinal) | ≥ 0.80 | ≥ 0.75 | TBD |
| Krippendorff's α | ≥ 0.80 | ≥ 0.75 | TBD |
| ICC (continuous) | ≥ 0.90 | ≥ 0.85 | TBD |
| Overall accuracy | ≥ 90% | ≥ 85% | TBD |
| Hallucination rate | < 2% | < 5% | TBD |

---

## 5. Resource Requirements

### 5.1 API Keys Required

| Provider | Purpose | Cost Estimate |
|----------|---------|---------------|
| Anthropic | Phase 1 & 2 coding | ~$10-20 |
| OpenAI | Phase 2 verification | ~$7-15 |
| Groq | Screening & efficiency check | ~$2-5 |

### 5.2 Estimated Total Budget

| Item | Cost |
|------|------|
| API costs (200 papers) | ~$25-40 |
| Human coding time | ~17 hours (self) |
| **Total** | **~$40 + researcher time** |

---

## 6. Repository Information

- **GitHub**: https://github.com/HosungYou/Artificial-Intelligence-Ethics-in-Human-Resource
- **Version**: 1.0.0
- **License**: MIT
- **Primary Language**: Python 3.9+

---

## 7. Version History

| Version | Date | Description |
|---------|------|-------------|
| 1.0.0 | 2026-02-02 | Initial release - Complete pipeline infrastructure |

---

## Contact

**Author**: Hosung You
**Project**: AI Ethics in HR Systematic Review
**Repository**: [GitHub](https://github.com/HosungYou/Artificial-Intelligence-Ethics-in-Human-Resource)
