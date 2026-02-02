# CLAUDE.md - AI Ethics in HR Systematic Review

This file provides guidance to Claude Code when working in this repository.

---

## Project Overview

**Project**: AI Ethics in Human Resource: A RAG-Enabled Systematic Literature Review
**Author**: Hosung You
**Version**: 1.0.0
**Date Started**: 2026-02-02

This project implements a comprehensive RAG-enabled systematic literature review examining AI ethics issues across all human resource management functions.

---

## 📋 MANDATORY: PRD and SPEC Reference

**CRITICAL**: Before making ANY changes to this project, Claude MUST:

1. **Read PRD** (`docs/PRD.md`) to understand:
   - Current project phase and status
   - Requirements and acceptance criteria
   - Success metrics and targets
   - Timeline and dependencies

2. **Read SPEC** (`docs/SPEC.md`) to understand:
   - System architecture
   - Component specifications
   - Data schemas
   - API interfaces
   - Error handling procedures

3. **Update Progress** after completing tasks:
   - Update PRD status tables
   - Check off acceptance criteria
   - Log any deviations or issues

### Quick Reference Links

| Document | Purpose | Location |
|----------|---------|----------|
| **PRD** | Requirements & Progress | `docs/PRD.md` |
| **SPEC** | Technical Specifications | `docs/SPEC.md` |
| **Research Plan** | Original Methodology | `docs/RESEARCH_PLAN.md` |
| **Coding Schema** | Field Definitions | `codebook/coding_schema.yaml` |
| **Codebook** | Coding Manual | `codebook/AI_Ethics_HR_Codebook.md` |

---

## Current Project Status

### Phase Tracker

| Phase | Status | PRD Section |
|-------|--------|-------------|
| **Phase 1: Infrastructure** | ✅ Complete | PRD §6.1 |
| **Phase 2: Data Collection** | ⏳ Current | PRD §6.2 |
| **Phase 3: RAG & Coding** | 🔜 Upcoming | PRD §6.3 |
| **Phase 4: Analysis & Writing** | 🔜 Upcoming | PRD §6.4 |

### Next Actions (Phase 2)

```bash
# 1. Execute database searches
python scripts/01_search.py

# 2. Run deduplication
python scripts/02_deduplicate.py

# 3. Run AI screening
python scripts/03_screen.py

# 4. Download full-text PDFs (manual step)
# Place PDFs in data/04_full_text/
```

---

## Development Commands

### Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with:
# - ANTHROPIC_API_KEY
# - OPENAI_API_KEY
# - GROQ_API_KEY
```

### Running the Pipeline

```bash
# Full pipeline execution order:
python scripts/01_search.py           # Search databases
python scripts/02_deduplicate.py      # Remove duplicates
python scripts/03_screen.py           # AI screening
python scripts/04_build_rag.py        # Build RAG index

# 6-Phase Coding:
python scripts/05_code/phase1_initial.py
python scripts/05_code/phase2_consensus.py
python scripts/05_code/phase3_sampling.py
python scripts/05_code/phase4_reliability.py
python scripts/05_code/phase5_resolution.py
python scripts/05_code/phase6_qa.py

# Sensitivity analysis:
python scripts/07_sensitivity.py
```

---

## Key Quality Targets

| Metric | Target | Acceptable | Check in |
|--------|--------|------------|----------|
| Cohen's κ (categorical) | ≥ 0.85 | ≥ 0.80 | Phase 4 |
| Weighted κ (ordinal) | ≥ 0.80 | ≥ 0.75 | Phase 4 |
| Krippendorff's α | ≥ 0.80 | ≥ 0.75 | Phase 4 |
| ICC (continuous) | ≥ 0.90 | ≥ 0.85 | Phase 4 |
| Overall accuracy | ≥ 90% | ≥ 85% | Phase 6 |
| Hallucination rate | < 2% | < 5% | Phase 6 |

---

## File Structure

```
AI-Ethics-HR-Review/
├── docs/
│   ├── PRD.md              # 📋 Requirements & Progress
│   ├── SPEC.md             # 📐 Technical Specifications
│   └── RESEARCH_PLAN.md    # 📝 Original Research Plan
├── configs/                 # Configuration files
├── scripts/                 # Pipeline scripts
│   ├── 01_search.py
│   ├── 02_deduplicate.py
│   ├── 03_screen.py
│   ├── 04_build_rag.py
│   ├── 05_code/            # 6-Phase Coding Module
│   ├── 07_sensitivity.py
│   └── utils/              # Utility modules
├── data/                   # Data directories (gitignored)
├── codebook/               # Coding documentation
├── manuscript/             # Paper drafts
└── validation/             # Validation outputs
```

---

## Working with This Project

### Before Starting Any Task

1. **Check current phase** in PRD §6
2. **Review relevant SPEC sections** for the task
3. **Verify dependencies** are met (PRD §7)
4. **Check acceptance criteria** (PRD §9)

### After Completing Tasks

1. **Update PRD status tables**
2. **Log any issues** in PRD §8 (Risks)
3. **Commit with descriptive message**

### Code Style

- Python 3.9+
- Type hints for all functions
- Docstrings for classes and public methods
- Follow existing patterns in codebase

---

## API Keys Required

| Provider | Purpose | Environment Variable |
|----------|---------|---------------------|
| Anthropic | Phase 1 & 2 coding | `ANTHROPIC_API_KEY` |
| OpenAI | Phase 2 verification | `OPENAI_API_KEY` |
| Groq | Screening & efficiency | `GROQ_API_KEY` |

---

## Research Context

### Target Journals
- Human Resource Development Review (HRDR)
- Human Resource Development Quarterly (HRDQ)
- Human Resource Development International (HRDI)

### Research Questions

**Primary (Systematic Review)**:
- RQ1: Key ethical issues across HR functions?
- RQ2: How do concerns vary by HR domain?
- RQ3: What theoretical frameworks applied?
- RQ4: What gaps exist in literature?

**Secondary (Methodology)**:
- RQ5: RAG methodology effectiveness?
- RQ6: AI-human coding reliability?

---

## Important Notes

1. **Data Privacy**: Full-text PDFs are gitignored
2. **API Costs**: Budget ~$40 for 200 papers
3. **Human Verification**: 20% sample coded manually
4. **Audit Trail**: All decisions logged in `data/05_coded/`

---

## Related Documents

- GitHub Repository: https://github.com/HosungYou/Artificial-Intelligence-Ethics-in-Human-Resource
- Release Notes: See `CHANGELOG.md`
- Project Status: See `PROJECT_STATUS.md`

---

*Last Updated: 2026-02-02*
*Version: 1.0.0*
