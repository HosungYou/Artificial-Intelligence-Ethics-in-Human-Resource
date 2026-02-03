# Product Requirements Document (PRD)
## AI Ethics in Human Resource: A RAG-Enabled Systematic Literature Review

**Version**: 1.1.0
**Date**: 2026-02-03
**Author**: Hosung You
**Status**: Phase 2 - Data Collection (Screening Complete)

---

## 1. Executive Summary

### 1.1 Project Vision
Develop a comprehensive RAG-enabled systematic literature review examining AI ethics issues across all human resource management functions, while simultaneously validating a novel multi-model consensus coding methodology.

### 1.2 Success Metrics

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| Pipeline Scripts Complete | 11 | 11 | ✅ |
| Configuration Files Complete | 7 | 7 | ✅ |
| Documentation Complete | 5 | 5 | ✅ |
| Database Searches Executed | 6 | 6 | ✅ |
| Papers Retrieved | 1,500-2,000 | 7,897 | ✅ (exceeded) |
| Papers After Deduplication | - | 7,121 | ✅ |
| Papers Screened | 100% | 100% | ✅ |
| Papers Included for Full-Text | 150-400 | 1,118 | ✅ |
| Full-Text PDFs Downloaded | ≥60% | 0% | ⏳ |
| Papers Coded | 100% | 0% | ⏳ |
| Inter-Coder Reliability (κ) | ≥ 0.85 | TBD | ⏳ |
| Manuscript Draft | 1 | 0 | ⏳ |

---

## 2. Problem Statement

### 2.1 Current State
- AI ethics in HR literature is fragmented across disciplines
- No comprehensive systematic review covers all HR functions
- Traditional systematic review methods are time-intensive
- AI-assisted coding lacks validated reliability protocols

### 2.2 Target State
- First comprehensive mapping of AI ethics across HR functions
- Validated RAG-enabled systematic review methodology
- Multi-model consensus achieving >97% accuracy
- Publication-ready manuscript for HRD journals

---

## 3. Requirements

### 3.1 Functional Requirements

#### FR-1: Literature Search (Priority: High)
- **FR-1.1**: Search Semantic Scholar API with rate limiting
- **FR-1.2**: Search OpenAlex API with polite pool compliance
- **FR-1.3**: Search arXiv API with delay compliance
- **FR-1.4**: Export results in standardized JSON format
- **FR-1.5**: Track PRISMA identification metrics

#### FR-2: Deduplication (Priority: High)
- **FR-2.1**: DOI-based exact matching
- **FR-2.2**: Title fuzzy matching (threshold ≥ 85%)
- **FR-2.3**: arXiv ID extraction and matching
- **FR-2.4**: Source prioritization and metadata merging

#### FR-3: AI Screening (Priority: High)
- **FR-3.1**: Groq LLM integration for screening
- **FR-3.2**: Inclusion/exclusion criteria application
- **FR-3.3**: Confidence score generation
- **FR-3.4**: Stratified human verification sampling (20%)

#### FR-4: RAG Building (Priority: High)
- **FR-4.1**: PDF text extraction with PyMuPDF
- **FR-4.2**: Text chunking (1000 tokens, 200 overlap)
- **FR-4.3**: Embedding generation (all-MiniLM-L6-v2)
- **FR-4.4**: ChromaDB vector store construction
- **FR-4.5**: Retrieval validation

#### FR-5: 6-Phase Validated Coding (Priority: Critical)
- **FR-5.1**: Phase 1 - Claude 3.5 Sonnet initial coding with RAG
- **FR-5.2**: Phase 2 - Multi-model consensus (Claude + GPT-4o + Groq)
- **FR-5.3**: Phase 3 - Stratified human verification sampling
- **FR-5.4**: Phase 4 - Inter-coder reliability calculation
- **FR-5.5**: Phase 5 - Discrepancy resolution with audit trail
- **FR-5.6**: Phase 6 - Quality assurance gate validation

#### FR-6: Sensitivity Analysis (Priority: Medium)
- **FR-6.1**: Model comparison (Claude vs GPT-4o vs Groq)
- **FR-6.2**: RAG configuration sensitivity
- **FR-6.3**: Temperature sensitivity
- **FR-6.4**: Prompt variation sensitivity

#### FR-7: Analysis & Synthesis (Priority: High)
- **FR-7.1**: Ethics × HR function cross-tabulation
- **FR-7.2**: Thematic synthesis
- **FR-7.3**: Gap analysis
- **FR-7.4**: PRISMA flow diagram generation

### 3.2 Non-Functional Requirements

#### NFR-1: Quality
- Inter-coder reliability κ ≥ 0.85
- Hallucination rate < 2%
- Systematic bias < 5%

#### NFR-2: Reproducibility
- Complete audit trail for all decisions
- Configuration version control
- Random seed documentation

#### NFR-3: Cost Efficiency
- API costs < $50 total
- Processing time < 20 hours

---

## 4. Scope

### 4.1 In Scope
- Systematic review of AI ethics in HR (2015-2025)
- RAG-enabled methodology validation
- Multi-model consensus coding
- Publication-ready manuscript

### 4.2 Out of Scope
- Meta-analysis (qualitative synthesis only)
- Primary data collection
- System deployment/productization

---

## 5. User Stories

### US-1: As a researcher
**I want to** search multiple academic databases automatically
**So that** I can comprehensively identify relevant literature

### US-2: As a researcher
**I want to** have AI assist with paper screening
**So that** I can process large volumes efficiently

### US-3: As a researcher
**I want to** code papers using RAG-enhanced LLM
**So that** I can extract information with source attribution

### US-4: As a researcher
**I want to** validate AI coding against human standards
**So that** I can trust the reliability of results

### US-5: As a reviewer
**I want to** see complete audit trails
**So that** I can verify methodological rigor

---

## 6. Timeline

### Phase 1: Infrastructure (Complete ✅)
**Duration**: Week 1-2
**Deliverables**:
- [x] Project structure
- [x] Pipeline configuration
- [x] All scripts implemented
- [x] Documentation complete

### Phase 2: Data Collection (Current ⏳)
**Duration**: Week 2-4
**Deliverables**:
- [x] Execute database searches (Scopus: 4,851 | WoS: 1,307 | PubMed: 437 | ERIC: 302 | Semantic Scholar: 500 | OpenAlex: 500)
- [x] Run deduplication (7,897 → 7,121, 776 duplicates removed)
- [x] Complete AI screening (Include: 1,118 | Exclude: 5,836 | Uncertain: 167)
- [ ] Human verification of screening sample (1,994 papers flagged)
- [ ] Download full-text PDFs (50 Open Access, ~1,068 institutional access required)

### Phase 3: RAG & Coding (Upcoming)
**Duration**: Week 4-8
**Deliverables**:
- [ ] Build RAG index
- [ ] Execute 6-Phase coding pipeline
- [ ] Calculate reliability metrics
- [ ] Resolve discrepancies

### Phase 4: Analysis & Writing (Upcoming)
**Duration**: Week 8-12
**Deliverables**:
- [ ] Run sensitivity analysis
- [ ] Generate PRISMA diagram
- [ ] Thematic synthesis
- [ ] Draft manuscript

---

## 7. Dependencies

### 7.1 External Dependencies
| Dependency | Provider | Status |
|------------|----------|--------|
| Anthropic API | Anthropic | Required |
| OpenAI API | OpenAI | Required |
| Groq API | Groq | Required |
| Semantic Scholar API | AI2 | Available |
| OpenAlex API | OpenAlex | Available |
| arXiv API | arXiv | Available |

### 7.2 Internal Dependencies
| Dependency | Description | Status |
|------------|-------------|--------|
| PDF Collection | Full-text PDFs for RAG | Pending |
| Human Coders | For verification sample | Self (researcher) |

---

## 8. Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| API rate limiting | Medium | Medium | Implement exponential backoff |
| Low PDF availability | Medium | High | Use multiple sources, OCR fallback |
| Inter-coder reliability below threshold | Low | High | Iterative schema refinement |
| LLM hallucination | Medium | High | Multi-model consensus, human verification |
| Context window limits | Low | Medium | Chunking strategy, retrieval optimization |

---

## 9. Acceptance Criteria

### 9.1 Phase 2 Acceptance
- [x] ≥ 1,500 unique papers identified (7,121 achieved)
- [x] Deduplication rate < 30% (9.8% achieved)
- [ ] Screening accuracy ≥ 90% vs human sample (pending human verification)
- [ ] ≥ 60% PDF retrieval rate (4.5% Open Access, need institutional access)

### 9.2 Phase 3 Acceptance
- [ ] RAG retrieval precision ≥ 0.80
- [ ] Cohen's κ ≥ 0.85 for categorical fields
- [ ] Weighted κ ≥ 0.80 for ordinal fields
- [ ] Krippendorff's α ≥ 0.80 for multi-select
- [ ] All discrepancies resolved with audit trail

### 9.3 Phase 4 Acceptance
- [ ] PRISMA flow diagram complete
- [ ] Ethics × HR matrix populated
- [ ] Manuscript draft complete
- [ ] All supplementary materials prepared

---

## 10. Glossary

| Term | Definition |
|------|------------|
| RAG | Retrieval-Augmented Generation |
| ICR | Inter-Coder Reliability |
| κ | Cohen's Kappa statistic |
| α | Krippendorff's Alpha |
| ICC | Intraclass Correlation Coefficient |
| PRISMA | Preferred Reporting Items for Systematic Reviews and Meta-Analyses |
| LLM | Large Language Model |

---

## 11. References

- See `docs/RESEARCH_PLAN.md` for detailed research methodology
- See `docs/SPEC.md` for technical specifications
- See `codebook/AI_Ethics_HR_Codebook.md` for coding manual

---

*Document Control*
*Created: 2026-02-02*
*Last Updated: 2026-02-03*
*Next Review: After full-text PDF collection*

---

## Appendix A: PRISMA Flow Summary (Phase 2)

```
IDENTIFICATION
├── Database Search (2026-02-02)
│   ├── Scopus:           4,851
│   ├── Web of Science:   1,307
│   ├── PubMed:             437
│   ├── ERIC:               302
│   ├── Semantic Scholar:   500
│   └── OpenAlex:           500
│   └── TOTAL:            7,897
│
SCREENING
├── Deduplication
│   ├── DOI duplicates:     714
│   ├── Title duplicates:    62
│   └── UNIQUE:           7,121
│
├── AI-Assisted Screening (Groq llama-3.3-70b)
│   ├── Included:         1,118 (15.7%)
│   ├── Excluded:         5,836 (82.0%)
│   └── Uncertain:          167 (2.3%)
│
└── Human Verification Sample: 1,994 (28.0% of total)

ELIGIBILITY (Pending)
├── Full-text PDF retrieval
│   ├── Open Access available:   50 (4.5%)
│   └── Institutional required: 1,068 (95.5%)
│
└── Full-text assessment: TBD

INCLUDED (Pending)
└── Studies in final review: TBD
```
