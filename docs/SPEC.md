# Technical Specification (SPEC)
## AI Ethics in Human Resource: RAG-Enabled Systematic Review Pipeline

**Version**: 1.0.0
**Date**: 2026-02-02
**Author**: Hosung You

---

## 1. System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    RAG-ENABLED SYSTEMATIC REVIEW PIPELINE                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │   SEARCH    │ →  │  DEDUP      │ →  │  SCREEN     │ →  │  RAG BUILD  │  │
│  │ 01_search.py│    │02_dedup.py  │    │03_screen.py │    │04_build_rag │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│         │                                                        │          │
│         ↓                                                        ↓          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │               6-PHASE VALIDATED CODING PIPELINE                      │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌────┐ │   │
│  │  │ Phase 1 │→│ Phase 2 │→│ Phase 3 │→│ Phase 4 │→│ Phase 5 │→│ P6 │ │   │
│  │  │ Initial │ │Consensus│ │ Human   │ │Reliabil.│ │Resolut. │ │ QA │ │   │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘ └────┘ │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│         │                                                                   │
│         ↓                                                                   │
│  ┌─────────────┐    ┌─────────────┐                                        │
│  │ SENSITIVITY │    │  ANALYSIS   │ →  Final Dataset + Manuscript          │
│  │07_sensitiv. │    │ 06_analyze  │                                        │
│  └─────────────┘    └─────────────┘                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Component Specifications

### 2.1 Search Module (`scripts/01_search.py`)

#### 2.1.1 Classes

| Class | Purpose | Dependencies |
|-------|---------|--------------|
| `SemanticScholarSearcher` | S2 API integration | `requests`, `tenacity` |
| `OpenAlexSearcher` | OpenAlex API integration | `requests` |
| `ArxivSearcher` | arXiv API integration | `requests`, `xml.etree` |
| `SearchPipeline` | Orchestrates all searchers | Above classes |

#### 2.1.2 Data Structures

```python
@dataclass
class Paper:
    paper_id: str
    title: str
    authors: List[str]
    year: int
    abstract: str
    doi: Optional[str]
    arxiv_id: Optional[str]
    pdf_url: Optional[str]
    source: str
    venue: Optional[str]
    citation_count: int
```

#### 2.1.3 API Specifications

| API | Rate Limit | Retry Strategy | Fields Retrieved |
|-----|------------|----------------|------------------|
| Semantic Scholar | 100/5min | Exponential backoff | title, authors, year, abstract, doi, openAccessPdf |
| OpenAlex | Polite pool | 3 retries | title, authors, year, abstract, doi, open_access |
| arXiv | 3s delay | Simple retry | title, authors, year, abstract, arxiv_id |

### 2.2 Deduplication Module (`scripts/02_deduplicate.py`)

#### 2.2.1 Matching Algorithms

| Method | Threshold | Priority |
|--------|-----------|----------|
| DOI exact match | 100% | 1 (highest) |
| arXiv ID match | 100% | 2 |
| Title fuzzy match | ≥ 85% | 3 |

#### 2.2.2 Source Priority

```python
SOURCE_PRIORITY = {
    "semantic_scholar": 3,  # Highest (best metadata)
    "openalex": 2,
    "arxiv": 1             # Lowest
}
```

### 2.3 Screening Module (`scripts/03_screen.py`)

#### 2.3.1 LLM Configuration

| Parameter | Value |
|-----------|-------|
| Provider | Groq |
| Model | llama-3.3-70b-versatile |
| Temperature | 0.1 |
| Max Tokens | 500 |

#### 2.3.2 Screening Prompt Template

```
You are screening papers for a systematic review on AI ethics in HR.

Paper Title: {title}
Abstract: {abstract}

Inclusion criteria:
1. Focus on AI/ML applications in HR contexts
2. Substantive discussion of ethical implications
3. Published 2015-2025, peer-reviewed
4. English language

Respond with JSON:
{
  "include": true/false,
  "confidence": 0.0-1.0,
  "reason": "brief explanation"
}
```

### 2.4 RAG Module (`scripts/04_build_rag.py`)

#### 2.4.1 Processing Pipeline

| Component | Tool/Library | Configuration |
|-----------|--------------|---------------|
| PDF Extraction | PyMuPDF | Pages 1-50, skip images |
| Text Chunking | RecursiveCharacterTextSplitter | chunk_size=1000, overlap=200 |
| Embedding | sentence-transformers | all-MiniLM-L6-v2 |
| Vector Store | ChromaDB | distance_metric=cosine |

#### 2.4.2 Chunking Strategy

```python
CHUNK_CONFIG = {
    "chunk_size": 1000,        # tokens
    "chunk_overlap": 200,       # tokens
    "separators": ["\n\n", "\n", ". ", " "],
    "length_function": len
}
```

#### 2.4.3 Embedding Specifications

| Parameter | Value |
|-----------|-------|
| Model | all-MiniLM-L6-v2 |
| Dimensions | 384 |
| Max Sequence | 256 tokens |
| Normalize | True |

### 2.5 6-Phase Coding Module (`scripts/05_code/`)

#### 2.5.1 Phase 1: Initial AI Coding

| Component | Specification |
|-----------|---------------|
| Model | claude-3-5-sonnet-20241022 |
| Temperature | 0.1 |
| Max Tokens | 4000 |
| Retrieval k | 8 chunks |
| Confidence Threshold | 0.75 |

#### 2.5.2 Phase 2: Multi-Model Consensus

| Model | Role | Cost/Paper |
|-------|------|------------|
| Claude 3.5 Sonnet | Primary extractor | $0.045 |
| GPT-4o | Verification | $0.035 |
| Groq Llama-3.3-70b | Efficiency check | $0.008 |

```python
CONSENSUS_RULES = {
    "threshold": "2_of_3",      # Majority agreement
    "tie_breaker": "model_a",   # Claude wins ties
    "critical_fields_require_3_of_3": [
        "fairness_bias.mentioned",
        "transparency.mentioned",
        "hr_function_primary"
    ]
}
```

#### 2.5.3 Phase 3: Human Sampling

```python
SAMPLING_CONFIG = {
    "sample_rate": 0.20,        # 20% of papers
    "minimum_papers": 30,
    "stratification": {
        "ethics_principle": "minimum_1_each",
        "hr_function": "proportional",
        "confidence_quartile": "oversample_low"
    }
}
```

#### 2.5.4 Phase 4: Reliability Metrics

| Metric | Formula | Target | Acceptable |
|--------|---------|--------|------------|
| Cohen's κ | (Po - Pe) / (1 - Pe) | ≥ 0.85 | ≥ 0.80 |
| Weighted κ | Σwij(pij - pij^e) / ... | ≥ 0.80 | ≥ 0.75 |
| Krippendorff's α | 1 - Do/De | ≥ 0.80 | ≥ 0.75 |
| ICC(2,1) | MSR - MSE / MSR + (k-1)MSE | ≥ 0.90 | ≥ 0.85 |

#### 2.5.5 Phase 5: Resolution Protocol

```python
RESOLUTION_FLOW = {
    "ai_ai_discordance": "human_verification",
    "ai_human_disagreement": "expert_arbitration",
    "human_human_disagreement": "consensus_meeting"
}
```

#### 2.5.6 Phase 6: Quality Gates

| Gate | Condition | Action if Failed |
|------|-----------|------------------|
| ICR Gate | κ ≥ 0.85 | Revise schema |
| Accuracy Gate | ≥ 90% | Increase human review |
| Bias Gate | < 5% directional | Field calibration |
| Completeness Gate | 100% coverage | Fill missing |
| Hallucination Gate | < 2% | Multi-model verify |

### 2.6 Sensitivity Module (`scripts/07_sensitivity.py`)

#### 2.6.1 Test Configurations

| Test Type | Parameters |
|-----------|------------|
| Model Comparison | Claude, GPT-4o, GPT-4o-mini, Groq |
| Chunk Size | 500, 750, 1000, 1500, 2000 |
| Retrieval k | 3, 5, 8, 10, 15 |
| Temperature | 0.0, 0.1, 0.3, 0.5 |
| Prompt Style | baseline, cot, few_shot, self_consistency |

---

## 3. Data Schemas

### 3.1 Paper Metadata Schema

```yaml
study_metadata:
  study_id: string (required)
  authors: string (required)
  year: integer (required, 2015-2025)
  title: string (required)
  journal: string
  doi: string (recommended)
  country: string
  methodology: enum
    - empirical_quantitative
    - empirical_qualitative
    - conceptual
    - review
    - case_study
    - mixed_methods
```

### 3.2 Coding Schema

```yaml
ethical_issues:
  fairness_bias:
    mentioned: boolean (required)
    type: array[enum]
      - algorithmic_bias
      - disparate_impact
      - protected_characteristics
      - historical_bias
      - proxy_discrimination
    severity: enum
      - major_focus
      - discussed
      - mentioned
      - not_addressed

  transparency:
    mentioned: boolean (required)
    type: array[enum]
      - explainability
      - black_box
      - interpretability
      - communication_to_employees
    severity: enum
      - major_focus
      - discussed
      - mentioned
      - not_addressed

  # ... (similar for accountability, privacy, autonomy, wellbeing)
```

### 3.3 Audit Log Schema

```python
@dataclass
class AuditEntry:
    timestamp: str          # ISO 8601
    phase: str              # "phase1", "phase2", etc.
    action: str             # "extraction", "consensus", "resolution"
    paper_id: str
    model: Optional[str]
    field: Optional[str]
    old_value: Optional[Any]
    new_value: Optional[Any]
    confidence: Optional[float]
    rationale: Optional[str]
    user: str               # "ai" or coder name
```

---

## 4. File Structure

```
AI-Ethics-HR-Review/
├── configs/
│   ├── pipeline_config.yaml       # Main configuration
│   └── phase_configs/
│       ├── phase1_config.yaml     # Claude extraction settings
│       ├── phase2_models.yaml     # Multi-model consensus
│       ├── phase3_sampling.yaml   # Human sampling strategy
│       ├── phase4_thresholds.yaml # ICR metric thresholds
│       ├── phase5_resolution.yaml # Discrepancy protocol
│       └── phase6_gates.yaml      # Quality assurance gates
├── scripts/
│   ├── 01_search.py               # Database search
│   ├── 02_deduplicate.py          # Deduplication
│   ├── 03_screen.py               # AI screening
│   ├── 04_build_rag.py            # RAG construction
│   ├── 05_code/
│   │   ├── __init__.py
│   │   ├── phase1_initial.py      # Initial AI coding
│   │   ├── phase2_consensus.py    # Multi-model consensus
│   │   ├── phase3_sampling.py     # Human sampling
│   │   ├── phase4_reliability.py  # ICR calculation
│   │   ├── phase5_resolution.py   # Discrepancy resolution
│   │   └── phase6_qa.py           # Quality assurance
│   ├── 07_sensitivity.py          # Sensitivity analysis
│   └── utils/
│       ├── metrics.py             # Statistical functions
│       ├── confidence.py          # Confidence calibration
│       └── audit.py               # Audit logging
├── data/
│   ├── 01_search_results/         # Raw API outputs
│   ├── 02_deduplicated/           # After deduplication
│   ├── 03_screened/               # After screening
│   ├── 04_full_text/              # PDFs
│   ├── 05_coded/
│   │   ├── phase1_raw/
│   │   ├── phase2_consensus/
│   │   ├── phase3_human/
│   │   ├── phase4_reliability/
│   │   ├── phase5_resolutions/
│   │   └── phase6_final/
│   └── 06_analysis/
├── rag/
│   ├── chroma_db/                 # Vector database
│   ├── embeddings/                # Cached embeddings
│   └── configs/                   # RAG settings
├── codebook/
│   ├── AI_Ethics_HR_Codebook.md   # Coding manual
│   ├── coding_schema.yaml         # Field definitions
│   └── coder_training/
│       └── training_protocol.md
├── validation/
│   ├── sensitivity_results/
│   ├── reliability_reports/
│   └── bias_analysis/
├── manuscript/
│   ├── main_manuscript_APA7.md
│   └── supplementary/
└── docs/
    ├── RESEARCH_PLAN.md           # Original plan
    ├── PRD.md                     # Requirements
    └── SPEC.md                    # This document
```

---

## 5. API Interfaces

### 5.1 Search Pipeline Interface

```python
class SearchPipeline:
    def __init__(self, config_path: str)
    def search_all(self, query: str, max_results: int) -> List[Paper]
    def save_results(self, papers: List[Paper], output_dir: str)
```

### 5.2 Coding Pipeline Interface

```python
class Phase1Coder:
    def __init__(self, rag_db: ChromaDB, model: str)
    def code_paper(self, paper_id: str) -> Phase1CodingResult

class Phase2ConsensusBuilder:
    def __init__(self, models: List[str])
    def build_consensus(self, phase1_results: List) -> ConsensusResult

class Phase4ReliabilityCalculator:
    def calculate_icr(self, ai_codes: List, human_codes: List) -> ICRMetrics
```

### 5.3 Output Formats

| Stage | Format | Location |
|-------|--------|----------|
| Search | JSON | `data/01_search_results/` |
| Dedup | JSON | `data/02_deduplicated/` |
| Screen | JSON | `data/03_screened/` |
| Phase 1-5 | JSON | `data/05_coded/phaseN_*/` |
| Phase 6 | CSV + JSON | `data/05_coded/phase6_final/` |
| Analysis | CSV + PNG | `data/06_analysis/` |

---

## 6. Error Handling

### 6.1 API Errors

| Error Type | Handling Strategy |
|------------|-------------------|
| Rate Limit (429) | Exponential backoff, max 5 retries |
| Timeout | Retry with increased timeout |
| Auth Error (401) | Fail with clear message |
| Server Error (500) | Retry 3 times, then skip |

### 6.2 Processing Errors

| Error Type | Handling Strategy |
|------------|-------------------|
| PDF Parse Failure | Log, skip to next paper |
| Empty Extraction | Flag for human review |
| Schema Validation | Reject, require re-extraction |
| Confidence < Threshold | Route to Phase 5 |

---

## 7. Performance Requirements

| Metric | Requirement |
|--------|-------------|
| Search throughput | ≥ 100 papers/minute |
| Screening throughput | ≥ 50 papers/minute |
| RAG indexing | ≤ 2 minutes/paper |
| Phase 1 coding | ≤ 30 seconds/paper |
| Phase 2 consensus | ≤ 45 seconds/paper |
| Total pipeline | ≤ 20 hours for 200 papers |

---

## 8. Security & Privacy

### 8.1 API Key Management
- Store in `.env` file (not committed)
- Use environment variables
- Never log API keys

### 8.2 Data Protection
- Full-text PDFs not committed to git
- Personal data excluded from coding
- Audit logs stored securely

---

## 9. Testing Strategy

### 9.1 Unit Tests
- Each utility function
- Schema validation
- Metric calculations

### 9.2 Integration Tests
- End-to-end pipeline on sample data
- API connection verification
- RAG retrieval quality

### 9.3 Validation Tests
- Human-AI agreement on gold standard
- Cross-model consistency
- Reproducibility check

---

## 10. Deployment

### 10.1 Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with actual keys
```

### 10.2 Execution Order

```bash
# Stage 1: Search
python scripts/01_search.py

# Stage 2: Deduplicate
python scripts/02_deduplicate.py

# Stage 3: Screen
python scripts/03_screen.py

# Stage 4: Download PDFs (manual)
# Place PDFs in data/04_full_text/

# Stage 5: Build RAG
python scripts/04_build_rag.py

# Stage 6: 6-Phase Coding
python scripts/05_code/phase1_initial.py
python scripts/05_code/phase2_consensus.py
# ... continue through phases

# Stage 7: Sensitivity Analysis
python scripts/07_sensitivity.py
```

---

## 11. Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-02-02 | Initial specification |

---

*Document Control*
*Created: 2026-02-02*
*Last Updated: 2026-02-02*
