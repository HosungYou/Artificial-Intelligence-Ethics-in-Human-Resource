# AI Ethics in Human Resource: A RAG-Enabled Systematic Literature Review

## Overview

**연구 제목**: Artificial Intelligence Ethics in Human Resource: A RAG-Enabled Systematic Literature Review

**연구 목표** (Hybrid Approach):
1. **체계적 문헌고찰**: HR 전 영역에서 AI 윤리 이슈 종합 분석
2. **방법론 기여**: RAG 기반 체계적 문헌고찰 방법론 제안 및 검증

**타겟 저널**: HRD 분야 저널
- Human Resource Development Review (HRDR) - 리뷰 논문 특화
- Human Resource Development Quarterly (HRDQ)
- Human Resource Development International (HRDI)

**연구 범위**: HR 전 영역 포괄적 검토
- AI in Recruitment & Selection
- AI in Performance Management
- AI in Learning & Development
- AI in Employee Analytics & Surveillance

---

## 1. Research Questions

### Primary RQs (체계적 문헌고찰)
- **RQ1**: What are the key ethical issues associated with AI applications across HR functions?
- **RQ2**: How do AI ethics concerns vary across different HR domains (recruitment, performance, L&D)?
- **RQ3**: What theoretical frameworks have been applied to understand AI ethics in HR contexts?
- **RQ4**: What gaps exist in the current literature on AI ethics in HR?

### Secondary RQs (방법론 기여)
- **RQ5**: How can RAG-enabled systematic review methodology improve literature synthesis efficiency and quality?
- **RQ6**: What are the accuracy and reliability of AI-assisted literature coding compared to human coding?

---

## 2. Conceptual Framework

### AI Ethics Dimensions in HR (Based on Floridi et al., 2018; Jobin et al., 2019)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    AI ETHICS IN HR FRAMEWORK                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ETHICAL PRINCIPLES                    HR FUNCTIONS                     │
│  ┌─────────────────────┐              ┌─────────────────────┐          │
│  │ • Fairness/Bias     │              │ • Recruitment       │          │
│  │ • Transparency      │      ×       │ • Selection         │          │
│  │ • Accountability    │              │ • Performance Mgmt  │          │
│  │ • Privacy           │              │ • L&D               │          │
│  │ • Human Autonomy    │              │ • Analytics         │          │
│  │ • Beneficence       │              │ • Employee Relations│          │
│  └─────────────────────┘              └─────────────────────┘          │
│                    ↓                            ↓                       │
│            OUTCOMES: Trust, Acceptance, Resistance, Regulation         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Ethical Principles to Analyze
1. **Fairness & Bias**: Algorithmic discrimination, disparate impact
2. **Transparency & Explainability**: Black-box algorithms, XAI
3. **Accountability**: Human oversight, liability
4. **Privacy & Data Protection**: Surveillance, consent, GDPR
5. **Human Autonomy**: Human-in-the-loop, deskilling
6. **Beneficence/Non-maleficence**: Employee wellbeing

### HR Functions to Cover
1. **Recruitment & Selection**: Resume screening, video interviews, chatbots
2. **Performance Management**: Continuous monitoring, predictive analytics
3. **Learning & Development**: Personalized learning, skill gap analysis
4. **People Analytics**: Workforce planning, attrition prediction
5. **Employee Relations**: Sentiment analysis, engagement monitoring

---

## 3. RAG-Enabled Methodology Design

### 3.1 Architecture Overview (5-Stage Pipeline)

```
┌─────────────────────────────────────────────────────────────────────────┐
│              RAG-ENABLED SYSTEMATIC REVIEW PIPELINE                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Stage 1: IDENTIFICATION                                                │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │  Database Search (Semantic Scholar, OpenAlex, Scopus)          │    │
│  │  → Boolean Query + AI-assisted expansion                       │    │
│  │  → Target: 1,500-2,000 initial records                         │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                              ↓                                          │
│  Stage 2: SCREENING (AI-Assisted with Human Verification)              │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │  Title/Abstract Screening: LLM + Inclusion/Exclusion Criteria  │    │
│  │  → Groq (llama-3.3-70b) for cost-effective screening           │    │
│  │  → Human verification on 20% sample                            │    │
│  │  → Target: 200-400 full-text eligible                          │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                              ↓                                          │
│  Stage 3: RAG INDEX BUILDING                                           │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │  PDF Processing: PyMuPDF direct text extraction                │    │
│  │  Chunking: RecursiveCharacterTextSplitter (1000 tokens)        │    │
│  │  Embedding: all-MiniLM-L6-v2 (local) or OpenAI ada-002         │    │
│  │  Vector Store: ChromaDB per-paper + unified collection         │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                              ↓                                          │
│  Stage 4: 6-PHASE VALIDATED CODING (NEW - See Section 3.3)            │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │  Phase 1: Initial AI Coding with RAG                           │    │
│  │  Phase 2: Multi-Model Consensus (Claude + GPT-4o + Groq)       │    │
│  │  Phase 3: Human Verification Sampling (20% stratified)         │    │
│  │  Phase 4: Inter-Coder Reliability (κ > 0.85, α > 0.80)         │    │
│  │  Phase 5: Discrepancy Resolution Protocol                      │    │
│  │  Phase 6: Final Quality Assurance                              │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                              ↓                                          │
│  Stage 5: SYNTHESIS & ANALYSIS                                         │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │  Thematic Analysis: Ethics principle × HR function mapping     │    │
│  │  Gap Analysis: Under-researched areas identification           │    │
│  │  Trend Analysis: Publication trends over time                  │    │
│  │  Research Agenda: Future directions                            │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.3 6-Phase Validated Coding Pipeline (NEW)

**목표**: 코딩 정확도 95% 이상, Cohen's κ > 0.85 달성

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              6-PHASE MULTI-LAYER VALIDATION PIPELINE                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  PHASE 1: INITIAL AI CODING (RAG-Enabled)                                   │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ • Primary Model: Claude 3.5 Sonnet via RAG pipeline                    │ │
│  │ • Per-field confidence scores generated                                │ │
│  │ • Ethics × HR Function matrix populated                                │ │
│  │ • Output: raw_coding_phase1.json                                       │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                              ↓                                               │
│  PHASE 2: AI CONSENSUS CODING (Multi-Model)                                  │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ • Model A: Claude 3.5 Sonnet (primary)                                 │ │
│  │ • Model B: GPT-4o (verification)                                       │ │
│  │ • Model C: Groq/Llama-3.3-70b (efficiency check)                       │ │
│  │ • Concordance threshold: 2/3 agreement → 97% accuracy                  │ │
│  │ • Discordance → Phase 5 queue                                          │ │
│  │ • Output: consensus_coding_phase2.json                                 │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                              ↓                                               │
│  PHASE 3: HUMAN VERIFICATION SAMPLING                                        │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ • Stratified sampling: 20% of corpus (minimum 30 papers)               │ │
│  │ • Stratification dimensions:                                           │ │
│  │   - Ethics principle coverage (1 paper per principle minimum)          │ │
│  │   - HR function distribution (proportional)                            │ │
│  │   - Confidence quartiles (oversample low confidence)                   │ │
│  │ • Dual coding by trained research assistants                           │ │
│  │ • Output: human_gold_standard_phase3.xlsx                              │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                              ↓                                               │
│  PHASE 4: INTER-CODER RELIABILITY CALCULATION                                │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ • Cohen's Kappa: Binary/nominal categorical fields (target > 0.85)     │ │
│  │ • Weighted Kappa: Ordinal fields like severity (target > 0.80)         │ │
│  │ • Krippendorff's Alpha: Multi-coder, mixed data types (target > 0.80)  │ │
│  │ • ICC(2,1): Continuous fields like confidence (target > 0.90)          │ │
│  │ • Field-level accuracy matrix generated                                │ │
│  │ • Output: reliability_metrics_phase4.json                              │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                              ↓                                               │
│  PHASE 5: DISCREPANCY RESOLUTION PROTOCOL                                    │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ • AI-Human discrepancies → Expert arbitration                          │ │
│  │ • AI-AI discordance → Human verification                               │ │
│  │ • Human-Human disagreements → Consensus meeting                        │ │
│  │ • Decision audit trail maintained                                      │ │
│  │ • Output: resolved_discrepancies_phase5.json                           │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                              ↓                                               │
│  PHASE 6: FINAL QUALITY ASSURANCE                                            │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ • Quality gates verification (all thresholds met?)                     │ │
│  │ • Systematic bias detection (ethics × HR function matrix)              │ │
│  │ • Completeness check (100% papers coded, no missing critical fields)   │ │
│  │ • Final confidence calibration via isotonic regression                 │ │
│  │ • Output: final_coded_dataset_phase6.csv                               │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Phase 1 Configuration

```yaml
phase_1_config:
  model: "claude-3-5-sonnet-20241022"
  temperature: 0.1  # Low for consistency
  max_tokens: 4000

  rag_settings:
    embedding_model: "all-MiniLM-L6-v2"
    chunk_size: 1000
    chunk_overlap: 200
    retrieval_k: 8  # Top-8 chunks per query

  confidence_thresholds:
    ethics_mention_boolean: 0.80
    ethics_severity_ordinal: 0.75
    hr_function_categorical: 0.85
    theory_name_freetext: 0.70
```

#### Phase 2 Multi-Model Configuration

```yaml
phase_2_models:
  model_a:
    name: "Claude 3.5 Sonnet"
    provider: "anthropic"
    role: "primary_extractor"
    cost_per_paper: $0.045

  model_b:
    name: "GPT-4o"
    provider: "openai"
    role: "verification_extractor"
    cost_per_paper: $0.035

  model_c:
    name: "Llama 3.3 70B"
    provider: "groq"
    role: "efficiency_checker"
    cost_per_paper: $0.008

consensus_rules:
  threshold: "2_of_3"  # Majority agreement
  tie_breaker: "model_a"
  critical_fields_require_3_of_3:
    - fairness_bias.mentioned
    - transparency.mentioned
    - hr_function_primary
```

#### Quality Gates (Phase 6)

```yaml
quality_gates:
  icr_thresholds:
    ethics_mention_kappa: 0.85
    ethics_severity_kappa: 0.80
    hr_function_kappa: 0.85
    krippendorff_alpha_min: 0.80
    icc_continuous: 0.90

  ai_human_thresholds:
    overall_accuracy: 0.90
    critical_field_accuracy: 0.95
    per_field_minimum: 0.80

  error_rate_thresholds:
    hallucination_rate: 0.02    # Max 2%
    extraction_miss_rate: 0.10  # Max 10%
    systematic_bias: 0.05       # Max 5% directional bias
```

#### Success Criteria

| Criterion | Target | Measurement |
|-----------|--------|-------------|
| Ethics field accuracy | > 95% | vs. gold standard |
| Cohen's Kappa (categorical) | > 0.85 | Human-human, AI-human |
| Weighted Kappa (ordinal) | > 0.80 | Severity fields |
| Krippendorff's Alpha | > 0.80 | Multi-select fields |
| ICC (continuous) | > 0.90 | Confidence scores |
| Hallucination rate | < 2% | AI-generated non-existent values |
| Systematic bias | < 5% | Per-field directional bias |
| Human review rate | < 15% | Papers requiring manual intervention |

### 3.4 Sensitivity Analysis

검증 엄밀성을 높이기 위한 민감도 분석:

```yaml
sensitivity_analysis:

  # 1. LLM Model Comparison
  model_comparison:
    models_tested:
      - claude_3_5_sonnet
      - gpt_4o
      - gpt_4o_mini
      - groq_llama_3_3
    metrics: [accuracy, latency, cost, hallucination_rate]

  # 2. RAG Configuration
  rag_sensitivity:
    chunk_sizes: [500, 750, 1000, 1500, 2000]
    retrieval_k: [3, 5, 8, 10, 15]
    metrics: [retrieval_precision, extraction_accuracy]

  # 3. Temperature Settings
  temperature_sensitivity:
    temperatures: [0.0, 0.1, 0.3, 0.5]
    metrics: [consistency, edge_case_detection]

  # 4. Prompt Variations
  prompt_sensitivity:
    variations: [baseline, chain_of_thought, few_shot, self_consistency]
    metrics: [accuracy_delta, cost_multiplier]
```

### 3.2 Coding Schema for AI Ethics in HR

```yaml
# AI-Ethics-HR Coding Schema v1.0

study_metadata:
  study_id: ""
  authors: ""
  year: 2024
  title: ""
  journal: ""
  doi: ""
  country: ""
  methodology: ["empirical_quantitative", "empirical_qualitative", "conceptual", "review", "case_study", "mixed_methods"]

hr_function:
  primary: ["recruitment", "selection", "performance_management", "learning_development", "people_analytics", "employee_relations", "workforce_planning", "multiple"]
  secondary: []  # If paper covers multiple functions

ai_technology:
  type: ["machine_learning", "nlp", "computer_vision", "chatbot", "predictive_analytics", "recommender_system", "robotic_process_automation", "generative_ai", "multiple", "not_specified"]
  specific_tool: ""  # e.g., "HireVue", "Pymetrics", "ChatGPT"

ethical_issues:
  fairness_bias:
    mentioned: [true, false]
    type: ["algorithmic_bias", "disparate_impact", "protected_characteristics", "historical_bias", "proxy_discrimination"]
    severity: ["major_focus", "discussed", "mentioned", "not_addressed"]

  transparency:
    mentioned: [true, false]
    type: ["explainability", "black_box", "interpretability", "communication_to_employees"]
    severity: ["major_focus", "discussed", "mentioned", "not_addressed"]

  accountability:
    mentioned: [true, false]
    type: ["human_oversight", "liability", "responsibility", "auditability"]
    severity: ["major_focus", "discussed", "mentioned", "not_addressed"]

  privacy:
    mentioned: [true, false]
    type: ["data_collection", "surveillance", "consent", "gdpr", "data_minimization"]
    severity: ["major_focus", "discussed", "mentioned", "not_addressed"]

  autonomy:
    mentioned: [true, false]
    type: ["human_in_the_loop", "deskilling", "agency", "decision_authority"]
    severity: ["major_focus", "discussed", "mentioned", "not_addressed"]

  wellbeing:
    mentioned: [true, false]
    type: ["job_quality", "psychological_safety", "work_intensification", "employee_experience"]
    severity: ["major_focus", "discussed", "mentioned", "not_addressed"]

theoretical_framework:
  applied: [true, false]
  theory_name: ""  # e.g., "Algorithmic Management Theory", "Technology Acceptance Model"
  framework_category: ["ethics_frameworks", "hr_theories", "technology_theories", "organizational_theories", "none"]

key_findings:
  summary: ""  # 2-3 sentence summary
  recommendations: ""  # Practical recommendations
  limitations_noted: [true, false]

research_gaps:
  identified: [true, false]
  gap_description: ""

quality_indicators:
  peer_reviewed: [true, false]
  empirical_evidence: [true, false]
  sample_size: ""  # For empirical studies
  geographic_scope: ["single_country", "multi_country", "global", "not_specified"]
```

---

## 4. Search Strategy

### 4.1 Database Selection
- **Primary**: Scopus, Web of Science, PsycINFO
- **Secondary**: Semantic Scholar (API), OpenAlex (open access)
- **Grey Literature**: SSRN, ProQuest Dissertations

### 4.2 Search Terms

```
# Main Query
(
  ("artificial intelligence" OR "AI" OR "machine learning" OR "algorithm*"
   OR "automated" OR "chatbot" OR "NLP" OR "predictive analytics")
  AND
  ("human resource*" OR "HR" OR "HRM" OR "talent management"
   OR "recruitment" OR "selection" OR "hiring" OR "performance management"
   OR "learning and development" OR "training" OR "workforce analytics"
   OR "people analytics" OR "employee*")
  AND
  ("ethic*" OR "bias" OR "fairness" OR "discrimination" OR "transparency"
   OR "accountability" OR "privacy" OR "surveillance" OR "trust"
   OR "responsible AI" OR "algorithmic")
)

# Filters
- Date: 2015-2025 (AI proliferation period)
- Language: English
- Document Type: Journal articles, Conference papers, Book chapters
```

### 4.3 Inclusion/Exclusion Criteria

**Inclusion**:
- Focus on AI/ML applications in HR contexts
- Addresses ethical implications or concerns
- Published in peer-reviewed outlets (2015-2025)
- English language

**Exclusion**:
- Pure technical papers without ethical discussion
- HR software reviews without ethical analysis
- Opinion pieces without scholarly rigor
- Duplicate publications

---

## 5. Implementation Plan

### Project Stage 1: Setup & Search (Week 1-2)
- [ ] Create new project directory: `AI-Ethics-HR-Review/`
- [ ] Configure RAG pipeline for HR domain
- [ ] Execute database searches (Semantic Scholar, OpenAlex, Scopus)
- [ ] Deduplicate results
- [ ] Deliverable: PRISMA identification count

### Project Stage 2: Screening (Week 2-3)
- [ ] AI-assisted title/abstract screening (Groq)
- [ ] Human verification on 20% sample
- [ ] Calculate screening reliability (kappa)
- [ ] Full-text retrieval
- [ ] Deliverable: PRISMA screening flowchart

### Project Stage 3: RAG Building (Week 3-4)
- [ ] PDF text extraction (PyMuPDF)
- [ ] Chunk and embed documents (1000 tokens, all-MiniLM-L6-v2)
- [ ] Build ChromaDB index
- [ ] Validate retrieval quality
- [ ] Deliverable: RAG index ready

### Project Stage 4: 6-Phase Validated Coding (Week 4-8) ⭐ NEW

#### Coding Phase 1: Initial AI Coding (Week 4)
- [ ] Configure Claude 3.5 Sonnet extraction prompts
- [ ] Run RAG-assisted coding on all papers
- [ ] Generate per-field confidence scores
- [ ] Output: `raw_coding_phase1.json`
- [ ] Quality Check: Flag papers with confidence < 0.75

#### Coding Phase 2: Multi-Model Consensus (Week 5)
- [ ] Run GPT-4o verification extraction
- [ ] Run Groq Llama efficiency check
- [ ] Calculate 3-model concordance
- [ ] Route discordant cases to Phase 5 queue
- [ ] Output: `consensus_coding_phase2.json`
- [ ] Target: 2/3 agreement → 97% accuracy

#### Coding Phase 3: Human Verification Sampling (Week 5-6)
- [ ] Select stratified 20% sample (min 30 papers)
- [ ] **Single Researcher Protocol** (adapted for 1 human coder):
  - AI consensus = primary coding
  - Human = gold standard arbiter
  - Intra-rater reliability: Re-code 20% after 2 weeks
- [ ] Output: `human_gold_standard_phase3.xlsx`

#### Coding Phase 4: Inter-Coder Reliability (Week 6)
- [ ] **Single Researcher Adaptation**:
  - AI-Human κ: AI consensus vs. Human gold standard (Target > 0.85)
  - Intra-rater κ: Human T1 vs. Human T2 (2-week gap, Target > 0.90)
- [ ] Calculate Weighted Kappa (ordinal): Target > 0.80
- [ ] Calculate Krippendorff's Alpha (AI models + Human): Target > 0.80
- [ ] Generate Ethics × HR accuracy matrix
- [ ] Output: `reliability_metrics_phase4.json`
- [ ] **Note**: With 1 human, AI-Human agreement becomes primary metric

#### Coding Phase 5: Discrepancy Resolution (Week 7)
- [ ] Resolve AI-AI discordance → Human verification
- [ ] Resolve AI-Human disagreement → Expert arbitration
- [ ] Resolve Human-Human disagreement → Consensus meeting
- [ ] Document all decisions with rationale
- [ ] Output: `resolved_discrepancies_phase5.json`
- [ ] Audit trail: Full provenance tracking

#### Coding Phase 6: Final Quality Assurance (Week 7-8)
- [ ] Verify all quality gates met
- [ ] Detect systematic bias (< 5% threshold)
- [ ] Calibrate confidence scores (isotonic regression)
- [ ] Complete 100% paper coverage check
- [ ] Output: `final_coded_dataset_phase6.csv`
- [ ] Deliverable: Publication-ready coded dataset

### Project Stage 5: Analysis & Writing (Week 8-12)
- [ ] Thematic synthesis
- [ ] Ethics × HR function matrix analysis
- [ ] Gap analysis
- [ ] Sensitivity analysis reporting
- [ ] Draft manuscript
- [ ] Deliverable: Manuscript draft

---

## 6. Expected Contributions

### Theoretical Contributions
1. **Integrated Framework**: First comprehensive mapping of AI ethics issues across all HR functions
2. **Taxonomic Clarification**: Clear categorization of ethical concerns by HR domain
3. **Research Agenda**: Evidence-based identification of under-researched areas

### Methodological Contributions
1. **RAG-SR Method**: Demonstration of RAG-enabled systematic review methodology
2. **Coding Schema**: Reusable schema for AI ethics in HR research
3. **Validation Data**: Human-AI coding comparison for reliability assessment

### Practical Contributions
1. **Practitioner Guidelines**: Ethical AI implementation recommendations for HR
2. **Policy Implications**: Regulatory and compliance considerations
3. **Assessment Tool**: Checklist for ethical AI deployment in HR

---

## 7. Project Structure

```
AI-Ethics-HR-Review/
├── data/
│   ├── 01_search_results/       # Raw search exports
│   ├── 02_deduplicated/         # After deduplication
│   ├── 03_screened/             # After title/abstract screening
│   ├── 04_full_text/            # PDFs for included studies
│   ├── 05_coded/                # 6-Phase Coding Outputs ⭐
│   │   ├── phase1_raw/          # Initial AI coding (Claude)
│   │   │   └── {study_id}_phase1.json
│   │   ├── phase2_consensus/    # Multi-model consensus
│   │   │   └── {study_id}_consensus.json
│   │   ├── phase3_human/        # Human gold standard
│   │   │   ├── coder1_gold_standard.xlsx
│   │   │   ├── coder2_gold_standard.xlsx
│   │   │   └── merged_gold_standard.xlsx
│   │   ├── phase4_reliability/  # ICR metrics
│   │   │   ├── icr_report.json
│   │   │   ├── accuracy_matrix.csv
│   │   │   └── bias_analysis.json
│   │   ├── phase5_resolutions/  # Discrepancy audit trail
│   │   │   └── resolutions_log.json
│   │   └── phase6_final/        # Final validated dataset
│   │       ├── final_coded_dataset.csv
│   │       ├── quality_report.md
│   │       └── calibration_model.pkl
│   └── 06_analysis/             # Analysis outputs
├── rag/
│   ├── chroma_db/               # Vector database
│   ├── embeddings/              # Cached embeddings
│   └── configs/                 # RAG configuration
├── scripts/
│   ├── 01_search.py             # Database API calls
│   ├── 02_deduplicate.py        # Duplicate removal
│   ├── 03_screen.py             # AI-assisted screening
│   ├── 04_build_rag.py          # RAG index construction
│   ├── 05_code/                 # 6-Phase Coding Module ⭐
│   │   ├── __init__.py
│   │   ├── phase1_initial.py    # Initial AI coding
│   │   ├── phase2_consensus.py  # Multi-model consensus
│   │   ├── phase3_sampling.py   # Human sampling strategy
│   │   ├── phase4_reliability.py # ICR calculations
│   │   ├── phase5_resolution.py # Discrepancy resolution
│   │   ├── phase6_qa.py         # Quality assurance
│   │   └── run_validated_pipeline.py
│   ├── 06_analyze.py            # Analysis scripts
│   ├── 07_sensitivity.py        # Sensitivity analysis ⭐
│   └── utils/                   # Helper functions
│       ├── confidence.py        # Confidence calculation
│       ├── metrics.py           # Kappa, ICC, Alpha
│       └── audit.py             # Audit trail logging
├── codebook/
│   ├── AI_Ethics_HR_Codebook.md # Full coding manual
│   ├── coding_schema.yaml       # Machine-readable schema
│   ├── coder_training/          # Training materials ⭐
│   │   ├── training_protocol.md
│   │   └── calibration_papers/
│   └── examples/                # Coding examples
├── validation/                  # Validation artifacts ⭐
│   ├── sensitivity_results/     # Sensitivity analysis outputs
│   ├── reliability_reports/     # ICR reports
│   └── bias_analysis/           # Systematic bias detection
├── manuscript/
│   ├── main_manuscript.md       # APA 7th format draft
│   ├── supplementary/           # Tables, figures
│   └── references.bib           # Bibliography
├── configs/
│   ├── pipeline_config.yaml     # Pipeline settings
│   ├── phase_configs/           # Per-phase configurations ⭐
│   │   ├── phase1_config.yaml
│   │   ├── phase2_models.yaml
│   │   ├── phase3_sampling.yaml
│   │   ├── phase4_thresholds.yaml
│   │   └── phase6_gates.yaml
│   └── sensitivity_config.yaml  # Sensitivity analysis settings
└── README.md
```

---

## 8. Key Files to Create

### Priority 1 (Week 1) - Setup
1. `configs/pipeline_config.yaml` - Pipeline configuration
2. `codebook/coding_schema.yaml` - Coding schema definition
3. `scripts/01_search.py` - Search execution script
4. `configs/phase_configs/` - All phase configuration files

### Priority 2 (Week 2-3) - Pre-Coding Pipeline
5. `scripts/02_deduplicate.py` - Deduplication script
6. `scripts/03_screen.py` - AI screening with Groq
7. `scripts/04_build_rag.py` - RAG index builder

### Priority 3 (Week 4-6) - 6-Phase Coding Module ⭐
8. `scripts/05_code/phase1_initial.py` - Initial AI coding with Claude
9. `scripts/05_code/phase2_consensus.py` - Multi-model consensus (Claude + GPT-4o + Groq)
10. `scripts/05_code/phase3_sampling.py` - Stratified human sampling
11. `scripts/05_code/phase4_reliability.py` - ICR calculations (Kappa, Alpha, ICC)
12. `scripts/05_code/phase5_resolution.py` - Discrepancy resolution protocol
13. `scripts/05_code/phase6_qa.py` - Quality assurance gates
14. `scripts/05_code/run_validated_pipeline.py` - Full pipeline orchestrator

### Priority 4 (Week 7-8) - Validation & Analysis
15. `scripts/07_sensitivity.py` - Sensitivity analysis
16. `scripts/utils/metrics.py` - Statistical metrics (Kappa, ICC, Alpha)
17. `scripts/utils/confidence.py` - Confidence calculation
18. `codebook/coder_training/training_protocol.md` - Human coder training guide

### Priority 5 (Week 9+) - Manuscript
19. `scripts/06_analyze.py` - Analysis and visualization
20. `manuscript/main_manuscript.md` - Manuscript draft
21. `validation/reliability_reports/` - Final reliability documentation

---

## 9. Verification Plan

### 9.1 Coding Quality Validation (6-Phase System)

| Phase | Metric | Target | Action if Failed |
|-------|--------|--------|-----------------|
| Phase 1 | Per-field confidence | > 0.75 | Route to Phase 2 consensus |
| Phase 2 | 3-model concordance | 2/3 agreement | Route to Phase 5 human review |
| Phase 3 | Human coder training | κ > 0.80 calibration | Recalibration session |
| Phase 4 | Cohen's Kappa (categorical) | > 0.85 | Revise schema definitions |
| Phase 4 | Weighted Kappa (ordinal) | > 0.80 | Simplify severity levels |
| Phase 4 | Krippendorff's Alpha | > 0.80 | Collapse multi-select options |
| Phase 4 | ICC (continuous) | > 0.90 | Review confidence calculation |
| Phase 5 | Discrepancy resolution | 100% resolved | Expert arbitration |
| Phase 6 | Hallucination rate | < 2% | Increase human review |
| Phase 6 | Systematic bias | < 5% per field | Field-specific calibration |
| Phase 6 | Completeness | 100% papers coded | Fill missing values |

### 9.2 Inter-Coder Reliability Metrics

**⚠️ Single Researcher Protocol**: 인간 연구자 1명 시나리오에 맞춤 설계

```
RELIABILITY METRICS (Single Human + Multi-AI)
═══════════════════════════════════════════════════════════════════════════
Comparison Type         | Metric          | Target | Acceptable | Purpose
═══════════════════════════════════════════════════════════════════════════
AI Consensus vs Human   | Cohen's κ       | > 0.85 | > 0.80     | Primary validation
Human T1 vs Human T2    | Intra-rater κ   | > 0.90 | > 0.85     | Self-consistency
Claude vs GPT-4o        | Cohen's κ       | > 0.90 | > 0.85     | Model agreement
3-Model Consensus       | Krippendorff α  | > 0.80 | > 0.75     | Multi-model reliability
═══════════════════════════════════════════════════════════════════════════

FIELD-LEVEL TARGETS (AI-Human Agreement)
═══════════════════════════════════════════════════════════════════
Field Category          | Metric          | Target | Acceptable
═══════════════════════════════════════════════════════════════════
Ethics mention (binary) | AI-Human κ      | > 0.85 | > 0.80
Ethics severity (ordinal)| Weighted κ     | > 0.80 | > 0.75
Ethics types (multi)    | Krippendorff α  | > 0.80 | > 0.75
HR function (nominal)   | AI-Human κ      | > 0.85 | > 0.80
AI technology (nominal) | AI-Human κ      | > 0.85 | > 0.80
Theory (binary)         | AI-Human κ      | > 0.80 | > 0.75
═══════════════════════════════════════════════════════════════════
```

**단일 연구자 신뢰도 확보 전략**:
1. **AI Consensus as Primary Coder**: 3개 LLM 합의를 1차 코딩으로 사용
2. **Human as Gold Standard**: 인간 연구자가 20% 샘플을 독립 코딩 → 정답 기준
3. **Intra-rater Check**: 2주 후 동일 샘플 재코딩으로 자기 일관성 검증
4. **Arbitration Role**: 불일치 시 인간이 최종 결정 (rationale 문서화)

### 9.3 Sensitivity Analysis Checklist

- [ ] **Model Comparison**: Claude vs GPT-4o vs Groq accuracy comparison
- [ ] **Chunk Size Impact**: Test 500, 750, 1000, 1500, 2000 tokens
- [ ] **Retrieval k Impact**: Test k = 3, 5, 8, 10, 15 chunks
- [ ] **Temperature Impact**: Test 0.0, 0.1, 0.3, 0.5
- [ ] **Prompt Variations**: Baseline vs CoT vs Few-shot vs Self-consistency

### 9.4 Documentation Audit Trail

- [ ] All LLM prompts versioned and stored
- [ ] Model versions documented (API snapshots)
- [ ] RAG configuration logged (YAML)
- [ ] Random seeds recorded (sampling reproducibility)
- [ ] Discrepancy rationales documented
- [ ] Schema revision history maintained

### 9.5 Methodology Validation

- [ ] Compare AI screening vs. human screening (kappa > 0.80)
- [ ] Compare AI coding vs. human coding (kappa > 0.85)
- [ ] Calculate time savings (target: 50%+ reduction)
- [ ] Document error types and rates by field
- [ ] Report AI-Human agreement per ethics principle

### 9.6 Review Quality (PRISMA 2020)

- [ ] Follow PRISMA 2020 guidelines
- [ ] Report all PRISMA items
- [ ] Pre-register protocol (OSF or PROSPERO)
- [ ] Make data and code available (GitHub)
- [ ] Include 6-Phase validation as methodological contribution

---

## 10. Resource Estimates (6-Phase Validation)

### Timeline (Total: 10-12 weeks)

| Stage | Duration | Dependencies |
|-------|----------|--------------|
| Stage 1: Setup & Search | 2 weeks | - |
| Stage 2: Screening | 1 week | Stage 1 complete |
| Stage 3: RAG Building | 1 week | Stage 2 complete |
| Stage 4: 6-Phase Coding | 4 weeks | Stage 3 complete |
| → Phase 1: Initial AI | 1 week | RAG index ready |
| → Phase 2: Consensus | 1 week | Phase 1 complete |
| → Phase 3: Human | 2 weeks | Sample selected |
| → Phase 4: Reliability | 3 days | Human coding complete |
| → Phase 5: Resolution | 1 week | Phase 4 metrics |
| → Phase 6: QA | 3 days | All resolved |
| Stage 5: Analysis & Writing | 4 weeks | Stage 4 complete |

### Budget Estimate

| Item | Cost |
|------|------|
| **API Costs (200 papers)** | |
| Phase 1: Claude Sonnet | $9.00 |
| Phase 2: GPT-4o | $7.00 |
| Phase 2: Groq Llama | $1.60 |
| Screening (Groq) | $5.00 |
| **API Subtotal** | **$22.60** |
| | |
| **Human Resources (Single Researcher)** | |
| Phase 3: Gold standard coding (20% = 40 papers × 15 min) | 10 hrs (self) |
| Phase 3: Intra-rater re-coding (8 papers × 15 min) | 2 hrs (self) |
| Phase 5: Discrepancy resolution | 5 hrs (self) |
| **Human Subtotal** | **~17 hrs (researcher time)** |
| | |
| **Total Estimated Cost** | **~$23 (API only)** |

**Note**: 단일 연구자 시나리오에서는 추가 인건비 없이 연구자 본인의 시간 투입으로 진행

### Python Dependencies

```
# Core
pandas>=2.0.0
numpy>=1.24.0

# LLM Integration
anthropic>=0.25.0
openai>=1.12.0
groq>=0.4.0

# RAG
langchain>=0.1.0
chromadb>=0.4.0
sentence-transformers>=2.2.0

# Inter-Coder Reliability
scikit-learn>=1.3.0
krippendorff>=0.6.0
pingouin>=0.5.0

# PDF Processing
pymupdf>=1.23.0
pdfplumber>=0.10.0

# Data Validation
pydantic>=2.0.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
```

---

## 11. Differences from GenAI-HE Project

| Aspect | GenAI-HE (Previous) | AI Ethics HR (New) |
|--------|--------------------|--------------------|
| Focus | Meta-analysis coding | Systematic review synthesis |
| Output | Effect sizes, statistics | Thematic analysis, framework |
| Schema | 27 quantitative fields | Qualitative coding categories |
| RAG Use | Moderator extraction | Full paper coding |
| Target | Research Synthesis Methods | HRD journals |
| Pipeline | extract_v10 | New: AI-Ethics-HR pipeline |
| **Validation** | Single model + human | **6-Phase multi-model consensus** ⭐ |
| **ICR Metrics** | Cohen's κ only | κ + Weighted κ + α + ICC |
| **Sensitivity** | Limited | Full model/RAG/prompt analysis |

---

## Summary

This plan outlines a **hybrid research approach** with **rigorous 6-Phase validation**:

1. **Conducts a rigorous systematic review** of AI ethics in HR across all HR functions
2. **Demonstrates RAG methodology** for literature review automation
3. **Targets HRD journals** with both substantive and methodological contributions
4. **Implements 6-Phase validation pipeline** for publication-quality coding accuracy ⭐

**Key Innovations**:
- First comprehensive RAG-enabled systematic review of AI ethics across the full HR function spectrum
- Multi-model consensus approach (Claude + GPT-4o + Groq) achieving 97% accuracy
- Rigorous inter-coder reliability validation (κ > 0.85, α > 0.80)
- Full sensitivity analysis across models, RAG configurations, and prompts
- Complete audit trail for reproducibility

---

## 12. GitHub Repository

**Repository**: https://github.com/HosungYou/Artificial-Intelligence-Ethics-in-Human-Resource

### Files to Upload on Completion
1. `data/` - Search results, screened papers, coded datasets
2. `scripts/` - All pipeline scripts (search, screen, RAG, 6-phase coding, analyze)
3. `scripts/05_code/` - Complete 6-Phase validation module ⭐
4. `codebook/` - Coding schema and manual
5. `codebook/coder_training/` - Human coder training protocol ⭐
6. `validation/` - Reliability reports, sensitivity analysis ⭐
7. `rag/configs/` - RAG configuration (excluding API keys)
8. `configs/phase_configs/` - All phase configurations ⭐
9. `manuscript/` - Final manuscript and supplementary materials
10. `README.md` - Project documentation and replication guide
11. `requirements.txt` - Python dependencies

### Open Science Commitment
- Pre-registration on OSF/PROSPERO before screening
- All code and data publicly available
- PRISMA 2020 checklist included
- Reproducible analysis pipeline
- **6-Phase validation audit trail** ⭐
- **Inter-coder reliability reports** ⭐
- **Sensitivity analysis results** ⭐

---

## 13. Detailed 6-Phase Reference

For complete technical specifications of the 6-Phase Validation Pipeline, see:
`/Users/hosung/.claude/plans/merry-zooming-shannon-agent-af94dec.md`

This reference document contains:
- Full Python dataclass definitions for all phases
- Detailed concordance comparison logic
- Human verification sampling strategy
- Complete ICR calculation implementations
- Discrepancy resolution flowchart
- Quality gate specifications
- Sensitivity analysis configurations
