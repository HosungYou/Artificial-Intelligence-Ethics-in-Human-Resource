# AI Ethics in HR Coding Manual

## Overview

This codebook provides detailed instructions for coding academic papers on AI ethics in human resource management. It accompanies the 6-Phase Validated Coding Pipeline for systematic literature review.

## Purpose

To ensure consistent, reliable extraction of information about:
1. HR functions addressed in AI applications
2. Types of AI technologies discussed
3. Ethical issues and concerns raised
4. Theoretical frameworks applied
5. Key findings and research gaps

## Coding Process

### Phase 1: Initial AI Coding
- Claude 3.5 Sonnet extracts data using RAG pipeline
- Each field receives a confidence score (0.0-1.0)
- Low confidence (<0.75) fields flagged for review

### Phase 2: Multi-Model Consensus
- Three models (Claude, GPT-4o, Llama) verify extractions
- 2/3 agreement required for consensus
- Discordant fields routed to Phase 5

### Phase 3: Human Verification
- 20% stratified sample coded by human
- Ensures coverage of all ethics principles and HR functions
- Creates gold standard for reliability calculation

### Phase 4: Inter-Coder Reliability
- Cohen's Kappa ≥ 0.85 for categorical fields
- Weighted Kappa ≥ 0.80 for ordinal fields
- Krippendorff's Alpha ≥ 0.80 for multi-select fields

### Phase 5: Discrepancy Resolution
- Human arbiter resolves disagreements
- All decisions documented with rationale

### Phase 6: Quality Assurance
- Final validation against quality gates
- Dataset completeness verification
- Confidence calibration

---

## Field Definitions

### 1. HR Function (Required)

**Definition**: The primary human resource function that the paper addresses in relation to AI applications.

| Value | Definition | Examples |
|-------|------------|----------|
| `recruitment` | Talent acquisition, job posting, sourcing candidates | Job ads, candidate sourcing, employer branding |
| `selection` | Screening, interviewing, hiring decisions | Resume screening, video interviews, assessment |
| `performance_management` | Evaluation, feedback, goal setting | Performance reviews, continuous feedback |
| `learning_development` | Training, skill development, career paths | Personalized learning, skill gap analysis |
| `people_analytics` | HR analytics, workforce planning | Predictive analytics, turnover prediction |
| `employee_relations` | Engagement, wellbeing, culture | Sentiment analysis, engagement surveys |
| `workforce_planning` | Strategic planning, succession | Demand forecasting, succession planning |
| `compensation_benefits` | Pay decisions, benefits | Pay equity analysis, benefits optimization |
| `multiple` | Paper addresses multiple functions equally | Cross-functional AI study |

**Coding Rules**:
- Code the MOST prominent function if multiple are discussed
- Use `multiple` only if 3+ functions receive equal attention
- If unclear, use evidence from abstract/introduction

---

### 2. AI Technology Type (Required)

**Definition**: The type of artificial intelligence technology discussed in the paper.

| Value | Definition | Examples |
|-------|------------|----------|
| `machine_learning` | General ML algorithms | Classification, regression, clustering |
| `nlp` | Natural language processing | Text analysis, sentiment analysis |
| `computer_vision` | Image/video analysis | Facial analysis, video screening |
| `chatbot` | Conversational AI | HR chatbots, virtual assistants |
| `predictive_analytics` | Predictive models | Turnover prediction, performance forecasting |
| `recommender_system` | Recommendation algorithms | Job matching, learning recommendations |
| `robotic_process_automation` | RPA for HR processes | Automated onboarding, document processing |
| `generative_ai` | LLMs, generative models | ChatGPT, content generation |
| `not_specified` | AI mentioned without specifics | General "AI in HR" discussion |

**Coding Rules**:
- Select ALL applicable technology types
- If specific tool mentioned (e.g., HireVue), also note in `specific_tool` field
- If paper is conceptual without specific technology, use `not_specified`

---

### 3. Ethical Issues (6 Principles)

Each ethical principle has three components:

#### a) mentioned (Boolean)
- `true`: Paper explicitly discusses this ethical concern
- `false`: No discussion of this concern

#### b) type (Multi-select)
- Select ALL applicable sub-types when mentioned=true

#### c) severity (Ordinal)
- `major_focus` (4): Central theme of the paper
- `discussed` (3): Substantive discussion (paragraph+)
- `mentioned` (2): Brief reference only
- `not_addressed` (1): Not discussed despite relevance

---

#### 3.1 Fairness & Bias

**Definition**: Concerns about algorithmic discrimination, unfair outcomes, or bias in AI-driven HR decisions.

| Type | Definition | Indicators |
|------|------------|------------|
| `algorithmic_bias` | Bias embedded in algorithms | "biased algorithm", "discriminatory AI" |
| `disparate_impact` | Disproportionate outcomes | "adverse impact", "disparate treatment" |
| `protected_characteristics` | Bias related to protected groups | Race, gender, age, disability mentions |
| `historical_bias` | Bias from training data | "historical data bias", "biased training data" |
| `proxy_discrimination` | Indirect discrimination | "proxy variables", "indirect discrimination" |

---

#### 3.2 Transparency & Explainability

**Definition**: Concerns about understanding AI decision-making processes.

| Type | Definition | Indicators |
|------|------------|------------|
| `explainability` | Ability to explain decisions | "explainable AI", "XAI", "explain decisions" |
| `black_box` | Opaque algorithms | "black box", "opaque", "lack of transparency" |
| `interpretability` | Understanding how AI works | "interpretable", "understandable" |
| `communication_to_employees` | Informing employees about AI | "disclosure", "inform employees" |

---

#### 3.3 Accountability

**Definition**: Concerns about responsibility for AI decisions and outcomes.

| Type | Definition | Indicators |
|------|------------|------------|
| `human_oversight` | Need for human supervision | "human-in-the-loop", "oversight", "review" |
| `liability` | Legal responsibility | "liability", "legal responsibility", "who is responsible" |
| `responsibility` | Organizational accountability | "accountable", "responsible party" |
| `auditability` | Ability to audit AI systems | "audit", "auditable", "review decisions" |

---

#### 3.4 Privacy & Data Protection

**Definition**: Concerns about employee data collection, use, and protection.

| Type | Definition | Indicators |
|------|------------|------------|
| `data_collection` | Data being collected | "data collected", "information gathered" |
| `surveillance` | Employee monitoring | "surveillance", "monitoring", "tracking" |
| `consent` | Informed consent | "consent", "permission", "opt-in" |
| `gdpr` | Data protection regulations | "GDPR", "data protection", "privacy law" |
| `data_minimization` | Collecting minimal data | "data minimization", "necessary data only" |

---

#### 3.5 Human Autonomy

**Definition**: Concerns about maintaining human agency and decision-making authority.

| Type | Definition | Indicators |
|------|------------|------------|
| `human_in_the_loop` | Maintaining human decision role | "human-in-the-loop", "final decision" |
| `deskilling` | Loss of human skills | "deskilling", "skill atrophy" |
| `agency` | Employee control and choice | "agency", "control", "choice" |
| `decision_authority` | Who makes final decisions | "decision authority", "who decides" |

---

#### 3.6 Employee Wellbeing

**Definition**: Concerns about AI impact on employee health and work quality.

| Type | Definition | Indicators |
|------|------------|------------|
| `job_quality` | Impact on job satisfaction | "job quality", "work satisfaction" |
| `psychological_safety` | Mental health impacts | "stress", "anxiety", "psychological safety" |
| `work_intensification` | Increased work pressure | "work intensification", "pressure" |
| `employee_experience` | Overall experience | "employee experience", "workplace quality" |

---

### 4. Theoretical Framework

**applied (Boolean)**: Does the paper use a theoretical/conceptual framework?

**theory_name (String)**: Name of the framework used.

Common frameworks in this domain:
- Algorithmic Management Theory
- Technology Acceptance Model (TAM)
- Procedural Justice Theory
- Stakeholder Theory
- Sociotechnical Systems Theory
- Rawlsian Ethics
- Utilitarian Ethics

**category (Categorical)**:
- `ethics_frameworks`: Ethical theories
- `hr_theories`: HR-specific theories
- `technology_theories`: Technology adoption theories
- `organizational_theories`: Organizational behavior theories

---

### 5. Key Findings

**summary**: 2-3 sentence summary of main findings

**Coding Rules**:
- Focus on findings related to AI ethics
- Include main conclusions about ethical implications
- Note any recommendations for practice

---

## Quality Targets

| Metric | Target | Acceptable |
|--------|--------|------------|
| Cohen's Kappa (categorical) | ≥ 0.85 | ≥ 0.80 |
| Weighted Kappa (ordinal) | ≥ 0.80 | ≥ 0.75 |
| Krippendorff's Alpha | ≥ 0.80 | ≥ 0.75 |
| ICC (continuous) | ≥ 0.90 | ≥ 0.85 |

---

## Examples

### Example 1: Recruitment Bias Paper

**Title**: "Algorithmic Hiring and Racial Bias: Evidence from Field Experiments"

**Coding**:
```yaml
hr_function:
  primary: recruitment

ai_technology:
  types: [machine_learning, nlp]
  specific_tools: []

ethical_issues:
  fairness_bias:
    mentioned: true
    types: [algorithmic_bias, disparate_impact, protected_characteristics]
    severity: major_focus
  transparency:
    mentioned: true
    types: [black_box]
    severity: discussed
  accountability:
    mentioned: false
  privacy:
    mentioned: false
  autonomy:
    mentioned: false
  wellbeing:
    mentioned: false

theoretical_framework:
  applied: true
  theory_name: "Procedural Justice Theory"
  category: organizational_theories

key_findings:
  summary: "AI hiring tools showed significant racial bias in resume screening. Recommendations include regular audits and human oversight of AI decisions."
```

### Example 2: People Analytics Privacy Paper

**Title**: "Employee Privacy in the Age of People Analytics"

**Coding**:
```yaml
hr_function:
  primary: people_analytics

ai_technology:
  types: [predictive_analytics]

ethical_issues:
  fairness_bias:
    mentioned: false
  transparency:
    mentioned: true
    types: [communication_to_employees]
    severity: discussed
  accountability:
    mentioned: true
    types: [human_oversight]
    severity: mentioned
  privacy:
    mentioned: true
    types: [data_collection, surveillance, consent, gdpr]
    severity: major_focus
  autonomy:
    mentioned: true
    types: [agency]
    severity: discussed
  wellbeing:
    mentioned: true
    types: [psychological_safety]
    severity: mentioned

theoretical_framework:
  applied: true
  theory_name: "Information Privacy Theory"
  category: organizational_theories

key_findings:
  summary: "Extensive employee monitoring raises significant privacy concerns. Organizations should implement transparent data practices and obtain meaningful consent."
```

---

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-02-02 | Initial version |
