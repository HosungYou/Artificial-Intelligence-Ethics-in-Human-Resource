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

### 6. Stance Classification (NEW in v1.1)

**Purpose**: Classify the paper's argumentative position toward AI in HR.

#### 6.1 Overall Tone (Required)

| Value | Definition | Indicators |
|-------|------------|------------|
| `AI_optimistic` | Emphasizes benefits, downplays risks | "opportunity", "potential", "innovation", positive framing |
| `AI_critical` | Emphasizes risks, skeptical of benefits | "concerns", "dangers", "risks", cautionary framing |
| `balanced` | Equal weight to benefits and risks | Explicit acknowledgment of both sides |
| `neutral` | Descriptive without normative position | Objective, research-focused tone |

**Coding Rules**:
- Consider the OVERALL framing, not isolated statements
- Look at abstract, introduction, and conclusion for clearest signals
- "Balanced" requires explicit engagement with both perspectives

#### 6.2 Argument Basis (Required)

| Value | Definition | Indicators |
|-------|------------|------------|
| `evidence_based` | Arguments supported by empirical data | Statistics, experiments, survey data cited |
| `opinion_based` | Arguments based on expert opinion/theory | "We argue", "It is believed", theoretical claims |
| `mixed` | Combination of evidence and opinion | Both empirical and theoretical support |

#### 6.3 Per-Principle Stance (Conditional)

*Only coded when the corresponding ethical principle is mentioned (mentioned == true)*

| Value | Level | Definition |
|-------|-------|------------|
| `concern_high` | 4 | Presents as critical/urgent concern requiring immediate action |
| `concern_moderate` | 3 | Acknowledges concern with nuance and context |
| `concern_low` | 2 | Minimizes concern or views as manageable with standard practices |
| `solution_focused` | 1 | Focuses primarily on solutions rather than problems |

**Coding Rules**:
- Code separately for each of the 6 ethical principles
- Only code if that principle is mentioned in the paper
- Consider the dominant framing, not exceptional statements

---

### 7. Solution Taxonomy (NEW in v1.1)

**Purpose**: Classify solutions proposed to address AI ethics challenges.

#### 7.1 Solutions Proposed (Required)

**Boolean**: Does the paper propose specific solutions to ethical challenges?

**Coding Rules**:
- Answer `true` if the paper offers concrete recommendations
- General calls for "more research" do NOT count as solutions
- Must be actionable recommendations

#### 7.2 Technical Solutions (Multi-select)

*Only coded when solutions_proposed == true*

| Value | Definition | Examples |
|-------|------------|----------|
| `algorithm_audit` | Regular auditing of AI algorithms | "Regular fairness audits", "Algorithm testing" |
| `explainable_AI` | XAI techniques for interpretability | "SHAP values", "LIME explanations" |
| `fairness_constraints` | Mathematical fairness in training | "Demographic parity", "Equalized odds" |
| `differential_privacy` | Privacy-preserving techniques | "Differential privacy", "Federated learning" |
| `bias_detection` | Tools to detect bias | "Bias metrics", "Fairness dashboards" |
| `model_documentation` | Documentation practices | "Model cards", "Datasheets" |
| `human_AI_interface` | Improved interaction design | "User-friendly explanations", "Decision support" |
| `synthetic_data` | Synthetic data for bias reduction | "Data augmentation", "Synthetic minority sampling" |

#### 7.3 Organizational Solutions (Multi-select)

*Only coded when solutions_proposed == true*

| Value | Definition | Examples |
|-------|------------|----------|
| `human_oversight` | Human review of AI decisions | "Human-in-the-loop", "Manager review" |
| `ethics_committee` | Ethics review boards | "AI ethics board", "Review committee" |
| `training_programs` | Training for HR professionals | "AI literacy training", "Ethics workshops" |
| `policy_development` | Internal AI use policies | "AI governance policy", "Usage guidelines" |
| `stakeholder_engagement` | Involving employees in AI design | "Employee input", "Participatory design" |
| `impact_assessment` | Regular AI impact assessments | "Algorithmic impact assessment", "Risk review" |
| `grievance_mechanism` | Appeal channels for AI decisions | "Appeal process", "Challenge mechanism" |
| `role_redesign` | HR role restructuring | "HR-AI collaboration roles", "Job redesign" |

#### 7.4 Regulatory Solutions (Multi-select)

*Only coded when solutions_proposed == true*

| Value | Definition | Examples |
|-------|------------|----------|
| `legislation` | Government laws and regulations | "EU AI Act", "Employment law reform" |
| `industry_standards` | Industry standards/guidelines | "ISO standards", "Industry best practices" |
| `certification` | Third-party certification | "AI certification", "Ethical AI seal" |
| `external_audit` | External auditing requirements | "Independent audit", "Third-party review" |
| `regulatory_sandbox` | Experimental frameworks | "Regulatory sandbox", "Pilot programs" |
| `disclosure_requirements` | Mandatory disclosure of AI use | "Transparency requirements", "Notice obligations" |

#### 7.5 Empirical Validation (Conditional)

*Only coded when solutions_proposed == true*

**validated (Boolean)**: Were proposed solutions empirically tested?

**validation_type (Multi-select)**:
| Value | Definition |
|-------|------------|
| `experiment` | Controlled experiment or A/B test |
| `case_study` | Real-world implementation case |
| `simulation` | Simulation or synthetic evaluation |
| `survey` | Survey of stakeholder perceptions |
| `field_study` | Field deployment and evaluation |

---

### 8. Temporal Metadata (NEW in v1.1)

**Purpose**: Enable time-based analysis and evolution tracking.

#### 8.1 Publication Quarter (Optional)

Derived automatically from publication date.

| Value | Months |
|-------|--------|
| `Q1` | January - March |
| `Q2` | April - June |
| `Q3` | July - September |
| `Q4` | October - December |

#### 8.2 Research Period (Required)

| Value | Years | Characteristics |
|-------|-------|-----------------|
| `2015_2017` | 2015-2017 | Early period: foundational discussions, limited empirical work |
| `2018_2020` | 2018-2020 | Growth period: GDPR, increasing corporate awareness |
| `2021_2023` | 2021-2023 | Maturation period: regulatory focus, EU AI Act drafting |
| `2024_2025` | 2024-2025 | Current period: generative AI emergence, LLM applications |

**Coding Rules**:
- Use the publication year to determine period
- This enables trend analysis across the review timeline

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

stance_classification:
  overall_tone: AI_critical
  argument_basis: evidence_based
  per_principle_stance:
    fairness_bias: concern_high
    transparency: concern_moderate

solution_taxonomy:
  solutions_proposed: true
  technical_solutions: [algorithm_audit, bias_detection]
  organizational_solutions: [human_oversight, impact_assessment]
  regulatory_solutions: [external_audit]
  empirical_validation:
    validated: true
    validation_type: [experiment]

temporal_metadata:
  publication_quarter: Q2
  research_period: 2021_2023
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

stance_classification:
  overall_tone: balanced
  argument_basis: mixed
  per_principle_stance:
    transparency: concern_moderate
    accountability: concern_low
    privacy: concern_high
    autonomy: concern_moderate
    wellbeing: concern_low

solution_taxonomy:
  solutions_proposed: true
  technical_solutions: [differential_privacy, model_documentation]
  organizational_solutions: [policy_development, stakeholder_engagement, grievance_mechanism]
  regulatory_solutions: [legislation, disclosure_requirements]
  empirical_validation:
    validated: false

temporal_metadata:
  publication_quarter: Q3
  research_period: 2021_2023
```

### Example 3: Generative AI in HR (Optimistic Perspective)

**Title**: "ChatGPT for HR: Transforming Employee Experience Through AI Assistants"

**Coding**:
```yaml
hr_function:
  primary: employee_relations

ai_technology:
  types: [chatbot, generative_ai, nlp]
  specific_tools: [ChatGPT, GPT-4]

ethical_issues:
  fairness_bias:
    mentioned: true
    types: [algorithmic_bias]
    severity: mentioned
  transparency:
    mentioned: true
    types: [explainability, communication_to_employees]
    severity: discussed
  accountability:
    mentioned: false
  privacy:
    mentioned: true
    types: [data_collection, consent]
    severity: mentioned
  autonomy:
    mentioned: true
    types: [human_in_the_loop]
    severity: discussed
  wellbeing:
    mentioned: true
    types: [employee_experience, job_quality]
    severity: major_focus

theoretical_framework:
  applied: true
  theory_name: "Technology Acceptance Model (TAM)"
  category: technology_theories

key_findings:
  summary: "Generative AI chatbots significantly improved employee satisfaction with HR services. Implementation success depends on clear communication and maintaining human escalation paths."

stance_classification:
  overall_tone: AI_optimistic
  argument_basis: evidence_based
  per_principle_stance:
    fairness_bias: concern_low
    transparency: solution_focused
    privacy: concern_low
    autonomy: solution_focused
    wellbeing: solution_focused

solution_taxonomy:
  solutions_proposed: true
  technical_solutions: [human_AI_interface, explainable_AI]
  organizational_solutions: [human_oversight, training_programs, policy_development]
  regulatory_solutions: []
  empirical_validation:
    validated: true
    validation_type: [survey, field_study]

temporal_metadata:
  publication_quarter: Q1
  research_period: 2024_2025
```

---

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-02-02 | Initial version |
| 1.1 | 2026-02-03 | Added Stance Classification, Solution Taxonomy, and Temporal Metadata sections |
