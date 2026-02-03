# Methods Section
## AI Ethics in Human Resource Management: A Systematic Literature Review with AI-Assisted Data Extraction

---

### Study Design

This systematic literature review follows PRISMA 2020 guidelines (Page et al., 2021) to examine ethical issues associated with artificial intelligence applications across human resource management functions. Given the dual objectives of (a) synthesizing AI ethics literature in HRD and (b) validating an AI-assisted systematic review methodology, we employed a novel multi-model consensus approach for data extraction that combines retrieval-augmented generation (RAG) with human verification. This methodological innovation addresses the scalability challenges of traditional systematic reviews while maintaining rigorous quality standards (van de Schoot et al., 2021).

### Search Strategy

We conducted comprehensive searches across six academic databases on February 2, 2026: Scopus (n = 4,851), Web of Science (n = 1,307), PubMed (n = 437), ERIC (n = 302), Semantic Scholar (n = 500), and OpenAlex (n = 500). The search strategy combined controlled vocabulary and natural language terms across three concept domains: (a) artificial intelligence technologies (e.g., "artificial intelligence," "machine learning," "algorithmic management"), (b) human resource functions (e.g., "recruitment," "selection," "performance management"), and (c) ethical concerns (e.g., "fairness," "bias," "transparency," "accountability"). We limited searches to English-language, peer-reviewed publications from 2015-2025, capturing the accelerated growth period of AI applications in HRD contexts. The complete search strings for each database are provided in Supplementary Materials A.

### Screening Process

Following duplicate removal (776 duplicates identified via DOI and title similarity matching), we screened 7,121 unique records using an AI-assisted approach. We employed Groq's Llama-3.3-70b-versatile model (temperature = 0.1) to apply predefined inclusion criteria to titles and abstracts. Papers were included if they (a) addressed AI or machine learning applications in HR contexts, (b) discussed ethical implications substantively, (c) were published in peer-reviewed outlets between 2015-2025, and (d) were available in English. The LLM generated binary inclusion decisions with confidence scores and justifications for each decision.

To validate AI screening accuracy, we implemented stratified random sampling of 30% of screened papers (n = 1,994) for human verification, ensuring representation across decision categories (included, excluded, uncertain) and confidence score quartiles. This dual-screening approach follows recommendations by Marshall et al. (2019) for augmenting systematic reviews with machine learning while maintaining methodological rigor through human oversight. Initial screening results identified 1,118 papers (15.7%) for full-text assessment, 5,836 papers (82.0%) for exclusion, and 167 papers (2.3%) flagged as uncertain for human adjudication.

### Data Extraction Protocol

Data extraction employed a multi-phase validation protocol designed to maximize extraction accuracy while documenting reliability. We developed a structured coding schema based on six ethical principles prominent in AI ethics discourse: fairness and bias, transparency and explainability, accountability, privacy and data protection, human autonomy, and employee wellbeing (Jobin et al., 2019). For each principle, coders assessed whether the issue was mentioned (yes/no), classified specific sub-types (e.g., algorithmic bias, disparate impact), and rated severity of discussion (major focus, discussed, mentioned, not addressed). Additional variables captured HR function addressed, AI technology type, theoretical frameworks applied, research methodology, and geographic scope.

**Phase 1 (Initial AI Extraction)**: We built a retrieval-augmented generation (RAG) system using ChromaDB vector database with sentence-transformers embeddings (all-MiniLM-L6-v2) to index full-text PDFs. Claude 3.5 Sonnet (Anthropic) retrieved semantically relevant passages (k = 8 chunks, 1000-token windows with 200-token overlap) and extracted structured data following the predefined coding schema. Each extracted field received a confidence score (0.0-1.0). This approach addresses the context limitation challenges of LLMs while maintaining source attribution—a critical requirement for systematic review transparency.

**Phase 2 (Multi-Model Consensus)**: To mitigate single-model biases and hallucinations, we implemented consensus coding across three models: Claude 3.5 Sonnet (primary), GPT-4o (OpenAI), and Llama-3.3-70b (Groq). Fields required 2/3 agreement for consensus acceptance, with critical fields (e.g., ethical issues mentioned) requiring unanimous agreement. Discordant extractions were flagged for human arbitration. This multi-model approach follows established practices in robust AI systems (Wang et al., 2024) and aligns with emerging methodological guidance for AI-assisted research synthesis (Tay et al., 2025).

**Phase 3 (Human Verification)**: Using stratified random sampling, we selected 20% of papers across six ethical principles and seven HR functions for independent human coding by the lead researcher. This sampling strategy ensured adequate coverage of rare categories while maintaining statistical power for reliability estimation (Krippendorff, 2018). Human coding followed the same structured protocol with operational definitions provided in the codebook (see Supplementary Materials B).

**Phase 4 (Inter-Coder Reliability)**: We calculated multiple reliability metrics appropriate for different data types: Cohen's κ for categorical fields (target ≥ 0.85), weighted κ for ordinal severity ratings (target ≥ 0.80), and Krippendorff's α for multi-select fields (target ≥ 0.80). These thresholds align with standards for high-stakes medical and social science systematic reviews (Hallgren, 2012). Reliability calculations compared AI consensus codes (Phase 2 output) against human codes (Phase 3) on the verification sample.

**Phase 5 (Discrepancy Resolution)**: All AI-human disagreements underwent structured review. The human coder re-examined source texts with retrieved RAG evidence, adjudicated discrepancies, and documented rationales. This iterative process improved both extraction accuracy and codebook clarity. Resolution decisions formed the final authoritative dataset.

**Phase 6 (Quality Assurance)**: We validated the final dataset against pre-specified quality gates: (a) inter-coder reliability exceeding thresholds, (b) overall accuracy ≥ 90%, (c) systematic bias < 5%, (d) hallucination rate < 2%, and (e) 100% completeness on required fields. Failed gates triggered corrective protocols (e.g., schema revision, increased human verification, model recalibration).

### Positionality and Methodological Transparency

We acknowledge that this study represents a methodological innovation—AI-assisted framework-based data extraction—distinct from inductive thematic analysis where themes emerge from data (Braun & Clarke, 2006). Our extraction applied predefined ethical categories derived from existing AI ethics taxonomies, with the AI serving as an efficient research assistant rather than an autonomous analyst. Human expertise remained central to research design, schema development, interpretation, and synthesis. All prompts, RAG configurations, and model specifications are documented in the project repository to support reproducibility and critical evaluation of AI assistance claims.

### Data Analysis and Synthesis

Following finalized data extraction, we conducted descriptive synthesis examining distributions of ethical issues across HR functions, temporal trends in ethical discourse, theoretical frameworks employed, and methodological approaches. We generated contingency tables for ethics-by-HR-function patterns, identified research gaps through absence analysis, and synthesized findings narratively to address research questions. Data visualization and statistical summaries were produced using Python (pandas, matplotlib, seaborn). The complete analysis workflow is reproducible through documented scripts in the project repository.

### Methodological Contribution

This study extends emerging methodologies for AI-assisted systematic reviews in three ways. First, we demonstrate RAG-augmented extraction as a solution to LLM context limitations, enabling grounded coding with source attribution. Second, we validate multi-model consensus as a mechanism for reducing single-model biases and hallucinations. Third, we establish a six-phase validation protocol with explicit reliability targets, addressing reviewer concerns about AI-assisted research synthesis trustworthiness (Khraisha et al., 2024). While efficiency gains are notable (estimated 120+ hours of manual coding time), the primary value proposition is replicability and auditability—all coding decisions, confidence scores, and consensus resolutions are computationally traceable.

---

**Word Count**: 1,089 words

---

## References

Braun, V., & Clarke, V. (2006). Using thematic analysis in psychology. *Qualitative Research in Psychology, 3*(2), 77-101.

Hallgren, K. A. (2012). Computing inter-rater reliability for observational data: An overview and tutorial. *Tutorials in Quantitative Methods for Psychology, 8*(1), 23-34.

Jobin, A., Ienca, M., & Vayena, E. (2019). The global landscape of AI ethics guidelines. *Nature Machine Intelligence, 1*(9), 389-399.

Khraisha, Q., Put, S., Kappenberg, J., Warraitch, A., & Hadfield, K. (2024). Can large language models replace humans in systematic reviews? Evaluating GPT-4's efficacy in screening and extracting data from peer-reviewed and grey literature in multiple languages. *Research Synthesis Methods*, 15(4), 616-626.

Krippendorff, K. (2018). *Content analysis: An introduction to its methodology* (4th ed.). SAGE Publications.

Marshall, I. J., Noel-Storr, A., Kuiper, J., Thomas, J., & Wallace, B. C. (2019). Machine learning for identifying randomized controlled trials: An evaluation and practitioner's guide. *Research Synthesis Methods, 10*(1), 12-28.

Page, M. J., McKenzie, J. E., Bossuyt, P. M., Boutron, I., Hoffmann, T. C., Mulrow, C. D., et al. (2021). The PRISMA 2020 statement: An updated guideline for reporting systematic reviews. *BMJ*, 372, n71.

Tay, A., Agrawal, A., & Teh, E. (2025). Adopting AI for systematic literature reviews: Challenges and recommendations. *Journal of Academic Librarianship, 51*(1), 102795.

van de Schoot, R., de Bruin, J., Schram, R., Zahedi, P., de Boer, J., Weijdema, F., et al. (2021). An open source machine learning framework for efficient and transparent systematic reviews. *Nature Machine Intelligence, 3*(2), 125-133.

Wang, S., Liu, J., & Mihalcea, R. (2024). Ensemble methods for robust AI systems: A survey. *ACM Computing Surveys, 56*(3), 1-37.

---

## Supplementary Materials (Referenced)

**Supplementary Materials A**: Complete database search strings and API parameters

**Supplementary Materials B**: Coding manual with operational definitions and decision rules

**Supplementary Materials C**: Multi-model consensus protocol and adjudication rules

**Supplementary Materials D**: Complete reliability metrics by field and category

**Supplementary Materials E**: Sensitivity analysis results (model comparison, RAG configuration, temperature effects)
