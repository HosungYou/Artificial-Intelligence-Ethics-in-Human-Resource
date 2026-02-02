# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-02-02

### Added

#### Core Pipeline Scripts
- `01_search.py` - Multi-database search (Semantic Scholar, OpenAlex, arXiv)
- `02_deduplicate.py` - Deduplication with DOI, title fuzzy matching, arXiv ID
- `03_screen.py` - AI-assisted screening using Groq LLM (llama-3.3-70b)
- `04_build_rag.py` - RAG index builder with ChromaDB and sentence-transformers

#### 6-Phase Validated Coding Module
- `phase1_initial.py` - Initial AI coding with Claude 3.5 Sonnet via RAG
- `phase2_consensus.py` - Multi-model consensus (Claude + GPT-4o + Groq)
- `phase3_sampling.py` - Stratified human verification sampling (20%)
- `phase4_reliability.py` - Inter-coder reliability calculation (κ, α, ICC)
- `phase5_resolution.py` - Discrepancy resolution protocol with audit trail
- `phase6_qa.py` - Quality assurance and final dataset generation

#### Utility Modules
- `utils/metrics.py` - Statistical metrics (Cohen's κ, Weighted κ, Krippendorff's α, ICC)
- `utils/confidence.py` - Confidence calculation and isotonic calibration
- `utils/audit.py` - Audit trail logging for reproducibility

#### Sensitivity Analysis
- `07_sensitivity.py` - Model, RAG, temperature, and prompt sensitivity testing

#### Configuration
- `pipeline_config.yaml` - Main pipeline configuration
- `phase_configs/` - Individual phase configuration files (1-6)
- `coding_schema.yaml` - Complete field definitions for coding

#### Documentation
- `README.md` - Project overview and usage instructions
- `AI_Ethics_HR_Codebook.md` - Comprehensive coding manual
- `training_protocol.md` - Human coder training protocol
- `requirements.txt` - Python dependencies

### Technical Specifications
- Python 3.9+ required
- LLM providers: Anthropic (Claude), OpenAI (GPT-4o), Groq (Llama)
- Vector store: ChromaDB with all-MiniLM-L6-v2 embeddings
- Target reliability: κ ≥ 0.85, α ≥ 0.80
