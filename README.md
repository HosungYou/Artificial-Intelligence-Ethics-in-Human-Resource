# AI Ethics in HR: RAG-Enabled Systematic Literature Review

A comprehensive systematic literature review examining ethical issues associated with artificial intelligence applications across human resource management functions, enhanced with a RAG (Retrieval-Augmented Generation) methodology and 6-Phase Validated Coding Pipeline.

## Overview

This project implements a rigorous systematic review methodology to:
1. **Map AI ethics issues** across all HR functions (recruitment, performance, L&D, analytics)
2. **Demonstrate RAG methodology** for literature review automation
3. **Validate AI-assisted coding** through multi-model consensus and human verification

## Key Features

- **Multi-Database Search**: Semantic Scholar, OpenAlex, arXiv APIs
- **AI-Assisted Screening**: Groq LLM for cost-effective title/abstract screening
- **RAG-Powered Extraction**: ChromaDB vector store with Claude 3.5 Sonnet
- **6-Phase Validation**: Multi-model consensus achieving >97% accuracy
- **PRISMA 2020 Compliance**: Full documentation and flowcharts

## Project Structure

```
AI-Ethics-HR-Review/
├── data/
│   ├── 01_search_results/      # Raw API search results
│   ├── 02_deduplicated/        # After duplicate removal
│   ├── 03_screened/            # After title/abstract screening
│   ├── 04_full_text/           # PDF files
│   ├── 05_coded/               # 6-Phase coding outputs
│   │   ├── phase1_raw/
│   │   ├── phase2_consensus/
│   │   ├── phase3_human/
│   │   ├── phase4_reliability/
│   │   ├── phase5_resolutions/
│   │   └── phase6_final/
│   └── 06_analysis/
├── rag/
│   └── chroma_db/              # Vector database
├── scripts/
│   ├── 01_search.py
│   ├── 02_deduplicate.py
│   ├── 03_screen.py
│   ├── 04_build_rag.py
│   ├── 05_code/               # 6-Phase coding module
│   ├── 06_analyze.py
│   └── 07_sensitivity.py
├── codebook/
├── validation/
├── manuscript/
└── configs/
```

## Installation

```bash
# Clone repository
git clone https://github.com/HosungYou/Artificial-Intelligence-Ethics-in-Human-Resource.git
cd AI-Ethics-HR-Review

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file:

```
ANTHROPIC_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
SEMANTIC_SCHOLAR_API_KEY=optional
```

## Usage

### Full Pipeline

```bash
# Stage 1: Search databases
python scripts/01_search.py --output data/01_search_results

# Stage 2: Deduplicate
python scripts/02_deduplicate.py

# Stage 3: Screen papers
python scripts/03_screen.py

# Stage 4: Build RAG index
python scripts/04_build_rag.py

# Stage 5: 6-Phase Validated Coding
python scripts/05_code/phase1_initial.py
python scripts/05_code/phase2_consensus.py
python scripts/05_code/phase3_sampling.py
python scripts/05_code/phase4_reliability.py
python scripts/05_code/phase5_resolution.py
python scripts/05_code/phase6_qa.py

# Stage 6: Analysis
python scripts/06_analyze.py

# Sensitivity Analysis
python scripts/07_sensitivity.py
```

### Individual Phase Execution

```bash
# Run Phase 1 with custom RAG settings
python scripts/05_code/phase1_initial.py \
  --input data/03_screened/screened_included.json \
  --rag-dir rag/chroma_db \
  --model claude-3-5-sonnet-20241022

# Run Phase 2 consensus
python scripts/05_code/phase2_consensus.py \
  --input data/05_coded/phase1_raw/all_phase1_results.json
```

## 6-Phase Validation Pipeline

| Phase | Description | Output |
|-------|-------------|--------|
| 1 | Initial AI Coding (Claude) | `phase1_raw/*.json` |
| 2 | Multi-Model Consensus | `phase2_consensus/*.json` |
| 3 | Human Verification Sample | `phase3_human/*.csv` |
| 4 | Inter-Coder Reliability | `phase4_reliability/icr_report.json` |
| 5 | Discrepancy Resolution | `phase5_resolutions/resolutions_log.json` |
| 6 | Quality Assurance | `phase6_final/final_coded_dataset.csv` |

## Quality Targets

| Metric | Target | Description |
|--------|--------|-------------|
| Cohen's κ | ≥ 0.85 | Categorical field agreement |
| Weighted κ | ≥ 0.80 | Ordinal severity agreement |
| Krippendorff's α | ≥ 0.80 | Multi-model reliability |
| Accuracy | ≥ 90% | AI-human agreement |
| Hallucination | < 2% | AI error rate |

## Ethical Framework

Six ethical principles analyzed:
1. **Fairness & Bias**: Algorithmic discrimination, disparate impact
2. **Transparency**: Explainability, black-box concerns
3. **Accountability**: Human oversight, liability
4. **Privacy**: Surveillance, consent, data protection
5. **Autonomy**: Human-in-the-loop, deskilling
6. **Wellbeing**: Job quality, psychological safety

## Research Questions

**Primary (Systematic Review)**:
- RQ1: Key ethical issues across HR functions
- RQ2: Variation across HR domains
- RQ3: Theoretical frameworks applied
- RQ4: Research gaps identified

**Secondary (Methodology)**:
- RQ5: RAG-enabled methodology effectiveness
- RQ6: AI vs. human coding accuracy

## Citation

```bibtex
@article{you2026aiethicshr,
  title={Artificial Intelligence Ethics in Human Resource: A RAG-Enabled Systematic Literature Review},
  author={You, Hosung},
  journal={Human Resource Development Review},
  year={2026},
  note={Under review}
}
```

## License

MIT License

## Contact

Hosung You - hosung.you@example.com

Project Link: https://github.com/HosungYou/Artificial-Intelligence-Ethics-in-Human-Resource
