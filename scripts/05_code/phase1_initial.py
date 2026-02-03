#!/usr/bin/env python3
"""
Phase 1: Initial AI Coding with RAG
Uses Claude 3.5 Sonnet to extract structured data from papers via RAG pipeline.
"""

import os
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict, field
import yaml

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class FieldConfidence:
    """Confidence score for a coded field."""
    field_name: str
    value: Any
    confidence: float
    evidence_chunks: List[str] = field(default_factory=list)
    reasoning: str = ""


@dataclass
class Phase1CodingResult:
    """Result of Phase 1 coding for a single paper."""
    paper_id: str
    title: str
    coding_date: str
    model: str

    # Extracted fields with confidence
    hr_function: FieldConfidence = None
    ai_technology: FieldConfidence = None
    ethical_issues: Dict[str, FieldConfidence] = field(default_factory=dict)
    theoretical_framework: FieldConfidence = None
    key_findings: FieldConfidence = None

    # v1.1.0: New extracted fields
    stance_classification: FieldConfidence = None
    solution_taxonomy: FieldConfidence = None
    per_principle_stance: Dict[str, FieldConfidence] = field(default_factory=dict)
    temporal_metadata: Dict[str, str] = field(default_factory=dict)

    # Metadata
    total_chunks_retrieved: int = 0
    avg_confidence: float = 0.0
    low_confidence_fields: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'paper_id': self.paper_id,
            'title': self.title,
            'coding_date': self.coding_date,
            'model': self.model,
            'hr_function': asdict(self.hr_function) if self.hr_function else None,
            'ai_technology': asdict(self.ai_technology) if self.ai_technology else None,
            'ethical_issues': {k: asdict(v) for k, v in self.ethical_issues.items()},
            'theoretical_framework': asdict(self.theoretical_framework) if self.theoretical_framework else None,
            'key_findings': asdict(self.key_findings) if self.key_findings else None,
            # v1.1.0: New fields
            'stance_classification': asdict(self.stance_classification) if self.stance_classification else None,
            'solution_taxonomy': asdict(self.solution_taxonomy) if self.solution_taxonomy else None,
            'per_principle_stance': {k: asdict(v) for k, v in self.per_principle_stance.items()},
            'temporal_metadata': self.temporal_metadata,
            # Metadata
            'total_chunks_retrieved': self.total_chunks_retrieved,
            'avg_confidence': self.avg_confidence,
            'low_confidence_fields': self.low_confidence_fields
        }


class RAGRetriever:
    """Retrieve relevant chunks from ChromaDB."""

    def __init__(self, persist_directory: str, collection_name: str = "ai_ethics_hr"):
        try:
            import chromadb
            from chromadb.config import Settings
            from sentence_transformers import SentenceTransformer

            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
            self.collection = self.client.get_collection(collection_name)
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            self.available = True

        except Exception as e:
            logger.error(f"Failed to initialize RAG retriever: {e}")
            self.available = False

    def retrieve(
        self,
        query: str,
        paper_id: str,
        n_results: int = 8
    ) -> List[Dict]:
        """Retrieve relevant chunks for a query within a specific paper."""
        if not self.available:
            return []

        query_embedding = self.embedding_model.encode(query).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where={"paper_id": paper_id}
        )

        chunks = []
        if results and results.get('documents'):
            for i, doc in enumerate(results['documents'][0]):
                chunks.append({
                    'content': doc,
                    'metadata': results['metadatas'][0][i] if results.get('metadatas') else {},
                    'distance': results['distances'][0][i] if results.get('distances') else None
                })

        return chunks


class ClaudeExtractor:
    """Extract structured data using Claude API with RAG."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-3-5-sonnet-20241022",
        temperature: float = 0.1
    ):
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
            self.model = model
            self.temperature = temperature
            self.available = True
        except ImportError:
            logger.warning("anthropic not installed. Install with: pip install anthropic")
            self.available = False
        except Exception as e:
            logger.error(f"Failed to initialize Claude client: {e}")
            self.available = False

    def _build_extraction_prompt(self, paper: Dict, chunks: List[Dict]) -> str:
        """Build the extraction prompt with RAG context."""

        # Format context from chunks
        context_parts = []
        for i, chunk in enumerate(chunks):
            context_parts.append(f"[Chunk {i+1}]\n{chunk['content']}")
        context = "\n\n".join(context_parts)

        prompt = f"""You are an expert research assistant extracting structured data from an academic paper on AI ethics in human resource management.

PAPER METADATA:
Title: {paper.get('title', 'Unknown')}
Authors: {', '.join(paper.get('authors', []))}
Year: {paper.get('year', 'Unknown')}
Abstract: {paper.get('abstract', 'Not available')[:500]}...

RELEVANT EXCERPTS FROM PAPER:
{context}

EXTRACTION TASK:
Extract the following information with confidence scores (0.0-1.0). Be conservative with confidence - only assign high confidence if the information is explicitly stated.

1. **HR Function**: What HR function(s) does this paper primarily address?
   - Options: recruitment, selection, performance_management, learning_development, people_analytics, employee_relations, workforce_planning, compensation_benefits, multiple

2. **AI Technology**: What type of AI technology is discussed?
   - Options: machine_learning, nlp, computer_vision, chatbot, predictive_analytics, recommender_system, robotic_process_automation, generative_ai, not_specified
   - Also note any specific tools mentioned (e.g., HireVue, ChatGPT)

3. **Ethical Issues**: For EACH of the following, indicate if mentioned and severity:
   a) Fairness/Bias: algorithmic_bias, disparate_impact, protected_characteristics, historical_bias, proxy_discrimination
   b) Transparency: explainability, black_box, interpretability, communication_to_employees
   c) Accountability: human_oversight, liability, responsibility, auditability
   d) Privacy: data_collection, surveillance, consent, gdpr, data_minimization
   e) Autonomy: human_in_the_loop, deskilling, agency, decision_authority
   f) Wellbeing: job_quality, psychological_safety, work_intensification, employee_experience

   Severity levels: major_focus (4), discussed (3), mentioned (2), not_addressed (1)

4. **Theoretical Framework**: Does the paper use a theoretical framework?
   - If yes, name it and categorize: ethics_frameworks, hr_theories, technology_theories, organizational_theories

5. **Key Findings**: Summarize main findings in 2-3 sentences.

6. **Stance Classification** (NEW):
   a) Overall Tone: AI_optimistic, AI_critical, balanced, neutral
   b) Argument Basis: evidence_based, opinion_based, mixed
   c) Per-Principle Stance (for each mentioned ethical principle):
      - concern_high (4), concern_moderate (3), concern_low (2), solution_focused (1)

7. **Solution Taxonomy** (NEW):
   a) Solutions Proposed: true/false
   b) If true, categorize solutions:
      - Technical: algorithm_audit, explainable_AI, fairness_constraints, differential_privacy, bias_detection, model_documentation, human_AI_interface, synthetic_data
      - Organizational: human_oversight, ethics_committee, training_programs, policy_development, stakeholder_engagement, impact_assessment, grievance_mechanism, role_redesign
      - Regulatory: legislation, industry_standards, certification, external_audit, regulatory_sandbox, disclosure_requirements
   c) Empirical Validation: validated (true/false), validation_type (experiment, case_study, simulation, survey, field_study)

8. **Temporal Metadata** (NEW):
   - Research Period: 2015_2017, 2018_2020, 2021_2023, 2024_2025 (based on publication year)

RESPOND IN JSON FORMAT:
{{
    "hr_function": {{
        "primary": "string",
        "secondary": ["list"],
        "confidence": 0.0-1.0,
        "evidence": "quote or reasoning"
    }},
    "ai_technology": {{
        "types": ["list"],
        "specific_tools": ["list"],
        "confidence": 0.0-1.0,
        "evidence": "quote or reasoning"
    }},
    "ethical_issues": {{
        "fairness_bias": {{
            "mentioned": true/false,
            "types": ["list"],
            "severity": "string",
            "confidence": 0.0-1.0,
            "evidence": "quote"
        }},
        "transparency": {{ ... }},
        "accountability": {{ ... }},
        "privacy": {{ ... }},
        "autonomy": {{ ... }},
        "wellbeing": {{ ... }}
    }},
    "theoretical_framework": {{
        "applied": true/false,
        "theory_name": "string or null",
        "category": "string or null",
        "confidence": 0.0-1.0,
        "evidence": "quote or reasoning"
    }},
    "key_findings": {{
        "summary": "2-3 sentences",
        "confidence": 0.0-1.0
    }},
    "stance_classification": {{
        "overall_tone": "AI_optimistic|AI_critical|balanced|neutral",
        "argument_basis": "evidence_based|opinion_based|mixed",
        "confidence": 0.0-1.0,
        "evidence": "quote or reasoning"
    }},
    "per_principle_stance": {{
        "fairness_bias": "concern_high|concern_moderate|concern_low|solution_focused or null",
        "transparency": "...",
        "accountability": "...",
        "privacy": "...",
        "autonomy": "...",
        "wellbeing": "..."
    }},
    "solution_taxonomy": {{
        "solutions_proposed": true/false,
        "technical_solutions": ["list of applicable solutions"],
        "organizational_solutions": ["list"],
        "regulatory_solutions": ["list"],
        "empirical_validation": {{
            "validated": true/false,
            "validation_type": ["list"]
        }},
        "confidence": 0.0-1.0,
        "evidence": "quote or reasoning"
    }},
    "temporal_metadata": {{
        "research_period": "2015_2017|2018_2020|2021_2023|2024_2025"
    }}
}}"""

        return prompt

    def extract(self, paper: Dict, chunks: List[Dict]) -> Dict:
        """Extract structured data from a paper using Claude."""
        if not self.available:
            return {}

        prompt = self._build_extraction_prompt(paper, chunks)

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4000,
                temperature=self.temperature,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            content = response.content[0].text

            # Extract JSON from response
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            return json.loads(content)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Claude response: {e}")
            return {}
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            return {}


class Phase1Coder:
    """Orchestrates Phase 1 coding process."""

    ETHICS_PRINCIPLES = [
        'fairness_bias', 'transparency', 'accountability',
        'privacy', 'autonomy', 'wellbeing'
    ]

    CONFIDENCE_THRESHOLD = 0.75

    def __init__(
        self,
        rag_directory: str,
        collection_name: str = "ai_ethics_hr",
        model: str = "claude-3-5-sonnet-20241022"
    ):
        self.retriever = RAGRetriever(rag_directory, collection_name)
        self.extractor = ClaudeExtractor(model=model)
        self.model = model

    def code_paper(self, paper: Dict) -> Phase1CodingResult:
        """Code a single paper using RAG-assisted extraction."""
        paper_id = paper.get('source_id', 'unknown')

        # Retrieve relevant chunks for different aspects
        all_chunks = []

        # Query for HR function
        hr_chunks = self.retriever.retrieve(
            "human resource HR function recruitment selection performance training analytics",
            paper_id, n_results=4
        )
        all_chunks.extend(hr_chunks)

        # Query for AI technology
        ai_chunks = self.retriever.retrieve(
            "artificial intelligence machine learning algorithm AI technology automation",
            paper_id, n_results=4
        )
        all_chunks.extend(ai_chunks)

        # Query for ethics
        ethics_chunks = self.retriever.retrieve(
            "ethics bias fairness transparency accountability privacy autonomy",
            paper_id, n_results=8
        )
        all_chunks.extend(ethics_chunks)

        # Remove duplicates based on content
        seen_content = set()
        unique_chunks = []
        for chunk in all_chunks:
            content = chunk.get('content', '')[:100]
            if content not in seen_content:
                seen_content.add(content)
                unique_chunks.append(chunk)

        # Extract using Claude
        extraction = self.extractor.extract(paper, unique_chunks[:12])  # Max 12 chunks

        # Build result
        result = Phase1CodingResult(
            paper_id=paper_id,
            title=paper.get('title', ''),
            coding_date=datetime.now().isoformat(),
            model=self.model,
            total_chunks_retrieved=len(unique_chunks)
        )

        # Process extracted fields
        if extraction:
            result = self._process_extraction(result, extraction, paper)

        return result

    def _process_extraction(
        self,
        result: Phase1CodingResult,
        extraction: Dict,
        paper: Dict = None
    ) -> Phase1CodingResult:
        """Process raw extraction into structured result with confidence."""

        # HR Function
        hr_data = extraction.get('hr_function', {})
        if hr_data:
            result.hr_function = FieldConfidence(
                field_name='hr_function',
                value={'primary': hr_data.get('primary'), 'secondary': hr_data.get('secondary', [])},
                confidence=hr_data.get('confidence', 0.5),
                evidence_chunks=[hr_data.get('evidence', '')],
                reasoning=hr_data.get('evidence', '')
            )

        # AI Technology
        ai_data = extraction.get('ai_technology', {})
        if ai_data:
            result.ai_technology = FieldConfidence(
                field_name='ai_technology',
                value={'types': ai_data.get('types', []), 'specific_tools': ai_data.get('specific_tools', [])},
                confidence=ai_data.get('confidence', 0.5),
                evidence_chunks=[ai_data.get('evidence', '')],
                reasoning=ai_data.get('evidence', '')
            )

        # Ethical Issues
        ethics_data = extraction.get('ethical_issues', {})
        for principle in self.ETHICS_PRINCIPLES:
            principle_data = ethics_data.get(principle, {})
            if principle_data:
                result.ethical_issues[principle] = FieldConfidence(
                    field_name=f'ethical_issues.{principle}',
                    value={
                        'mentioned': principle_data.get('mentioned', False),
                        'types': principle_data.get('types', []),
                        'severity': principle_data.get('severity', 'not_addressed')
                    },
                    confidence=principle_data.get('confidence', 0.5),
                    evidence_chunks=[principle_data.get('evidence', '')],
                    reasoning=principle_data.get('evidence', '')
                )

        # Theoretical Framework
        theory_data = extraction.get('theoretical_framework', {})
        if theory_data:
            result.theoretical_framework = FieldConfidence(
                field_name='theoretical_framework',
                value={
                    'applied': theory_data.get('applied', False),
                    'theory_name': theory_data.get('theory_name'),
                    'category': theory_data.get('category')
                },
                confidence=theory_data.get('confidence', 0.5),
                evidence_chunks=[theory_data.get('evidence', '')],
                reasoning=theory_data.get('evidence', '')
            )

        # Key Findings
        findings_data = extraction.get('key_findings', {})
        if findings_data:
            result.key_findings = FieldConfidence(
                field_name='key_findings',
                value={'summary': findings_data.get('summary', '')},
                confidence=findings_data.get('confidence', 0.5),
                evidence_chunks=[],
                reasoning=""
            )

        # v1.1.0: Stance Classification
        stance_data = extraction.get('stance_classification', {})
        if stance_data:
            result.stance_classification = FieldConfidence(
                field_name='stance_classification',
                value={
                    'overall_tone': stance_data.get('overall_tone'),
                    'argument_basis': stance_data.get('argument_basis')
                },
                confidence=stance_data.get('confidence', 0.5),
                evidence_chunks=[stance_data.get('evidence', '')],
                reasoning=stance_data.get('evidence', '')
            )

        # v1.1.0: Per-Principle Stance
        per_stance_data = extraction.get('per_principle_stance', {})
        for principle in self.ETHICS_PRINCIPLES:
            stance_value = per_stance_data.get(principle)
            if stance_value and stance_value != 'null':
                result.per_principle_stance[principle] = FieldConfidence(
                    field_name=f'per_principle_stance.{principle}',
                    value=stance_value,
                    confidence=0.7,  # Default confidence for derived field
                    evidence_chunks=[],
                    reasoning=""
                )

        # v1.1.0: Solution Taxonomy
        solution_data = extraction.get('solution_taxonomy', {})
        if solution_data:
            result.solution_taxonomy = FieldConfidence(
                field_name='solution_taxonomy',
                value={
                    'solutions_proposed': solution_data.get('solutions_proposed', False),
                    'technical_solutions': solution_data.get('technical_solutions', []),
                    'organizational_solutions': solution_data.get('organizational_solutions', []),
                    'regulatory_solutions': solution_data.get('regulatory_solutions', []),
                    'empirical_validation': solution_data.get('empirical_validation', {})
                },
                confidence=solution_data.get('confidence', 0.5),
                evidence_chunks=[solution_data.get('evidence', '')],
                reasoning=solution_data.get('evidence', '')
            )

        # v1.1.0: Temporal Metadata (derive from paper year if not extracted)
        temporal_data = extraction.get('temporal_metadata', {})
        paper_year = paper.get('year') if paper else None
        if temporal_data.get('research_period'):
            result.temporal_metadata['research_period'] = temporal_data['research_period']
        elif paper_year:
            # Derive research period from publication year
            year = int(paper_year) if isinstance(paper_year, str) else paper_year
            if year <= 2017:
                result.temporal_metadata['research_period'] = '2015_2017'
            elif year <= 2020:
                result.temporal_metadata['research_period'] = '2018_2020'
            elif year <= 2023:
                result.temporal_metadata['research_period'] = '2021_2023'
            else:
                result.temporal_metadata['research_period'] = '2024_2025'

        # Calculate average confidence and identify low-confidence fields
        confidences = []
        low_conf_fields = []

        for field_name, field_conf in [
            ('hr_function', result.hr_function),
            ('ai_technology', result.ai_technology),
            ('theoretical_framework', result.theoretical_framework),
            ('key_findings', result.key_findings),
            ('stance_classification', result.stance_classification),
            ('solution_taxonomy', result.solution_taxonomy)
        ]:
            if field_conf:
                confidences.append(field_conf.confidence)
                if field_conf.confidence < self.CONFIDENCE_THRESHOLD:
                    low_conf_fields.append(field_name)

        for principle, field_conf in result.ethical_issues.items():
            confidences.append(field_conf.confidence)
            if field_conf.confidence < self.CONFIDENCE_THRESHOLD:
                low_conf_fields.append(f'ethical_issues.{principle}')

        for principle, field_conf in result.per_principle_stance.items():
            confidences.append(field_conf.confidence)
            if field_conf.confidence < self.CONFIDENCE_THRESHOLD:
                low_conf_fields.append(f'per_principle_stance.{principle}')

        result.avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        result.low_confidence_fields = low_conf_fields

        return result

    def code_batch(
        self,
        papers: List[Dict],
        progress_callback=None
    ) -> List[Phase1CodingResult]:
        """Code a batch of papers."""
        results = []

        for i, paper in enumerate(papers):
            try:
                result = self.code_paper(paper)
                results.append(result)

                if progress_callback:
                    progress_callback(i + 1, len(papers), result)

            except Exception as e:
                logger.error(f"Failed to code paper {paper.get('source_id', 'unknown')}: {e}")
                # Create empty result
                results.append(Phase1CodingResult(
                    paper_id=paper.get('source_id', 'unknown'),
                    title=paper.get('title', ''),
                    coding_date=datetime.now().isoformat(),
                    model=self.model
                ))

        return results


def load_papers(input_path: str) -> List[Dict]:
    """Load papers from JSON file."""
    with open(input_path) as f:
        return json.load(f)


def save_results(results: List[Phase1CodingResult], output_dir: str):
    """Save Phase 1 coding results."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save individual results
    for result in results:
        result_file = output_path / f"{result.paper_id}_phase1.json"
        with open(result_file, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

    # Save summary
    summary = {
        "phase": 1,
        "coding_date": datetime.now().isoformat(),
        "total_papers": len(results),
        "avg_confidence": sum(r.avg_confidence for r in results) / len(results) if results else 0,
        "papers_with_low_confidence": sum(1 for r in results if r.low_confidence_fields),
        "papers_processed": [r.paper_id for r in results]
    }

    summary_file = output_path / "phase1_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    # Save combined results for Phase 2
    combined = [r.to_dict() for r in results]
    combined_file = output_path / "all_phase1_results.json"
    with open(combined_file, "w") as f:
        json.dump(combined, f, indent=2)

    logger.info(f"Saved {len(results)} Phase 1 results to {output_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Phase 1: Initial AI Coding with RAG"
    )
    parser.add_argument(
        "--input", "-i",
        default="./data/03_screened/screened_included.json",
        help="Input file with screened papers"
    )
    parser.add_argument(
        "--rag-dir",
        default="./rag/chroma_db",
        help="Directory with ChromaDB index"
    )
    parser.add_argument(
        "--output", "-o",
        default="./data/05_coded/phase1_raw",
        help="Output directory for Phase 1 results"
    )
    parser.add_argument(
        "--model",
        default="claude-3-5-sonnet-20241022",
        help="Claude model to use"
    )
    parser.add_argument(
        "--collection",
        default="ai_ethics_hr",
        help="ChromaDB collection name"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of papers (for testing)"
    )

    args = parser.parse_args()

    # Load papers
    logger.info(f"Loading papers from {args.input}")
    papers = load_papers(args.input)

    if args.limit:
        papers = papers[:args.limit]

    logger.info(f"Coding {len(papers)} papers")

    # Initialize coder
    coder = Phase1Coder(
        rag_directory=args.rag_dir,
        collection_name=args.collection,
        model=args.model
    )

    # Progress callback
    def progress(current, total, result):
        conf = f"{result.avg_confidence:.2f}" if result.avg_confidence else "N/A"
        logger.info(f"[{current}/{total}] {result.paper_id} - Confidence: {conf}")

    # Code papers
    results = coder.code_batch(papers, progress_callback=progress)

    # Save results
    save_results(results, args.output)

    # Print summary
    avg_conf = sum(r.avg_confidence for r in results) / len(results) if results else 0
    low_conf_count = sum(1 for r in results if r.low_confidence_fields)

    print("\n" + "="*60)
    print("PHASE 1: INITIAL AI CODING COMPLETE")
    print("="*60)
    print(f"Papers coded:           {len(results)}")
    print(f"Average confidence:     {avg_conf:.2%}")
    print(f"Papers with low conf:   {low_conf_count}")
    print(f"Model used:             {args.model}")
    print("="*60)
    print(f"Results saved to: {args.output}")
    print("="*60)


if __name__ == "__main__":
    main()
