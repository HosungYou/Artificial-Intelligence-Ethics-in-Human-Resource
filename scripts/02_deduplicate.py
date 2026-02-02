#!/usr/bin/env python3
"""
Stage 2: Deduplication Script for AI-Ethics-HR Systematic Review
Removes duplicate papers across multiple database sources.
"""

import os
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import re

# For fuzzy matching
try:
    from rapidfuzz import fuzz
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False
    logging.warning("rapidfuzz not installed. Install with: pip install rapidfuzz")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class DeduplicationStats:
    """Statistics from deduplication process."""
    total_input: int = 0
    duplicates_by_doi: int = 0
    duplicates_by_title: int = 0
    duplicates_by_arxiv: int = 0
    total_duplicates: int = 0
    unique_papers: int = 0
    sources_merged: Dict[str, int] = None

    def __post_init__(self):
        if self.sources_merged is None:
            self.sources_merged = {}


class PaperDeduplicator:
    """Handles deduplication of papers from multiple sources."""

    def __init__(self, title_similarity_threshold: float = 0.92):
        """
        Initialize deduplicator.

        Args:
            title_similarity_threshold: Minimum similarity (0-1) to consider titles as duplicates
        """
        self.title_threshold = title_similarity_threshold
        self.stats = DeduplicationStats()

    def normalize_doi(self, doi: Optional[str]) -> Optional[str]:
        """Normalize DOI to standard format."""
        if not doi:
            return None

        # Remove common prefixes
        doi = doi.strip().lower()
        prefixes = ['https://doi.org/', 'http://doi.org/', 'doi:', 'doi.org/']
        for prefix in prefixes:
            if doi.startswith(prefix):
                doi = doi[len(prefix):]

        # Validate DOI format
        if doi.startswith('10.') and '/' in doi:
            return doi

        return None

    def normalize_title(self, title: Optional[str]) -> str:
        """Normalize title for comparison."""
        if not title:
            return ""

        # Convert to lowercase
        title = title.lower()

        # Remove punctuation and extra whitespace
        title = re.sub(r'[^\w\s]', ' ', title)
        title = ' '.join(title.split())

        return title

    def extract_arxiv_id(self, paper: Dict) -> Optional[str]:
        """Extract arXiv ID from paper data."""
        # Check source_id for arXiv papers
        if paper.get('source') == 'arxiv':
            return paper.get('source_id')

        # Check URL
        url = paper.get('url', '') or ''
        if 'arxiv.org' in url:
            match = re.search(r'(\d{4}\.\d{4,5})', url)
            if match:
                return match.group(1)

        # Check DOI for arXiv DOIs
        doi = paper.get('doi', '') or ''
        if 'arxiv' in doi.lower():
            match = re.search(r'(\d{4}\.\d{4,5})', doi)
            if match:
                return match.group(1)

        return None

    def calculate_title_similarity(self, title1: str, title2: str) -> float:
        """Calculate similarity between two titles."""
        if not FUZZY_AVAILABLE:
            # Fall back to simple equality
            return 1.0 if title1 == title2 else 0.0

        return fuzz.ratio(title1, title2) / 100.0

    def deduplicate(self, papers: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Remove duplicates from list of papers.

        Args:
            papers: List of paper dictionaries

        Returns:
            Tuple of (deduplicated papers, duplicates log)
        """
        self.stats = DeduplicationStats(total_input=len(papers))

        # Track seen identifiers
        seen_dois: Set[str] = set()
        seen_arxiv_ids: Set[str] = set()
        seen_normalized_titles: Dict[str, Dict] = {}  # normalized_title -> paper

        unique_papers = []
        duplicates_log = []

        # Sort papers to prioritize by source (prefer Semantic Scholar/OpenAlex over arXiv)
        source_priority = {'semantic_scholar': 0, 'openalex': 1, 'scopus': 2, 'arxiv': 3}
        sorted_papers = sorted(
            papers,
            key=lambda p: source_priority.get(p.get('source', ''), 99)
        )

        for paper in sorted_papers:
            is_duplicate = False
            duplicate_reason = None
            duplicate_of = None

            # Check DOI
            doi = self.normalize_doi(paper.get('doi'))
            if doi:
                if doi in seen_dois:
                    is_duplicate = True
                    duplicate_reason = 'doi'
                    self.stats.duplicates_by_doi += 1
                else:
                    seen_dois.add(doi)

            # Check arXiv ID (if not already duplicate)
            if not is_duplicate:
                arxiv_id = self.extract_arxiv_id(paper)
                if arxiv_id:
                    if arxiv_id in seen_arxiv_ids:
                        is_duplicate = True
                        duplicate_reason = 'arxiv_id'
                        self.stats.duplicates_by_arxiv += 1
                    else:
                        seen_arxiv_ids.add(arxiv_id)

            # Check title similarity (if not already duplicate)
            if not is_duplicate:
                norm_title = self.normalize_title(paper.get('title'))
                if norm_title and len(norm_title) > 20:  # Skip very short titles
                    # Check against existing titles
                    for existing_title, existing_paper in seen_normalized_titles.items():
                        similarity = self.calculate_title_similarity(norm_title, existing_title)
                        if similarity >= self.title_threshold:
                            is_duplicate = True
                            duplicate_reason = 'title_similarity'
                            duplicate_of = existing_paper.get('title')
                            self.stats.duplicates_by_title += 1
                            break

                    if not is_duplicate:
                        seen_normalized_titles[norm_title] = paper

            if is_duplicate:
                duplicates_log.append({
                    'title': paper.get('title'),
                    'source': paper.get('source'),
                    'reason': duplicate_reason,
                    'duplicate_of': duplicate_of
                })
            else:
                unique_papers.append(paper)

                # Track source
                source = paper.get('source', 'unknown')
                self.stats.sources_merged[source] = self.stats.sources_merged.get(source, 0) + 1

        self.stats.total_duplicates = (
            self.stats.duplicates_by_doi +
            self.stats.duplicates_by_title +
            self.stats.duplicates_by_arxiv
        )
        self.stats.unique_papers = len(unique_papers)

        return unique_papers, duplicates_log

    def merge_paper_metadata(self, papers: List[Dict]) -> List[Dict]:
        """
        Merge metadata from duplicate papers to enrich records.

        For papers found in multiple databases, combines information
        like abstracts, citation counts, and PDF URLs.
        """
        # Group by DOI first
        doi_groups: Dict[str, List[Dict]] = defaultdict(list)
        no_doi_papers = []

        for paper in papers:
            doi = self.normalize_doi(paper.get('doi'))
            if doi:
                doi_groups[doi].append(paper)
            else:
                no_doi_papers.append(paper)

        merged_papers = []

        # Merge papers with same DOI
        for doi, group in doi_groups.items():
            if len(group) == 1:
                merged_papers.append(group[0])
            else:
                merged = self._merge_group(group)
                merged_papers.append(merged)

        # Add papers without DOI
        merged_papers.extend(no_doi_papers)

        return merged_papers

    def _merge_group(self, papers: List[Dict]) -> Dict:
        """Merge a group of duplicate papers into one enriched record."""
        # Use first paper as base (already sorted by priority)
        merged = papers[0].copy()

        for paper in papers[1:]:
            # Fill in missing abstract
            if not merged.get('abstract') and paper.get('abstract'):
                merged['abstract'] = paper['abstract']

            # Use higher citation count
            if paper.get('citation_count'):
                if not merged.get('citation_count') or paper['citation_count'] > merged['citation_count']:
                    merged['citation_count'] = paper['citation_count']

            # Add PDF URL if missing
            if not merged.get('pdf_url') and paper.get('pdf_url'):
                merged['pdf_url'] = paper['pdf_url']

            # Combine fields of study
            existing_fields = set(merged.get('fields_of_study', []))
            new_fields = set(paper.get('fields_of_study', []))
            merged['fields_of_study'] = list(existing_fields | new_fields)

        # Track sources
        merged['sources'] = list(set(p.get('source') for p in papers))

        return merged


def load_search_results(input_dir: str) -> List[Dict]:
    """Load all search results from input directory."""
    input_path = Path(input_dir)
    all_papers = []

    # Try to load combined results first
    combined_file = input_path / "all_search_results.json"
    if combined_file.exists():
        logger.info(f"Loading combined results from {combined_file}")
        with open(combined_file) as f:
            return json.load(f)

    # Otherwise, load individual database files
    for json_file in input_path.glob("*_results.json"):
        if json_file.name != "all_search_results.json":
            logger.info(f"Loading {json_file}")
            with open(json_file) as f:
                papers = json.load(f)
                all_papers.extend(papers)

    return all_papers


def save_deduplicated_results(
    papers: List[Dict],
    duplicates_log: List[Dict],
    stats: DeduplicationStats,
    output_dir: str
):
    """Save deduplicated results and statistics."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save deduplicated papers
    papers_file = output_path / "deduplicated_papers.json"
    with open(papers_file, "w") as f:
        json.dump(papers, f, indent=2)
    logger.info(f"Saved {len(papers)} unique papers to {papers_file}")

    # Save duplicates log
    duplicates_file = output_path / "duplicates_removed.json"
    with open(duplicates_file, "w") as f:
        json.dump(duplicates_log, f, indent=2)
    logger.info(f"Saved duplicates log to {duplicates_file}")

    # Save PRISMA deduplication summary
    summary = {
        "deduplication_date": datetime.now().isoformat(),
        "total_input": stats.total_input,
        "duplicates_removed": {
            "by_doi": stats.duplicates_by_doi,
            "by_title_similarity": stats.duplicates_by_title,
            "by_arxiv_id": stats.duplicates_by_arxiv,
            "total": stats.total_duplicates
        },
        "unique_papers": stats.unique_papers,
        "papers_by_source": stats.sources_merged
    }

    summary_file = output_path / "prisma_deduplication_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Deduplicate papers for AI Ethics in HR systematic review"
    )
    parser.add_argument(
        "--input", "-i",
        default="./data/01_search_results",
        help="Input directory with search results"
    )
    parser.add_argument(
        "--output", "-o",
        default="./data/02_deduplicated",
        help="Output directory for deduplicated results"
    )
    parser.add_argument(
        "--similarity-threshold", "-t",
        type=float,
        default=0.92,
        help="Title similarity threshold (0-1) for fuzzy matching"
    )
    parser.add_argument(
        "--merge-metadata",
        action="store_true",
        help="Merge metadata from duplicate papers before deduplication"
    )

    args = parser.parse_args()

    # Load papers
    logger.info(f"Loading papers from {args.input}")
    papers = load_search_results(args.input)
    logger.info(f"Loaded {len(papers)} total papers")

    if not papers:
        logger.error("No papers found to deduplicate")
        return

    # Initialize deduplicator
    deduplicator = PaperDeduplicator(
        title_similarity_threshold=args.similarity_threshold
    )

    # Optionally merge metadata first
    if args.merge_metadata:
        logger.info("Merging metadata from duplicate papers...")
        papers = deduplicator.merge_paper_metadata(papers)

    # Deduplicate
    logger.info("Starting deduplication...")
    unique_papers, duplicates_log = deduplicator.deduplicate(papers)
    stats = deduplicator.stats

    # Save results
    summary = save_deduplicated_results(
        unique_papers, duplicates_log, stats, args.output
    )

    # Print summary
    print("\n" + "="*60)
    print("STAGE 2: DEDUPLICATION COMPLETE")
    print("="*60)
    print(f"Total input papers:     {stats.total_input}")
    print(f"Duplicates removed:")
    print(f"  - By DOI:             {stats.duplicates_by_doi}")
    print(f"  - By title:           {stats.duplicates_by_title}")
    print(f"  - By arXiv ID:        {stats.duplicates_by_arxiv}")
    print(f"  - Total:              {stats.total_duplicates}")
    print(f"Unique papers:          {stats.unique_papers}")
    print("-"*60)
    print("Papers by source:")
    for source, count in stats.sources_merged.items():
        print(f"  - {source}: {count}")
    print("="*60)
    print(f"Results saved to: {args.output}")
    print("="*60)


if __name__ == "__main__":
    main()
