#!/usr/bin/env python3
"""
Stage 1: Database Search Script for AI-Ethics-HR Systematic Review
Searches Semantic Scholar, OpenAlex, and arXiv APIs for relevant papers.
"""

import os
import json
import time
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field
import requests
from urllib.parse import quote
import yaml

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class Paper:
    """Standardized paper metadata structure."""
    source: str
    source_id: str
    title: str
    authors: List[str]
    year: Optional[int]
    abstract: Optional[str]
    doi: Optional[str]
    url: Optional[str]
    pdf_url: Optional[str]
    venue: Optional[str]
    citation_count: Optional[int]
    fields_of_study: List[str] = field(default_factory=list)
    retrieved_at: str = field(default_factory=lambda: datetime.now().isoformat())


class SemanticScholarSearcher:
    """Search Semantic Scholar API."""

    BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
    FIELDS = "paperId,title,authors,year,abstract,venue,citationCount,fieldsOfStudy,externalIds,openAccessPdf,url"
    RATE_LIMIT_DELAY = 3.0  # seconds between requests

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("SEMANTIC_SCHOLAR_API_KEY")
        self.session = requests.Session()
        if self.api_key:
            self.session.headers["x-api-key"] = self.api_key

    def search(self, query: str, year_range: tuple, limit: int = 1000) -> List[Paper]:
        """Search for papers matching query within year range."""
        papers = []
        offset = 0
        batch_size = 100  # Max allowed by API

        year_filter = f"{year_range[0]}-{year_range[1]}"

        while offset < limit:
            try:
                params = {
                    "query": query,
                    "fields": self.FIELDS,
                    "offset": offset,
                    "limit": min(batch_size, limit - offset),
                    "year": year_filter
                }

                response = self.session.get(self.BASE_URL, params=params)
                response.raise_for_status()
                data = response.json()

                if not data.get("data"):
                    logger.info(f"Semantic Scholar: No more results at offset {offset}")
                    break

                for item in data["data"]:
                    paper = Paper(
                        source="semantic_scholar",
                        source_id=item.get("paperId", ""),
                        title=item.get("title", ""),
                        authors=[a.get("name", "") for a in item.get("authors", [])],
                        year=item.get("year"),
                        abstract=item.get("abstract"),
                        doi=item.get("externalIds", {}).get("DOI"),
                        url=item.get("url"),
                        pdf_url=item.get("openAccessPdf", {}).get("url") if item.get("openAccessPdf") else None,
                        venue=item.get("venue"),
                        citation_count=item.get("citationCount"),
                        fields_of_study=[f.get("category", "") for f in item.get("fieldsOfStudy", []) if f]
                    )
                    papers.append(paper)

                offset += len(data["data"])
                logger.info(f"Semantic Scholar: Retrieved {len(papers)} papers")

                time.sleep(self.RATE_LIMIT_DELAY)

            except requests.exceptions.RequestException as e:
                logger.error(f"Semantic Scholar API error: {e}")
                break

        return papers


class OpenAlexSearcher:
    """Search OpenAlex API."""

    BASE_URL = "https://api.openalex.org/works"
    RATE_LIMIT_DELAY = 0.1  # Polite pool allows faster requests

    def __init__(self, email: Optional[str] = None):
        self.email = email or os.getenv("OPENALEX_EMAIL", "research@example.com")
        self.session = requests.Session()

    def search(self, query: str, year_range: tuple, limit: int = 1000) -> List[Paper]:
        """Search for papers matching query within year range."""
        papers = []
        cursor = "*"

        # Build filter string
        filter_str = f"publication_year:{year_range[0]}-{year_range[1]}"

        while len(papers) < limit and cursor:
            try:
                params = {
                    "search": query,
                    "filter": filter_str,
                    "per-page": 100,
                    "cursor": cursor,
                    "mailto": self.email
                }

                response = self.session.get(self.BASE_URL, params=params)
                response.raise_for_status()
                data = response.json()

                results = data.get("results", [])
                if not results:
                    break

                for item in results:
                    # Extract authors
                    authors = []
                    for authorship in item.get("authorships", []):
                        author = authorship.get("author", {})
                        if author.get("display_name"):
                            authors.append(author["display_name"])

                    paper = Paper(
                        source="openalex",
                        source_id=item.get("id", "").replace("https://openalex.org/", ""),
                        title=item.get("title", ""),
                        authors=authors,
                        year=item.get("publication_year"),
                        abstract=item.get("abstract_inverted_index"),  # Needs reconstruction
                        doi=item.get("doi", "").replace("https://doi.org/", "") if item.get("doi") else None,
                        url=item.get("id"),
                        pdf_url=item.get("open_access", {}).get("oa_url"),
                        venue=item.get("primary_location", {}).get("source", {}).get("display_name") if item.get("primary_location") else None,
                        citation_count=item.get("cited_by_count"),
                        fields_of_study=[c.get("display_name", "") for c in item.get("concepts", [])[:5]]
                    )

                    # Reconstruct abstract from inverted index if present
                    if isinstance(paper.abstract, dict):
                        paper.abstract = self._reconstruct_abstract(paper.abstract)

                    papers.append(paper)

                cursor = data.get("meta", {}).get("next_cursor")
                logger.info(f"OpenAlex: Retrieved {len(papers)} papers")

                time.sleep(self.RATE_LIMIT_DELAY)

            except requests.exceptions.RequestException as e:
                logger.error(f"OpenAlex API error: {e}")
                break

        return papers[:limit]

    def _reconstruct_abstract(self, inverted_index: Dict) -> str:
        """Reconstruct abstract from OpenAlex inverted index format."""
        if not inverted_index:
            return ""

        # Create position -> word mapping
        positions = {}
        for word, pos_list in inverted_index.items():
            for pos in pos_list:
                positions[pos] = word

        # Reconstruct in order
        if positions:
            max_pos = max(positions.keys())
            words = [positions.get(i, "") for i in range(max_pos + 1)]
            return " ".join(words)
        return ""


class ArxivSearcher:
    """Search arXiv API."""

    BASE_URL = "http://export.arxiv.org/api/query"
    RATE_LIMIT_DELAY = 3.0  # arXiv requires 3-second delay

    def __init__(self):
        self.session = requests.Session()

    def search(self, query: str, year_range: tuple, limit: int = 500) -> List[Paper]:
        """Search arXiv for papers."""
        import xml.etree.ElementTree as ET

        papers = []
        start = 0
        max_results = 100

        # arXiv query syntax
        arxiv_query = f'all:"{query}"'

        while len(papers) < limit:
            try:
                params = {
                    "search_query": arxiv_query,
                    "start": start,
                    "max_results": min(max_results, limit - len(papers)),
                    "sortBy": "relevance"
                }

                response = self.session.get(self.BASE_URL, params=params)
                response.raise_for_status()

                root = ET.fromstring(response.content)
                ns = {"atom": "http://www.w3.org/2005/Atom"}

                entries = root.findall("atom:entry", ns)
                if not entries:
                    break

                for entry in entries:
                    # Extract year from published date
                    published = entry.find("atom:published", ns)
                    year = None
                    if published is not None and published.text:
                        year = int(published.text[:4])

                        # Filter by year range
                        if year < year_range[0] or year > year_range[1]:
                            continue

                    # Extract arXiv ID
                    entry_id = entry.find("atom:id", ns)
                    arxiv_id = entry_id.text.split("/abs/")[-1] if entry_id is not None else ""

                    # Extract authors
                    authors = []
                    for author in entry.findall("atom:author", ns):
                        name = author.find("atom:name", ns)
                        if name is not None and name.text:
                            authors.append(name.text)

                    # Extract title and abstract
                    title_elem = entry.find("atom:title", ns)
                    title = title_elem.text.strip().replace("\n", " ") if title_elem is not None else ""

                    summary_elem = entry.find("atom:summary", ns)
                    abstract = summary_elem.text.strip().replace("\n", " ") if summary_elem is not None else ""

                    # Extract categories
                    categories = []
                    for cat in entry.findall("atom:category", ns):
                        term = cat.get("term")
                        if term:
                            categories.append(term)

                    paper = Paper(
                        source="arxiv",
                        source_id=arxiv_id,
                        title=title,
                        authors=authors,
                        year=year,
                        abstract=abstract,
                        doi=None,  # arXiv papers may not have DOIs
                        url=f"https://arxiv.org/abs/{arxiv_id}",
                        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}.pdf",
                        venue="arXiv",
                        citation_count=None,
                        fields_of_study=categories
                    )
                    papers.append(paper)

                start += len(entries)
                logger.info(f"arXiv: Retrieved {len(papers)} papers")

                time.sleep(self.RATE_LIMIT_DELAY)

            except Exception as e:
                logger.error(f"arXiv API error: {e}")
                break

        return papers[:limit]


class SearchPipeline:
    """Orchestrates search across multiple databases."""

    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)

        # Initialize searchers
        self.searchers = {
            "semantic_scholar": SemanticScholarSearcher(),
            "openalex": OpenAlexSearcher(),
            "arxiv": ArxivSearcher()
        }

    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from YAML file."""
        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                return yaml.safe_load(f)

        # Default config
        return {
            "search": {
                "query": {
                    "filters": {
                        "year_range": [2015, 2025]
                    }
                }
            }
        }

    def search_all(self, query: str, databases: List[str] = None,
                   year_range: tuple = (2015, 2025),
                   limits: Dict[str, int] = None) -> Dict[str, List[Paper]]:
        """
        Search all specified databases.

        Args:
            query: Search query string
            databases: List of databases to search (default: all)
            year_range: Tuple of (start_year, end_year)
            limits: Dict of database -> max results

        Returns:
            Dict mapping database name to list of Paper objects
        """
        if databases is None:
            databases = list(self.searchers.keys())

        if limits is None:
            limits = {
                "semantic_scholar": 1000,
                "openalex": 1000,
                "arxiv": 500
            }

        results = {}

        for db in databases:
            if db not in self.searchers:
                logger.warning(f"Unknown database: {db}")
                continue

            logger.info(f"Searching {db}...")
            searcher = self.searchers[db]
            limit = limits.get(db, 500)

            papers = searcher.search(query, year_range, limit)
            results[db] = papers

            logger.info(f"{db}: Found {len(papers)} papers")

        return results

    def save_results(self, results: Dict[str, List[Paper]], output_dir: str):
        """Save search results to JSON files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        total_count = 0
        counts = {}

        for db, papers in results.items():
            # Save individual database results
            db_file = output_path / f"{db}_results.json"
            with open(db_file, "w") as f:
                json.dump([asdict(p) for p in papers], f, indent=2)

            counts[db] = len(papers)
            total_count += len(papers)
            logger.info(f"Saved {len(papers)} papers to {db_file}")

        # Save combined results
        all_papers = []
        for papers in results.values():
            all_papers.extend([asdict(p) for p in papers])

        combined_file = output_path / "all_search_results.json"
        with open(combined_file, "w") as f:
            json.dump(all_papers, f, indent=2)

        # Save PRISMA identification summary
        summary = {
            "search_date": datetime.now().isoformat(),
            "total_records_identified": total_count,
            "records_by_database": counts,
            "query_used": self.config.get("search", {}).get("query", {}).get("main", "")
        }

        summary_file = output_path / "prisma_identification_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"\nPRISMA Identification Complete:")
        logger.info(f"Total records identified: {total_count}")
        for db, count in counts.items():
            logger.info(f"  - {db}: {count}")

        return summary


def build_search_query() -> str:
    """Build the main search query for AI ethics in HR."""

    # AI/Technology terms
    ai_terms = [
        '"artificial intelligence"', '"AI"', '"machine learning"',
        '"algorithm*"', '"automated"', '"chatbot"', '"NLP"',
        '"predictive analytics"', '"deep learning"'
    ]

    # HR terms
    hr_terms = [
        '"human resource*"', '"HR"', '"HRM"', '"talent management"',
        '"recruitment"', '"selection"', '"hiring"', '"performance management"',
        '"learning and development"', '"training"', '"workforce analytics"',
        '"people analytics"', '"employee*"'
    ]

    # Ethics terms
    ethics_terms = [
        '"ethic*"', '"bias"', '"fairness"', '"discrimination"',
        '"transparency"', '"accountability"', '"privacy"',
        '"surveillance"', '"trust"', '"responsible AI"', '"algorithmic"'
    ]

    # Simplified query for API compatibility
    query = "artificial intelligence ethics human resource bias fairness"

    return query


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Search databases for AI Ethics in HR systematic review"
    )
    parser.add_argument(
        "--config", "-c",
        default="./configs/pipeline_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output", "-o",
        default="./data/01_search_results",
        help="Output directory for search results"
    )
    parser.add_argument(
        "--databases", "-d",
        nargs="+",
        default=["semantic_scholar", "openalex", "arxiv"],
        help="Databases to search"
    )
    parser.add_argument(
        "--year-start",
        type=int,
        default=2015,
        help="Start year for search"
    )
    parser.add_argument(
        "--year-end",
        type=int,
        default=2025,
        help="End year for search"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Maximum papers per database"
    )

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = SearchPipeline(args.config)

    # Build query
    query = build_search_query()
    logger.info(f"Search query: {query}")

    # Execute search
    limits = {db: args.limit for db in args.databases}
    results = pipeline.search_all(
        query=query,
        databases=args.databases,
        year_range=(args.year_start, args.year_end),
        limits=limits
    )

    # Save results
    summary = pipeline.save_results(results, args.output)

    print("\n" + "="*60)
    print("STAGE 1: IDENTIFICATION COMPLETE")
    print("="*60)
    print(f"Total records identified: {summary['total_records_identified']}")
    print(f"Results saved to: {args.output}")
    print("="*60)


if __name__ == "__main__":
    main()
