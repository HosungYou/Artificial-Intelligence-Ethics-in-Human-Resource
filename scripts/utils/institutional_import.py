"""
Institutional Database Import Module

Handles import of manually exported files from:
- Scopus (CSV, RIS, BibTeX)
- Web of Science (CSV, RIS, BibTeX)
- PubMed/MEDLINE (CSV, RIS, XML)
- ERIC (CSV, RIS)

Usage:
    from utils.institutional_import import InstitutionalImporter

    importer = InstitutionalImporter()
    papers = importer.import_all("data/01_search_results/institutional/")
"""

import csv
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime


@dataclass
class Paper:
    """Standardized paper representation."""
    paper_id: str
    title: str
    authors: List[str]
    year: int
    abstract: str
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    pdf_url: Optional[str] = None
    source: str = "unknown"
    venue: Optional[str] = None
    citation_count: int = 0
    keywords: List[str] = None

    def __post_init__(self):
        if self.keywords is None:
            self.keywords = []


class RISParser:
    """Parser for RIS (Research Information Systems) format."""

    TAG_MAP = {
        'TI': 'title',
        'T1': 'title',
        'AU': 'authors',
        'A1': 'authors',
        'PY': 'year',
        'Y1': 'year',
        'AB': 'abstract',
        'N2': 'abstract',
        'DO': 'doi',
        'JO': 'venue',
        'JF': 'venue',
        'T2': 'venue',
        'KW': 'keywords',
    }

    def parse(self, filepath: Path) -> List[Dict]:
        """Parse RIS file and return list of paper dictionaries."""
        papers = []
        current_paper = {}

        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()

                if line.startswith('ER  -'):
                    if current_paper:
                        papers.append(self._finalize_paper(current_paper))
                    current_paper = {}
                    continue

                if '  - ' in line:
                    tag = line[:2]
                    value = line[6:].strip()

                    if tag in self.TAG_MAP:
                        field = self.TAG_MAP[tag]

                        if field in ['authors', 'keywords']:
                            if field not in current_paper:
                                current_paper[field] = []
                            current_paper[field].append(value)
                        elif field == 'year':
                            # Extract year from various formats
                            year_match = re.search(r'\d{4}', value)
                            if year_match:
                                current_paper[field] = int(year_match.group())
                        else:
                            current_paper[field] = value

        return papers

    def _finalize_paper(self, paper: Dict) -> Dict:
        """Finalize paper dictionary with defaults."""
        return {
            'title': paper.get('title', ''),
            'authors': paper.get('authors', []),
            'year': paper.get('year', 0),
            'abstract': paper.get('abstract', ''),
            'doi': paper.get('doi'),
            'venue': paper.get('venue'),
            'keywords': paper.get('keywords', []),
        }


class ScopusImporter:
    """Import papers from Scopus CSV export."""

    def import_csv(self, filepath: Path) -> List[Paper]:
        """Import Scopus CSV export."""
        papers = []

        with open(filepath, 'r', encoding='utf-8-sig', errors='ignore') as f:
            reader = csv.DictReader(f)

            for row in reader:
                try:
                    # Parse authors
                    authors_str = row.get('Authors', '') or row.get('Author full names', '')
                    authors = [a.strip() for a in authors_str.split(';') if a.strip()]

                    # Parse year
                    year_str = row.get('Year', '') or row.get('Publication year', '')
                    year = int(year_str) if year_str.isdigit() else 0

                    # Parse keywords
                    keywords_str = row.get('Author Keywords', '') or row.get('Index Keywords', '')
                    keywords = [k.strip() for k in keywords_str.split(';') if k.strip()]

                    paper = Paper(
                        paper_id=f"scopus_{row.get('EID', '')}",
                        title=row.get('Title', ''),
                        authors=authors,
                        year=year,
                        abstract=row.get('Abstract', ''),
                        doi=row.get('DOI'),
                        source='scopus',
                        venue=row.get('Source title', ''),
                        citation_count=int(row.get('Cited by', 0) or 0),
                        keywords=keywords,
                    )

                    if paper.title:  # Only add if has title
                        papers.append(paper)

                except Exception as e:
                    print(f"Warning: Failed to parse Scopus row: {e}")
                    continue

        return papers


class WebOfScienceImporter:
    """Import papers from Web of Science export."""

    def import_csv(self, filepath: Path) -> List[Paper]:
        """Import Web of Science tab-delimited or CSV export."""
        papers = []

        # Try to detect delimiter
        with open(filepath, 'r', encoding='utf-8-sig', errors='ignore') as f:
            first_line = f.readline()
            delimiter = '\t' if '\t' in first_line else ','

        with open(filepath, 'r', encoding='utf-8-sig', errors='ignore') as f:
            reader = csv.DictReader(f, delimiter=delimiter)

            for row in reader:
                try:
                    # Parse authors (WoS uses different field names)
                    authors_str = (row.get('AU', '') or row.get('Authors', '') or
                                   row.get('Author Full Names', ''))
                    authors = [a.strip() for a in authors_str.split(';') if a.strip()]

                    # Parse year
                    year_str = row.get('PY', '') or row.get('Publication Year', '')
                    year = int(year_str) if year_str.isdigit() else 0

                    # Parse keywords
                    keywords_str = (row.get('DE', '') or row.get('Author Keywords', '') or
                                   row.get('ID', '') or row.get('Keywords Plus', ''))
                    keywords = [k.strip() for k in keywords_str.split(';') if k.strip()]

                    paper = Paper(
                        paper_id=f"wos_{row.get('UT', '') or row.get('Accession Number', '')}",
                        title=row.get('TI', '') or row.get('Article Title', ''),
                        authors=authors,
                        year=year,
                        abstract=row.get('AB', '') or row.get('Abstract', ''),
                        doi=row.get('DI', '') or row.get('DOI', ''),
                        source='web_of_science',
                        venue=row.get('SO', '') or row.get('Source Title', ''),
                        citation_count=int(row.get('TC', 0) or row.get('Times Cited', 0) or 0),
                        keywords=keywords,
                    )

                    if paper.title:
                        papers.append(paper)

                except Exception as e:
                    print(f"Warning: Failed to parse WoS row: {e}")
                    continue

        return papers


class PubMedImporter:
    """Import papers from PubMed/MEDLINE export."""

    def import_csv(self, filepath: Path) -> List[Paper]:
        """Import PubMed CSV export."""
        papers = []

        with open(filepath, 'r', encoding='utf-8-sig', errors='ignore') as f:
            reader = csv.DictReader(f)

            for row in reader:
                try:
                    # Parse authors
                    authors_str = row.get('Authors', '') or row.get('Author', '')
                    authors = [a.strip() for a in authors_str.split(',') if a.strip()]

                    # Parse year from publication date
                    pub_date = row.get('Publication Year', '') or row.get('Create Date', '')
                    year_match = re.search(r'\d{4}', pub_date)
                    year = int(year_match.group()) if year_match else 0

                    paper = Paper(
                        paper_id=f"pubmed_{row.get('PMID', '')}",
                        title=row.get('Title', ''),
                        authors=authors,
                        year=year,
                        abstract=row.get('Abstract', ''),
                        doi=row.get('DOI', ''),
                        source='pubmed',
                        venue=row.get('Journal/Book', '') or row.get('Source', ''),
                    )

                    if paper.title:
                        papers.append(paper)

                except Exception as e:
                    print(f"Warning: Failed to parse PubMed row: {e}")
                    continue

        return papers

    def import_xml(self, filepath: Path) -> List[Paper]:
        """Import PubMed XML export."""
        papers = []

        try:
            tree = ET.parse(filepath)
            root = tree.getroot()

            for article in root.findall('.//PubmedArticle'):
                try:
                    # Extract PMID
                    pmid_elem = article.find('.//PMID')
                    pmid = pmid_elem.text if pmid_elem is not None else ''

                    # Extract title
                    title_elem = article.find('.//ArticleTitle')
                    title = title_elem.text if title_elem is not None else ''

                    # Extract abstract
                    abstract_parts = []
                    for abs_text in article.findall('.//AbstractText'):
                        if abs_text.text:
                            abstract_parts.append(abs_text.text)
                    abstract = ' '.join(abstract_parts)

                    # Extract authors
                    authors = []
                    for author in article.findall('.//Author'):
                        lastname = author.find('LastName')
                        forename = author.find('ForeName')
                        if lastname is not None and forename is not None:
                            authors.append(f"{lastname.text}, {forename.text}")

                    # Extract year
                    year_elem = article.find('.//PubDate/Year')
                    year = int(year_elem.text) if year_elem is not None else 0

                    # Extract journal
                    journal_elem = article.find('.//Journal/Title')
                    journal = journal_elem.text if journal_elem is not None else ''

                    # Extract DOI
                    doi = None
                    for id_elem in article.findall('.//ArticleId'):
                        if id_elem.get('IdType') == 'doi':
                            doi = id_elem.text
                            break

                    paper = Paper(
                        paper_id=f"pubmed_{pmid}",
                        title=title,
                        authors=authors,
                        year=year,
                        abstract=abstract,
                        doi=doi,
                        source='pubmed',
                        venue=journal,
                    )

                    if paper.title:
                        papers.append(paper)

                except Exception as e:
                    print(f"Warning: Failed to parse PubMed article: {e}")
                    continue

        except Exception as e:
            print(f"Error parsing PubMed XML: {e}")

        return papers


class ERICImporter:
    """Import papers from ERIC database export."""

    def import_csv(self, filepath: Path) -> List[Paper]:
        """Import ERIC CSV export."""
        papers = []

        with open(filepath, 'r', encoding='utf-8-sig', errors='ignore') as f:
            reader = csv.DictReader(f)

            for row in reader:
                try:
                    # Parse authors
                    authors_str = row.get('Author', '') or row.get('Authors', '')
                    authors = [a.strip() for a in authors_str.split(';') if a.strip()]

                    # Parse year
                    year_str = row.get('Publication Date', '') or row.get('Year', '')
                    year_match = re.search(r'\d{4}', year_str)
                    year = int(year_match.group()) if year_match else 0

                    # Parse descriptors as keywords
                    keywords_str = row.get('Descriptors', '') or row.get('Keywords', '')
                    keywords = [k.strip() for k in keywords_str.split(',') if k.strip()]

                    paper = Paper(
                        paper_id=f"eric_{row.get('ERIC Number', '') or row.get('ERIC_ID', '')}",
                        title=row.get('Title', ''),
                        authors=authors,
                        year=year,
                        abstract=row.get('Abstract', ''),
                        doi=row.get('DOI', ''),
                        source='eric',
                        venue=row.get('Source', '') or row.get('Journal', ''),
                        keywords=keywords,
                    )

                    if paper.title:
                        papers.append(paper)

                except Exception as e:
                    print(f"Warning: Failed to parse ERIC row: {e}")
                    continue

        return papers


class InstitutionalImporter:
    """Main importer class that handles all institutional database formats."""

    def __init__(self):
        self.scopus = ScopusImporter()
        self.wos = WebOfScienceImporter()
        self.pubmed = PubMedImporter()
        self.eric = ERICImporter()
        self.ris_parser = RISParser()

    def detect_source(self, filepath: Path) -> str:
        """Detect the source database from filename or content."""
        filename = filepath.name.lower()

        if 'scopus' in filename:
            return 'scopus'
        elif 'wos' in filename or 'webofscience' in filename or 'web_of_science' in filename:
            return 'web_of_science'
        elif 'pubmed' in filename or 'medline' in filename:
            return 'pubmed'
        elif 'eric' in filename:
            return 'eric'

        # Try to detect from content
        try:
            with open(filepath, 'r', encoding='utf-8-sig', errors='ignore') as f:
                content = f.read(1000)

                if 'EID' in content or 'Scopus' in content:
                    return 'scopus'
                elif 'UT' in content or 'WOS' in content:
                    return 'web_of_science'
                elif 'PMID' in content or 'PubMed' in content:
                    return 'pubmed'
                elif 'ERIC Number' in content or 'ERIC_ID' in content:
                    return 'eric'
        except:
            pass

        return 'unknown'

    def import_file(self, filepath: Path) -> List[Paper]:
        """Import a single file, auto-detecting format and source."""
        filepath = Path(filepath)

        if not filepath.exists():
            print(f"Warning: File not found: {filepath}")
            return []

        source = self.detect_source(filepath)
        suffix = filepath.suffix.lower()

        papers = []

        if suffix == '.ris':
            # Parse RIS and convert to Paper objects
            ris_papers = self.ris_parser.parse(filepath)
            for rp in ris_papers:
                paper = Paper(
                    paper_id=f"{source}_{hash(rp.get('title', ''))}",
                    title=rp.get('title', ''),
                    authors=rp.get('authors', []),
                    year=rp.get('year', 0),
                    abstract=rp.get('abstract', ''),
                    doi=rp.get('doi'),
                    source=source,
                    venue=rp.get('venue'),
                    keywords=rp.get('keywords', []),
                )
                if paper.title:
                    papers.append(paper)

        elif suffix == '.csv':
            if source == 'scopus':
                papers = self.scopus.import_csv(filepath)
            elif source == 'web_of_science':
                papers = self.wos.import_csv(filepath)
            elif source == 'pubmed':
                papers = self.pubmed.import_csv(filepath)
            elif source == 'eric':
                papers = self.eric.import_csv(filepath)
            else:
                # Try each importer
                for importer, name in [
                    (self.scopus, 'scopus'),
                    (self.wos, 'web_of_science'),
                    (self.pubmed, 'pubmed'),
                    (self.eric, 'eric'),
                ]:
                    try:
                        papers = importer.import_csv(filepath)
                        if papers:
                            print(f"  Detected as {name} format")
                            break
                    except:
                        continue

        elif suffix == '.xml':
            if source == 'pubmed' or 'pubmed' in filepath.name.lower():
                papers = self.pubmed.import_xml(filepath)

        print(f"  Imported {len(papers)} papers from {filepath.name}")
        return papers

    def import_directory(self, directory: Path) -> List[Paper]:
        """Import all supported files from a directory."""
        directory = Path(directory)
        all_papers = []

        supported_extensions = ['.csv', '.ris', '.xml']

        for ext in supported_extensions:
            for filepath in directory.glob(f'*{ext}'):
                papers = self.import_file(filepath)
                all_papers.extend(papers)

        print(f"\nTotal imported from institutional databases: {len(all_papers)} papers")
        return all_papers

    def import_all(self, directory: str) -> List[Paper]:
        """Import all files and return as list of Paper objects."""
        return self.import_directory(Path(directory))

    def save_to_json(self, papers: List[Paper], output_path: Path):
        """Save papers to JSON format compatible with search pipeline."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "source": "institutional_import",
            "timestamp": datetime.now().isoformat(),
            "total_papers": len(papers),
            "papers": [asdict(p) for p in papers]
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"Saved {len(papers)} papers to {output_path}")


def main():
    """CLI interface for institutional import."""
    import argparse

    parser = argparse.ArgumentParser(description='Import papers from institutional databases')
    parser.add_argument('input_dir', help='Directory containing export files')
    parser.add_argument('-o', '--output', default='data/01_search_results/institutional_papers.json',
                        help='Output JSON file path')

    args = parser.parse_args()

    importer = InstitutionalImporter()
    papers = importer.import_all(args.input_dir)

    if papers:
        importer.save_to_json(papers, Path(args.output))


if __name__ == '__main__':
    main()
