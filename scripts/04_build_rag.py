#!/usr/bin/env python3
"""
Stage 4: RAG Index Builder for AI-Ethics-HR Systematic Review
Builds vector database from PDFs for RAG-enabled coding.
"""

import os
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict, field
import hashlib

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """A chunk of document text with metadata."""
    chunk_id: str
    paper_id: str
    content: str
    metadata: Dict[str, Any]
    chunk_index: int
    total_chunks: int


@dataclass
class RAGBuildStats:
    """Statistics from RAG building process."""
    total_papers: int = 0
    papers_processed: int = 0
    papers_failed: int = 0
    total_chunks: int = 0
    total_tokens_estimated: int = 0
    avg_chunks_per_paper: float = 0.0


class PDFProcessor:
    """Process PDFs and extract text."""

    def __init__(self):
        try:
            import fitz  # PyMuPDF
            self.fitz = fitz
            self.available = True
        except ImportError:
            logger.warning("PyMuPDF not installed. Install with: pip install pymupdf")
            self.available = False

    def extract_text(self, pdf_path: str) -> Optional[str]:
        """Extract text from a PDF file."""
        if not self.available:
            return None

        try:
            doc = self.fitz.open(pdf_path)
            text_parts = []

            for page in doc:
                text = page.get_text()
                if text:
                    text_parts.append(text)

            doc.close()

            full_text = "\n\n".join(text_parts)

            # Basic cleaning
            full_text = self._clean_text(full_text)

            return full_text if full_text.strip() else None

        except Exception as e:
            logger.error(f"Failed to extract text from {pdf_path}: {e}")
            return None

    def _clean_text(self, text: str) -> str:
        """Clean extracted text."""
        import re

        # Remove excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)

        # Remove page numbers (common patterns)
        text = re.sub(r'\n\d+\n', '\n', text)

        # Remove hyphenation at line breaks
        text = re.sub(r'-\n', '', text)

        return text.strip()


class TextChunker:
    """Chunk text for embedding."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: List[str] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " "]

    def chunk_text(
        self,
        text: str,
        paper_id: str,
        metadata: Dict[str, Any] = None
    ) -> List[DocumentChunk]:
        """Split text into overlapping chunks."""
        if not text:
            return []

        # Use recursive character splitting
        chunks_text = self._recursive_split(text)

        # Create DocumentChunk objects
        chunks = []
        for i, chunk_text in enumerate(chunks_text):
            chunk_id = f"{paper_id}_chunk_{i:04d}"

            chunk = DocumentChunk(
                chunk_id=chunk_id,
                paper_id=paper_id,
                content=chunk_text,
                metadata={
                    **(metadata or {}),
                    'chunk_index': i,
                    'total_chunks': len(chunks_text),
                    'char_count': len(chunk_text)
                },
                chunk_index=i,
                total_chunks=len(chunks_text)
            )
            chunks.append(chunk)

        return chunks

    def _recursive_split(self, text: str) -> List[str]:
        """Recursively split text using separators."""
        chunks = []

        # Try each separator in order
        for separator in self.separators:
            if separator in text:
                splits = text.split(separator)
                current_chunk = ""

                for split in splits:
                    # Check if adding this split would exceed chunk size
                    test_chunk = current_chunk + separator + split if current_chunk else split

                    if len(test_chunk) <= self.chunk_size:
                        current_chunk = test_chunk
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                        current_chunk = split

                if current_chunk:
                    chunks.append(current_chunk)

                if chunks:
                    break

        # If no separator worked, split by character count
        if not chunks:
            for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
                chunks.append(text[i:i + self.chunk_size])

        # Add overlap between chunks
        overlapped_chunks = []
        for i, chunk in enumerate(chunks):
            if i > 0 and self.chunk_overlap > 0:
                # Get overlap from previous chunk
                prev_chunk = chunks[i - 1]
                overlap_text = prev_chunk[-self.chunk_overlap:]
                chunk = overlap_text + chunk

            # Trim to max size
            if len(chunk) > self.chunk_size:
                chunk = chunk[:self.chunk_size]

            overlapped_chunks.append(chunk)

        return overlapped_chunks


class EmbeddingGenerator:
    """Generate embeddings for text chunks."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.model_name = model_name
            self.available = True
            logger.info(f"Loaded embedding model: {model_name}")
        except ImportError:
            logger.warning("sentence-transformers not installed. Install with: pip install sentence-transformers")
            self.available = False
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self.available = False

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        if not self.available:
            return []

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        return embeddings.tolist()

    def embed_single(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        if not self.available:
            return []

        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()


class ChromaDBBuilder:
    """Build and manage ChromaDB vector store."""

    def __init__(
        self,
        persist_directory: str,
        collection_name: str = "ai_ethics_hr",
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        try:
            import chromadb
            from chromadb.config import Settings

            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
            self.collection_name = collection_name
            self.embedding_generator = EmbeddingGenerator(embedding_model)
            self.available = True

            logger.info(f"ChromaDB initialized at {persist_directory}")

        except ImportError:
            logger.warning("chromadb not installed. Install with: pip install chromadb")
            self.available = False
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            self.available = False

    def create_collection(self, reset: bool = False):
        """Create or get the collection."""
        if not self.available:
            return None

        if reset:
            try:
                self.client.delete_collection(self.collection_name)
                logger.info(f"Deleted existing collection: {self.collection_name}")
            except Exception:
                pass

        collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "AI Ethics in HR systematic review papers"}
        )

        return collection

    def add_chunks(
        self,
        collection,
        chunks: List[DocumentChunk],
        batch_size: int = 100
    ):
        """Add document chunks to collection."""
        if not self.available or not chunks:
            return

        # Process in batches
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]

            ids = [c.chunk_id for c in batch]
            documents = [c.content for c in batch]
            metadatas = [c.metadata for c in batch]

            # Generate embeddings
            embeddings = self.embedding_generator.embed_texts(documents)

            if embeddings:
                collection.add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas,
                    embeddings=embeddings
                )

            logger.info(f"Added batch {i // batch_size + 1}: {len(batch)} chunks")

    def query(
        self,
        collection,
        query_text: str,
        n_results: int = 5,
        filter_metadata: Dict = None
    ) -> Dict:
        """Query the collection."""
        if not self.available:
            return {}

        query_embedding = self.embedding_generator.embed_single(query_text)

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=filter_metadata
        )

        return results


class RAGPipeline:
    """Orchestrates RAG index building."""

    def __init__(
        self,
        persist_directory: str,
        collection_name: str = "ai_ethics_hr",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        self.pdf_processor = PDFProcessor()
        self.chunker = TextChunker(chunk_size, chunk_overlap)
        self.db_builder = ChromaDBBuilder(
            persist_directory, collection_name, embedding_model
        )
        self.stats = RAGBuildStats()

    def build_from_papers(
        self,
        papers: List[Dict],
        pdf_dir: str,
        reset_collection: bool = False
    ) -> Tuple[Any, RAGBuildStats]:
        """
        Build RAG index from papers and their PDFs.

        Args:
            papers: List of paper metadata dictionaries
            pdf_dir: Directory containing PDF files
            reset_collection: Whether to reset existing collection

        Returns:
            Tuple of (collection, statistics)
        """
        if not self.db_builder.available:
            logger.error("ChromaDB not available")
            return None, self.stats

        # Create collection
        collection = self.db_builder.create_collection(reset=reset_collection)

        self.stats.total_papers = len(papers)
        all_chunks = []

        pdf_path = Path(pdf_dir)

        for i, paper in enumerate(papers):
            paper_id = paper.get('source_id', f'paper_{i}')
            title = paper.get('title', 'Unknown')

            # Try to find PDF
            pdf_file = self._find_pdf(paper, pdf_path)

            if pdf_file:
                # Extract text from PDF
                text = self.pdf_processor.extract_text(str(pdf_file))

                if text:
                    # Create chunks
                    metadata = {
                        'paper_id': paper_id,
                        'title': title,
                        'authors': ', '.join(paper.get('authors', [])),
                        'year': paper.get('year'),
                        'source': paper.get('source', '')
                    }

                    chunks = self.chunker.chunk_text(text, paper_id, metadata)
                    all_chunks.extend(chunks)

                    self.stats.papers_processed += 1
                    logger.info(f"Processed {paper_id}: {len(chunks)} chunks")
                else:
                    self.stats.papers_failed += 1
                    logger.warning(f"Failed to extract text from {paper_id}")
            else:
                # Use abstract if no PDF
                abstract = paper.get('abstract', '')
                if abstract:
                    metadata = {
                        'paper_id': paper_id,
                        'title': title,
                        'authors': ', '.join(paper.get('authors', [])),
                        'year': paper.get('year'),
                        'source': paper.get('source', ''),
                        'is_abstract_only': True
                    }

                    chunks = self.chunker.chunk_text(abstract, paper_id, metadata)
                    all_chunks.extend(chunks)

                    self.stats.papers_processed += 1
                    logger.info(f"Processed {paper_id} (abstract only): {len(chunks)} chunks")
                else:
                    self.stats.papers_failed += 1
                    logger.warning(f"No PDF or abstract for {paper_id}")

            if (i + 1) % 10 == 0:
                logger.info(f"Progress: {i + 1}/{len(papers)} papers")

        # Add all chunks to collection
        if all_chunks:
            logger.info(f"Adding {len(all_chunks)} chunks to vector store...")
            self.db_builder.add_chunks(collection, all_chunks)

        # Update statistics
        self.stats.total_chunks = len(all_chunks)
        self.stats.total_tokens_estimated = sum(
            len(c.content.split()) for c in all_chunks
        )
        if self.stats.papers_processed > 0:
            self.stats.avg_chunks_per_paper = (
                self.stats.total_chunks / self.stats.papers_processed
            )

        return collection, self.stats

    def _find_pdf(self, paper: Dict, pdf_dir: Path) -> Optional[Path]:
        """Try to find PDF file for a paper."""
        paper_id = paper.get('source_id', '')
        doi = paper.get('doi', '')

        # Try various naming conventions
        possible_names = [
            f"{paper_id}.pdf",
            f"{paper_id.replace('/', '_')}.pdf",
            f"{doi.replace('/', '_')}.pdf" if doi else None,
            f"{hashlib.md5(paper.get('title', '').encode()).hexdigest()[:16]}.pdf"
        ]

        for name in possible_names:
            if name:
                pdf_file = pdf_dir / name
                if pdf_file.exists():
                    return pdf_file

        return None

    def validate_retrieval(
        self,
        collection,
        test_queries: List[str] = None
    ) -> Dict:
        """Validate retrieval quality with test queries."""
        if not test_queries:
            test_queries = [
                "algorithmic bias in hiring",
                "AI transparency in performance management",
                "employee privacy surveillance",
                "fairness in recruitment AI",
                "accountability for AI decisions"
            ]

        validation_results = []

        for query in test_queries:
            results = self.db_builder.query(collection, query, n_results=3)

            if results and results.get('documents'):
                docs = results['documents'][0]
                distances = results.get('distances', [[]])[0]

                validation_results.append({
                    'query': query,
                    'num_results': len(docs),
                    'avg_distance': sum(distances) / len(distances) if distances else None,
                    'sample_content': docs[0][:200] if docs else None
                })

        return {
            'validation_date': datetime.now().isoformat(),
            'num_queries_tested': len(test_queries),
            'results': validation_results
        }


def load_papers(input_path: str) -> List[Dict]:
    """Load papers from JSON file."""
    with open(input_path) as f:
        return json.load(f)


def save_build_results(
    stats: RAGBuildStats,
    validation: Dict,
    output_dir: str
):
    """Save RAG building results and statistics."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save statistics
    stats_file = output_path / "rag_build_stats.json"
    with open(stats_file, "w") as f:
        json.dump(asdict(stats), f, indent=2)

    # Save validation results
    validation_file = output_path / "rag_validation.json"
    with open(validation_file, "w") as f:
        json.dump(validation, f, indent=2)

    logger.info(f"Build results saved to {output_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Build RAG index for AI Ethics in HR systematic review"
    )
    parser.add_argument(
        "--input", "-i",
        default="./data/03_screened/screened_included.json",
        help="Input file with screened papers"
    )
    parser.add_argument(
        "--pdf-dir",
        default="./data/04_full_text",
        help="Directory containing PDF files"
    )
    parser.add_argument(
        "--output", "-o",
        default="./rag",
        help="Output directory for RAG index"
    )
    parser.add_argument(
        "--collection-name",
        default="ai_ethics_hr",
        help="Name of the ChromaDB collection"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Chunk size in characters"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Overlap between chunks"
    )
    parser.add_argument(
        "--embedding-model",
        default="all-MiniLM-L6-v2",
        help="Sentence transformer model for embeddings"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset existing collection"
    )

    args = parser.parse_args()

    # Load papers
    logger.info(f"Loading papers from {args.input}")
    papers = load_papers(args.input)
    logger.info(f"Loaded {len(papers)} papers")

    # Ensure PDF directory exists
    pdf_dir = Path(args.pdf_dir)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    # Initialize pipeline
    persist_dir = str(Path(args.output) / "chroma_db")
    pipeline = RAGPipeline(
        persist_directory=persist_dir,
        collection_name=args.collection_name,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embedding_model=args.embedding_model
    )

    # Build index
    collection, stats = pipeline.build_from_papers(
        papers, str(pdf_dir), reset_collection=args.reset
    )

    # Validate retrieval
    validation = {}
    if collection:
        validation = pipeline.validate_retrieval(collection)

    # Save results
    configs_dir = Path(args.output) / "configs"
    save_build_results(stats, validation, str(configs_dir))

    # Print summary
    print("\n" + "="*60)
    print("STAGE 4: RAG INDEX BUILDING COMPLETE")
    print("="*60)
    print(f"Total papers:           {stats.total_papers}")
    print(f"Papers processed:       {stats.papers_processed}")
    print(f"Papers failed:          {stats.papers_failed}")
    print(f"Total chunks:           {stats.total_chunks}")
    print(f"Estimated tokens:       {stats.total_tokens_estimated:,}")
    print(f"Avg chunks per paper:   {stats.avg_chunks_per_paper:.1f}")
    print("-"*60)
    print(f"Collection:             {args.collection_name}")
    print(f"Persist directory:      {persist_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
