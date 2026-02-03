#!/usr/bin/env python3
"""
Concept Co-occurrence Network Analysis

Analyzes co-occurrence patterns of ethical concepts across papers:
- Builds co-occurrence matrices at multiple levels
- Creates network graphs with centrality analysis
- Detects concept communities
- Generates publication-ready visualizations

Version: 1.1.0
"""

import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Constants
ETHICS_PRINCIPLES = [
    'fairness_bias', 'transparency', 'accountability',
    'privacy', 'autonomy', 'wellbeing'
]

ETHICS_SUBTYPES = {
    'fairness_bias': [
        'algorithmic_bias', 'disparate_impact', 'protected_characteristics',
        'historical_bias', 'proxy_discrimination'
    ],
    'transparency': [
        'explainability', 'black_box', 'interpretability', 'communication_to_employees'
    ],
    'accountability': [
        'human_oversight', 'liability', 'responsibility', 'auditability'
    ],
    'privacy': [
        'data_collection', 'surveillance', 'consent', 'gdpr', 'data_minimization'
    ],
    'autonomy': [
        'human_in_the_loop', 'deskilling', 'agency', 'decision_authority'
    ],
    'wellbeing': [
        'job_quality', 'psychological_safety', 'work_intensification', 'employee_experience'
    ]
}


@dataclass
class ConceptOccurrence:
    """Represents concept occurrences in a single paper."""
    paper_id: str
    principles: Set[str] = field(default_factory=set)
    subtypes: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    solutions: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    stance: Optional[str] = None
    period: Optional[str] = None


class ConceptExtractor:
    """Extracts concepts from coded papers."""

    def __init__(self, coded_data: List[Dict]):
        """
        Initialize with coded paper data.

        Args:
            coded_data: List of coded paper dictionaries
        """
        self.coded_data = coded_data
        self.occurrences: List[ConceptOccurrence] = []

    def extract_all(self) -> List[ConceptOccurrence]:
        """Extract concepts from all papers."""
        for paper in self.coded_data:
            occurrence = self._extract_paper(paper)
            if occurrence:
                self.occurrences.append(occurrence)

        logger.info(f"Extracted concepts from {len(self.occurrences)} papers")
        return self.occurrences

    def _extract_paper(self, paper: Dict) -> Optional[ConceptOccurrence]:
        """Extract concepts from a single paper."""
        paper_id = paper.get('paper_id', 'unknown')

        occurrence = ConceptOccurrence(paper_id=paper_id)

        # Extract ethical issues
        ethical_issues = paper.get('ethical_issues', {})
        for principle in ETHICS_PRINCIPLES:
            principle_data = ethical_issues.get(principle, {})

            # Handle both direct dict and FieldConfidence format
            if isinstance(principle_data, dict):
                value = principle_data.get('value', principle_data)
            else:
                continue

            if isinstance(value, dict) and value.get('mentioned', False):
                occurrence.principles.add(principle)

                # Extract subtypes
                types = value.get('types', [])
                if types:
                    occurrence.subtypes[principle] = set(types)

        # Extract solutions (v1.1.0)
        solution_data = paper.get('solution_taxonomy', {})
        if isinstance(solution_data, dict):
            value = solution_data.get('value', solution_data)
            if isinstance(value, dict) and value.get('solutions_proposed', False):
                occurrence.solutions['technical'] = set(value.get('technical_solutions', []))
                occurrence.solutions['organizational'] = set(value.get('organizational_solutions', []))
                occurrence.solutions['regulatory'] = set(value.get('regulatory_solutions', []))

        # Extract stance (v1.1.0)
        stance_data = paper.get('stance_classification', {})
        if isinstance(stance_data, dict):
            value = stance_data.get('value', stance_data)
            if isinstance(value, dict):
                occurrence.stance = value.get('overall_tone')

        # Extract temporal period (v1.1.0)
        temporal = paper.get('temporal_metadata', {})
        if isinstance(temporal, dict):
            occurrence.period = temporal.get('research_period')

        return occurrence if occurrence.principles else None


class CooccurrenceAnalyzer:
    """Builds and analyzes co-occurrence matrices and networks."""

    def __init__(self, occurrences: List[ConceptOccurrence]):
        """
        Initialize analyzer with concept occurrences.

        Args:
            occurrences: List of ConceptOccurrence objects
        """
        self.occurrences = occurrences
        self.principle_matrix: Optional[pd.DataFrame] = None
        self.subtype_matrix: Optional[pd.DataFrame] = None
        self.network_stats: Dict = {}

    def build_principle_matrix(self) -> pd.DataFrame:
        """Build 6x6 co-occurrence matrix for ethical principles."""
        n = len(ETHICS_PRINCIPLES)
        matrix = np.zeros((n, n), dtype=int)

        for occ in self.occurrences:
            principles_list = list(occ.principles)
            for p in principles_list:
                i = ETHICS_PRINCIPLES.index(p)
                matrix[i, i] += 1  # Diagonal = occurrence count

            # Co-occurrences
            for p1, p2 in combinations(principles_list, 2):
                i, j = ETHICS_PRINCIPLES.index(p1), ETHICS_PRINCIPLES.index(p2)
                matrix[i, j] += 1
                matrix[j, i] += 1

        self.principle_matrix = pd.DataFrame(
            matrix,
            index=ETHICS_PRINCIPLES,
            columns=ETHICS_PRINCIPLES
        )

        logger.info("Built principle co-occurrence matrix")
        return self.principle_matrix

    def build_subtype_matrix(self) -> pd.DataFrame:
        """Build co-occurrence matrix for all subtypes across all principles."""
        all_subtypes = []
        for subtypes in ETHICS_SUBTYPES.values():
            all_subtypes.extend(subtypes)

        n = len(all_subtypes)
        matrix = np.zeros((n, n), dtype=int)

        for occ in self.occurrences:
            # Flatten all subtypes from this paper
            paper_subtypes = []
            for principle, types in occ.subtypes.items():
                paper_subtypes.extend(types)

            for st in paper_subtypes:
                if st in all_subtypes:
                    i = all_subtypes.index(st)
                    matrix[i, i] += 1

            for st1, st2 in combinations(paper_subtypes, 2):
                if st1 in all_subtypes and st2 in all_subtypes:
                    i, j = all_subtypes.index(st1), all_subtypes.index(st2)
                    matrix[i, j] += 1
                    matrix[j, i] += 1

        self.subtype_matrix = pd.DataFrame(
            matrix,
            index=all_subtypes,
            columns=all_subtypes
        )

        logger.info("Built subtype co-occurrence matrix")
        return self.subtype_matrix

    def analyze_network(self, level: str = 'principle') -> Dict:
        """
        Analyze network properties using networkx.

        Args:
            level: 'principle' or 'subtype'

        Returns:
            Dictionary of network statistics
        """
        try:
            import networkx as nx
        except ImportError:
            logger.error("networkx not installed. Install with: pip install networkx")
            return {}

        matrix = self.principle_matrix if level == 'principle' else self.subtype_matrix
        if matrix is None:
            logger.error(f"No {level} matrix available. Build matrix first.")
            return {}

        # Create graph from adjacency matrix (excluding diagonal)
        adj_matrix = matrix.values.copy()
        np.fill_diagonal(adj_matrix, 0)

        G = nx.from_numpy_array(adj_matrix)

        # Relabel nodes
        labels = matrix.index.tolist()
        mapping = {i: labels[i] for i in range(len(labels))}
        G = nx.relabel_nodes(G, mapping)

        # Calculate statistics
        stats = {
            'level': level,
            'n_nodes': G.number_of_nodes(),
            'n_edges': G.number_of_edges(),
            'density': nx.density(G),
            'avg_clustering': nx.average_clustering(G, weight='weight'),
        }

        # Centrality measures
        stats['degree_centrality'] = dict(nx.degree_centrality(G))
        stats['betweenness_centrality'] = dict(nx.betweenness_centrality(G, weight='weight'))
        stats['eigenvector_centrality'] = dict(nx.eigenvector_centrality_numpy(G, weight='weight'))

        # Top concepts by each centrality measure
        stats['top_by_degree'] = sorted(
            stats['degree_centrality'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        stats['top_by_betweenness'] = sorted(
            stats['betweenness_centrality'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        # Community detection
        try:
            from networkx.algorithms.community import greedy_modularity_communities
            communities = list(greedy_modularity_communities(G, weight='weight'))
            stats['n_communities'] = len(communities)
            stats['communities'] = [list(c) for c in communities]
        except Exception as e:
            logger.warning(f"Community detection failed: {e}")
            stats['n_communities'] = 0
            stats['communities'] = []

        self.network_stats[level] = stats
        logger.info(f"Network analysis complete for {level} level")
        return stats


class CooccurrenceVisualizer:
    """Generates visualizations for co-occurrence analysis."""

    def __init__(
        self,
        analyzer: CooccurrenceAnalyzer,
        output_dir: str = './output/analysis'
    ):
        """
        Initialize visualizer.

        Args:
            analyzer: CooccurrenceAnalyzer with computed matrices
            output_dir: Directory for output files
        """
        self.analyzer = analyzer
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_heatmap(
        self,
        level: str = 'principle',
        figsize: Tuple[int, int] = (10, 8),
        cmap: str = 'YlOrRd'
    ) -> str:
        """
        Generate heatmap visualization of co-occurrence matrix.

        Args:
            level: 'principle' or 'subtype'
            figsize: Figure size tuple
            cmap: Colormap name

        Returns:
            Path to saved figure
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        matrix = (self.analyzer.principle_matrix if level == 'principle'
                  else self.analyzer.subtype_matrix)

        if matrix is None:
            raise ValueError(f"No {level} matrix available")

        fig, ax = plt.subplots(figsize=figsize)

        # Create mask for diagonal (optional)
        mask = None  # np.eye(len(matrix), dtype=bool)

        sns.heatmap(
            matrix,
            ax=ax,
            cmap=cmap,
            annot=True,
            fmt='d',
            mask=mask,
            linewidths=0.5,
            cbar_kws={'label': 'Co-occurrence Count'}
        )

        ax.set_title(f'AI Ethics Concept Co-occurrence ({level.title()} Level)')
        ax.set_xlabel('Ethical Concepts')
        ax.set_ylabel('Ethical Concepts')

        # Rotate labels for readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

        plt.tight_layout()

        output_path = self.output_dir / f'cooccurrence_heatmap_{level}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved heatmap to {output_path}")
        return str(output_path)

    def plot_network(
        self,
        level: str = 'principle',
        figsize: Tuple[int, int] = (12, 10),
        min_edge_weight: int = 1
    ) -> str:
        """
        Generate network graph visualization.

        Args:
            level: 'principle' or 'subtype'
            figsize: Figure size tuple
            min_edge_weight: Minimum edge weight to display

        Returns:
            Path to saved figure
        """
        import matplotlib.pyplot as plt
        import networkx as nx

        matrix = (self.analyzer.principle_matrix if level == 'principle'
                  else self.analyzer.subtype_matrix)

        if matrix is None:
            raise ValueError(f"No {level} matrix available")

        # Create graph
        adj_matrix = matrix.values.copy()
        np.fill_diagonal(adj_matrix, 0)

        G = nx.from_numpy_array(adj_matrix)
        labels = matrix.index.tolist()
        mapping = {i: labels[i] for i in range(len(labels))}
        G = nx.relabel_nodes(G, mapping)

        # Remove weak edges
        edges_to_remove = [
            (u, v) for u, v, d in G.edges(data=True)
            if d.get('weight', 0) < min_edge_weight
        ]
        G.remove_edges_from(edges_to_remove)

        fig, ax = plt.subplots(figsize=figsize)

        # Layout
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

        # Node sizes based on occurrence count (diagonal values)
        node_sizes = [matrix.loc[node, node] * 100 + 300 for node in G.nodes()]

        # Edge widths based on weights
        edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
        max_weight = max(edge_weights) if edge_weights else 1
        edge_widths = [2 + (w / max_weight) * 4 for w in edge_weights]

        # Draw
        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            node_size=node_sizes,
            node_color='lightblue',
            alpha=0.8,
            edgecolors='darkblue',
            linewidths=2
        )

        nx.draw_networkx_edges(
            G, pos, ax=ax,
            width=edge_widths,
            alpha=0.6,
            edge_color='gray'
        )

        # Labels with better formatting
        label_mapping = {k: k.replace('_', '\n') for k in G.nodes()}
        nx.draw_networkx_labels(
            G, pos, ax=ax,
            labels=label_mapping,
            font_size=9,
            font_weight='bold'
        )

        ax.set_title(f'AI Ethics Concept Network ({level.title()} Level)')
        ax.axis('off')

        plt.tight_layout()

        output_path = self.output_dir / f'network_graph_{level}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved network graph to {output_path}")
        return str(output_path)

    def create_interactive_network(
        self,
        level: str = 'principle'
    ) -> str:
        """
        Create interactive network visualization using Plotly.

        Args:
            level: 'principle' or 'subtype'

        Returns:
            Path to saved HTML file
        """
        try:
            import plotly.graph_objects as go
            import networkx as nx
        except ImportError:
            logger.error("plotly not installed. Install with: pip install plotly")
            return ""

        matrix = (self.analyzer.principle_matrix if level == 'principle'
                  else self.analyzer.subtype_matrix)

        if matrix is None:
            raise ValueError(f"No {level} matrix available")

        # Create graph
        adj_matrix = matrix.values.copy()
        np.fill_diagonal(adj_matrix, 0)

        G = nx.from_numpy_array(adj_matrix)
        labels = matrix.index.tolist()
        mapping = {i: labels[i] for i in range(len(labels))}
        G = nx.relabel_nodes(G, mapping)

        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

        # Edge traces
        edge_x = []
        edge_y = []
        edge_weights = []

        for u, v, d in G.edges(data=True):
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_weights.append(d.get('weight', 1))

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )

        # Node traces
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        node_text = []
        node_sizes = []

        for node in G.nodes():
            count = matrix.loc[node, node]
            degree = sum(G[node][neighbor].get('weight', 0) for neighbor in G.neighbors(node))
            node_text.append(f'{node}<br>Occurrences: {count}<br>Connections: {degree}')
            node_sizes.append(count + 10)

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=list(G.nodes()),
            textposition='top center',
            hovertext=node_text,
            marker=dict(
                size=node_sizes,
                color='lightblue',
                line=dict(width=2, color='darkblue')
            )
        )

        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=f'AI Ethics Concept Network ({level.title()} Level)',
                showlegend=False,
                hovermode='closest',
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
        )

        output_path = self.output_dir / f'network_interactive_{level}.html'
        fig.write_html(str(output_path))

        logger.info(f"Saved interactive network to {output_path}")
        return str(output_path)

    def save_matrices(self) -> List[str]:
        """Save co-occurrence matrices to CSV files."""
        paths = []

        if self.analyzer.principle_matrix is not None:
            path = self.output_dir / 'cooccurrence_matrix_principle.csv'
            self.analyzer.principle_matrix.to_csv(path)
            paths.append(str(path))

        if self.analyzer.subtype_matrix is not None:
            path = self.output_dir / 'cooccurrence_matrix_subtype.csv'
            self.analyzer.subtype_matrix.to_csv(path)
            paths.append(str(path))

        logger.info(f"Saved matrices to {self.output_dir}")
        return paths

    def save_network_stats(self) -> str:
        """Save network statistics to JSON."""
        path = self.output_dir / 'network_stats.json'

        # Convert numpy types for JSON serialization
        stats = {}
        for level, level_stats in self.analyzer.network_stats.items():
            stats[level] = {}
            for k, v in level_stats.items():
                if isinstance(v, (np.int64, np.float64)):
                    stats[level][k] = float(v)
                elif isinstance(v, dict):
                    stats[level][k] = {
                        str(kk): float(vv) if isinstance(vv, (np.int64, np.float64)) else vv
                        for kk, vv in v.items()
                    }
                else:
                    stats[level][k] = v

        with open(path, 'w') as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Saved network stats to {path}")
        return str(path)


def load_coded_data(input_path: str) -> List[Dict]:
    """Load coded paper data from JSON file."""
    with open(input_path) as f:
        return json.load(f)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Concept Co-occurrence Network Analysis"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input JSON file with coded papers"
    )
    parser.add_argument(
        "--output", "-o",
        default="./output/analysis/cooccurrence",
        help="Output directory for results"
    )
    parser.add_argument(
        "--level",
        choices=['principle', 'subtype', 'both'],
        default='both',
        help="Analysis level"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Generate interactive visualizations"
    )

    args = parser.parse_args()

    # Load data
    logger.info(f"Loading coded data from {args.input}")
    coded_data = load_coded_data(args.input)

    # Extract concepts
    extractor = ConceptExtractor(coded_data)
    occurrences = extractor.extract_all()

    # Analyze
    analyzer = CooccurrenceAnalyzer(occurrences)

    levels = ['principle', 'subtype'] if args.level == 'both' else [args.level]

    for level in levels:
        if level == 'principle':
            analyzer.build_principle_matrix()
        else:
            analyzer.build_subtype_matrix()

        analyzer.analyze_network(level)

    # Visualize
    visualizer = CooccurrenceVisualizer(analyzer, args.output)

    for level in levels:
        visualizer.plot_heatmap(level)
        visualizer.plot_network(level)

        if args.interactive:
            visualizer.create_interactive_network(level)

    visualizer.save_matrices()
    visualizer.save_network_stats()

    print("\n" + "="*60)
    print("CO-OCCURRENCE ANALYSIS COMPLETE")
    print("="*60)
    print(f"Papers analyzed: {len(occurrences)}")
    print(f"Output saved to: {args.output}")
    print("="*60)


if __name__ == "__main__":
    main()
