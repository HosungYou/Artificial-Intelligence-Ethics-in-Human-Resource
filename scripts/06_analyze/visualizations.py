#!/usr/bin/env python3
"""
Publication-Ready Visualizations

Generates high-quality figures for academic publications:
- PRISMA flow diagram
- HR function distribution
- Ethics × HR heatmap
- Solution taxonomy treemap
- Summary figures

Version: 1.1.0
"""

import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Style configuration for publication
PUBLICATION_STYLE = {
    'figure.figsize': (8, 6),
    'figure.dpi': 300,
    'font.size': 11,
    'font.family': 'serif',
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
}

# Color schemes
ETHICS_COLORS = {
    'fairness_bias': '#e74c3c',
    'transparency': '#3498db',
    'accountability': '#2ecc71',
    'privacy': '#9b59b6',
    'autonomy': '#f39c12',
    'wellbeing': '#1abc9c'
}

HR_FUNCTION_COLORS = {
    'recruitment': '#264653',
    'selection': '#2a9d8f',
    'performance_management': '#e9c46a',
    'learning_development': '#f4a261',
    'people_analytics': '#e76f51',
    'employee_relations': '#8ecae6',
    'workforce_planning': '#219ebc',
    'compensation_benefits': '#023047',
    'multiple': '#a8dadc'
}


class PublicationVisualizer:
    """Generates publication-quality visualizations."""

    def __init__(
        self,
        coded_data: List[Dict],
        output_dir: str = './output/figures'
    ):
        """
        Initialize visualizer.

        Args:
            coded_data: List of coded paper dictionaries
            output_dir: Directory for output files
        """
        self.coded_data = coded_data
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Apply publication style
        self._setup_style()

    def _setup_style(self):
        """Configure matplotlib for publication quality."""
        import matplotlib.pyplot as plt
        plt.rcParams.update(PUBLICATION_STYLE)

    def plot_prisma_flow(
        self,
        counts: Dict[str, int],
        figsize: Tuple[int, int] = (12, 10)
    ) -> str:
        """
        Generate PRISMA 2020 flow diagram.

        Args:
            counts: Dictionary with PRISMA stage counts
                Required keys: identification, screened, excluded_title,
                excluded_abstract, retrieved, included
            figsize: Figure size

        Returns:
            Path to saved figure
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        fig, ax = plt.subplots(figsize=figsize)

        # Box style
        box_style = dict(
            boxstyle='round,pad=0.3',
            facecolor='white',
            edgecolor='black',
            linewidth=1.5
        )

        # PRISMA stages
        stages = [
            ('Identification', counts.get('identification', 0), (0.5, 0.9)),
            ('Duplicates Removed', counts.get('duplicates', 0), (0.5, 0.75)),
            ('Records Screened', counts.get('screened', 0), (0.5, 0.6)),
            ('Excluded (Title/Abstract)', counts.get('excluded_title', 0), (0.8, 0.6)),
            ('Full-text Retrieved', counts.get('retrieved', 0), (0.5, 0.45)),
            ('Excluded (Full-text)', counts.get('excluded_fulltext', 0), (0.8, 0.45)),
            ('Included in Review', counts.get('included', 0), (0.5, 0.25))
        ]

        # Draw boxes
        for stage, count, pos in stages:
            text = f"{stage}\n(n = {count})"
            ax.annotate(
                text,
                xy=pos,
                ha='center',
                va='center',
                fontsize=10,
                fontweight='bold' if 'Included' in stage else 'normal',
                bbox=box_style
            )

        # Draw arrows
        arrows = [
            ((0.5, 0.87), (0.5, 0.78)),  # Identification -> Duplicates
            ((0.5, 0.72), (0.5, 0.63)),  # Duplicates -> Screened
            ((0.6, 0.6), (0.73, 0.6)),   # Screened -> Excluded
            ((0.5, 0.57), (0.5, 0.48)),  # Screened -> Retrieved
            ((0.6, 0.45), (0.73, 0.45)), # Retrieved -> Excluded
            ((0.5, 0.42), (0.5, 0.28))   # Retrieved -> Included
        ]

        for start, end in arrows:
            ax.annotate(
                '',
                xy=end,
                xytext=start,
                arrowprops=dict(
                    arrowstyle='->',
                    connectionstyle='arc3',
                    color='black',
                    linewidth=1.5
                )
            )

        # Labels
        ax.text(0.1, 0.9, 'IDENTIFICATION', fontsize=12, fontweight='bold', rotation=90, va='center')
        ax.text(0.1, 0.52, 'SCREENING', fontsize=12, fontweight='bold', rotation=90, va='center')
        ax.text(0.1, 0.25, 'INCLUDED', fontsize=12, fontweight='bold', rotation=90, va='center')

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('PRISMA 2020 Flow Diagram', fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()

        output_path = self.output_dir / 'prisma_flow_diagram.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        logger.info(f"Saved PRISMA flow diagram to {output_path}")
        return str(output_path)

    def plot_hr_function_distribution(
        self,
        figsize: Tuple[int, int] = (10, 6)
    ) -> str:
        """
        Plot distribution of papers by HR function.

        Returns:
            Path to saved figure
        """
        import matplotlib.pyplot as plt

        # Count HR functions
        hr_counts = Counter()
        for paper in self.coded_data:
            hr_data = paper.get('hr_function', {})
            if isinstance(hr_data, dict):
                value = hr_data.get('value', hr_data)
                if isinstance(value, dict):
                    primary = value.get('primary')
                    if primary:
                        hr_counts[primary] += 1

        if not hr_counts:
            logger.warning("No HR function data found")
            return ""

        # Sort by count
        sorted_items = sorted(hr_counts.items(), key=lambda x: x[1], reverse=True)
        labels, values = zip(*sorted_items)

        # Colors
        colors = [HR_FUNCTION_COLORS.get(l, '#999999') for l in labels]

        fig, ax = plt.subplots(figsize=figsize)

        bars = ax.barh(
            range(len(labels)),
            values,
            color=colors,
            edgecolor='white',
            linewidth=0.5
        )

        # Labels
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels([l.replace('_', ' ').title() for l in labels])
        ax.set_xlabel('Number of Papers')
        ax.set_title('Distribution of Papers by HR Function')

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, values)):
            ax.text(val + 0.3, i, str(val), va='center', fontsize=10)

        ax.invert_yaxis()
        plt.tight_layout()

        output_path = self.output_dir / 'hr_function_distribution.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved HR function distribution to {output_path}")
        return str(output_path)

    def plot_ethics_hr_heatmap(
        self,
        figsize: Tuple[int, int] = (12, 8)
    ) -> str:
        """
        Plot heatmap of ethical issues by HR function.

        Returns:
            Path to saved figure
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        ETHICS_PRINCIPLES = [
            'fairness_bias', 'transparency', 'accountability',
            'privacy', 'autonomy', 'wellbeing'
        ]

        HR_FUNCTIONS = [
            'recruitment', 'selection', 'performance_management',
            'learning_development', 'people_analytics', 'employee_relations',
            'workforce_planning', 'compensation_benefits'
        ]

        # Build matrix
        matrix = np.zeros((len(ETHICS_PRINCIPLES), len(HR_FUNCTIONS)))

        for paper in self.coded_data:
            hr_data = paper.get('hr_function', {})
            if isinstance(hr_data, dict):
                value = hr_data.get('value', hr_data)
                hr_func = value.get('primary') if isinstance(value, dict) else None
            else:
                hr_func = None

            if hr_func not in HR_FUNCTIONS:
                continue

            j = HR_FUNCTIONS.index(hr_func)

            ethical_issues = paper.get('ethical_issues', {})
            for i, principle in enumerate(ETHICS_PRINCIPLES):
                principle_data = ethical_issues.get(principle, {})
                if isinstance(principle_data, dict):
                    pvalue = principle_data.get('value', principle_data)
                    if isinstance(pvalue, dict) and pvalue.get('mentioned', False):
                        matrix[i, j] += 1

        # Create dataframe
        df = pd.DataFrame(
            matrix,
            index=[p.replace('_', ' ').title() for p in ETHICS_PRINCIPLES],
            columns=[h.replace('_', ' ').title() for h in HR_FUNCTIONS]
        )

        fig, ax = plt.subplots(figsize=figsize)

        sns.heatmap(
            df,
            ax=ax,
            cmap='YlOrRd',
            annot=True,
            fmt='.0f',
            linewidths=0.5,
            cbar_kws={'label': 'Number of Papers'}
        )

        ax.set_title('Ethical Issues by HR Function')
        ax.set_xlabel('HR Function')
        ax.set_ylabel('Ethical Issue')

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        output_path = self.output_dir / 'ethics_hr_heatmap.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved ethics-HR heatmap to {output_path}")
        return str(output_path)

    def plot_solution_treemap(
        self,
        figsize: Tuple[int, int] = (14, 10)
    ) -> str:
        """
        Create treemap visualization of solution taxonomy.

        Returns:
            Path to saved figure
        """
        try:
            import plotly.express as px
        except ImportError:
            logger.error("plotly not installed. Install with: pip install plotly")
            return ""

        # Count solutions
        solution_counts = {
            'technical': Counter(),
            'organizational': Counter(),
            'regulatory': Counter()
        }

        for paper in self.coded_data:
            solution_data = paper.get('solution_taxonomy', {})
            if isinstance(solution_data, dict):
                value = solution_data.get('value', solution_data)
                if isinstance(value, dict) and value.get('solutions_proposed', False):
                    for sol in value.get('technical_solutions', []):
                        solution_counts['technical'][sol] += 1
                    for sol in value.get('organizational_solutions', []):
                        solution_counts['organizational'][sol] += 1
                    for sol in value.get('regulatory_solutions', []):
                        solution_counts['regulatory'][sol] += 1

        # Build hierarchical data
        data = []
        for category, counts in solution_counts.items():
            for solution, count in counts.items():
                if count > 0:
                    data.append({
                        'category': category.title(),
                        'solution': solution.replace('_', ' ').title(),
                        'count': count
                    })

        if not data:
            logger.warning("No solution data found for treemap")
            return ""

        df = pd.DataFrame(data)

        fig = px.treemap(
            df,
            path=['category', 'solution'],
            values='count',
            color='category',
            color_discrete_map={
                'Technical': '#3498db',
                'Organizational': '#2ecc71',
                'Regulatory': '#e74c3c'
            },
            title='Solution Taxonomy Distribution'
        )

        fig.update_layout(
            font_size=12,
            title_font_size=14
        )

        output_path = self.output_dir / 'solution_treemap.html'
        fig.write_html(str(output_path))

        # Also save as static image if kaleido is available
        try:
            png_path = self.output_dir / 'solution_treemap.png'
            fig.write_image(str(png_path), width=1400, height=1000)
            logger.info(f"Saved solution treemap PNG to {png_path}")
        except Exception as e:
            logger.warning(f"Could not save PNG (install kaleido): {e}")

        logger.info(f"Saved solution treemap HTML to {output_path}")
        return str(output_path)

    def plot_stance_distribution(
        self,
        figsize: Tuple[int, int] = (8, 8)
    ) -> str:
        """
        Plot pie chart of stance distribution.

        Returns:
            Path to saved figure
        """
        import matplotlib.pyplot as plt

        stance_counts = Counter()
        for paper in self.coded_data:
            stance_data = paper.get('stance_classification', {})
            if isinstance(stance_data, dict):
                value = stance_data.get('value', stance_data)
                if isinstance(value, dict):
                    tone = value.get('overall_tone')
                    if tone:
                        stance_counts[tone] += 1

        if not stance_counts:
            logger.warning("No stance data found")
            return ""

        labels = list(stance_counts.keys())
        values = list(stance_counts.values())

        colors = {
            'AI_optimistic': '#2ecc71',
            'AI_critical': '#e74c3c',
            'balanced': '#3498db',
            'neutral': '#95a5a6'
        }
        pie_colors = [colors.get(l, '#999999') for l in labels]

        fig, ax = plt.subplots(figsize=figsize)

        wedges, texts, autotexts = ax.pie(
            values,
            labels=[l.replace('_', ' ').title() for l in labels],
            colors=pie_colors,
            autopct='%1.1f%%',
            startangle=90,
            explode=[0.02] * len(values)
        )

        for autotext in autotexts:
            autotext.set_fontsize(10)
            autotext.set_fontweight('bold')

        ax.set_title('Distribution of Paper Stances Toward AI in HR')

        plt.tight_layout()

        output_path = self.output_dir / 'stance_distribution.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved stance distribution to {output_path}")
        return str(output_path)

    def generate_all(self) -> Dict[str, str]:
        """
        Generate all visualizations.

        Returns:
            Dictionary mapping visualization name to file path
        """
        results = {}

        # HR function distribution
        try:
            results['hr_distribution'] = self.plot_hr_function_distribution()
        except Exception as e:
            logger.error(f"Failed to generate HR distribution: {e}")

        # Ethics-HR heatmap
        try:
            results['ethics_hr_heatmap'] = self.plot_ethics_hr_heatmap()
        except Exception as e:
            logger.error(f"Failed to generate ethics-HR heatmap: {e}")

        # Solution treemap
        try:
            results['solution_treemap'] = self.plot_solution_treemap()
        except Exception as e:
            logger.error(f"Failed to generate solution treemap: {e}")

        # Stance distribution
        try:
            results['stance_distribution'] = self.plot_stance_distribution()
        except Exception as e:
            logger.error(f"Failed to generate stance distribution: {e}")

        logger.info(f"Generated {len(results)} visualizations")
        return results


def load_coded_data(input_path: str) -> List[Dict]:
    """Load coded paper data from JSON file."""
    with open(input_path) as f:
        return json.load(f)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate Publication-Ready Visualizations"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input JSON file with coded papers"
    )
    parser.add_argument(
        "--output", "-o",
        default="./output/figures",
        help="Output directory for figures"
    )
    parser.add_argument(
        "--prisma",
        type=str,
        help="JSON file with PRISMA counts (optional)"
    )

    args = parser.parse_args()

    # Load data
    logger.info(f"Loading coded data from {args.input}")
    coded_data = load_coded_data(args.input)

    # Initialize visualizer
    visualizer = PublicationVisualizer(coded_data, args.output)

    # Generate all visualizations
    results = visualizer.generate_all()

    # Generate PRISMA diagram if counts provided
    if args.prisma:
        with open(args.prisma) as f:
            prisma_counts = json.load(f)
        results['prisma_flow'] = visualizer.plot_prisma_flow(prisma_counts)

    print("\n" + "="*60)
    print("VISUALIZATION GENERATION COMPLETE")
    print("="*60)
    print(f"Generated {len(results)} figures:")
    for name, path in results.items():
        if path:
            print(f"  - {name}: {path}")
    print("="*60)


if __name__ == "__main__":
    main()
