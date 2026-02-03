#!/usr/bin/env python3
"""
Concept Evolution Tracking

Analyzes how AI ethics concepts have evolved over time:
- Ethical principle trends by research period
- Stance evolution (optimistic vs. critical)
- Solution emergence patterns
- Theoretical framework adoption

Version: 1.1.0
"""

import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict, Counter

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Constants
RESEARCH_PERIODS = ['2015_2017', '2018_2020', '2021_2023', '2024_2025']
PERIOD_LABELS = {
    '2015_2017': '2015-2017',
    '2018_2020': '2018-2020',
    '2021_2023': '2021-2023',
    '2024_2025': '2024-2025'
}

ETHICS_PRINCIPLES = [
    'fairness_bias', 'transparency', 'accountability',
    'privacy', 'autonomy', 'wellbeing'
]

STANCE_TYPES = ['AI_optimistic', 'AI_critical', 'balanced', 'neutral']

SOLUTION_CATEGORIES = ['technical', 'organizational', 'regulatory']


@dataclass
class PeriodStats:
    """Statistics for a single research period."""
    period: str
    n_papers: int = 0
    principle_counts: Dict[str, int] = field(default_factory=dict)
    principle_rates: Dict[str, float] = field(default_factory=dict)
    stance_counts: Dict[str, int] = field(default_factory=dict)
    stance_rates: Dict[str, float] = field(default_factory=dict)
    solution_counts: Dict[str, int] = field(default_factory=dict)
    solution_rates: Dict[str, float] = field(default_factory=dict)
    theory_counts: Dict[str, int] = field(default_factory=dict)
    avg_principles_per_paper: float = 0.0


@dataclass
class TrendAnalysis:
    """Trend analysis results for a concept."""
    concept: str
    periods: List[str]
    values: List[float]
    slope: float
    direction: str  # 'increasing', 'decreasing', 'stable'
    change_percent: float  # Total change from first to last period


class EvolutionTracker:
    """Tracks evolution of concepts across research periods."""

    def __init__(self, coded_data: List[Dict]):
        """
        Initialize with coded paper data.

        Args:
            coded_data: List of coded paper dictionaries
        """
        self.coded_data = coded_data
        self.period_stats: Dict[str, PeriodStats] = {}
        self.trends: Dict[str, TrendAnalysis] = {}

    def compute_period_stats(self) -> Dict[str, PeriodStats]:
        """Compute statistics for each research period."""
        # Group papers by period
        papers_by_period = defaultdict(list)

        for paper in self.coded_data:
            temporal = paper.get('temporal_metadata', {})
            period = temporal.get('research_period')

            if not period:
                # Try to derive from year
                year = paper.get('year')
                if year:
                    year = int(year) if isinstance(year, str) else year
                    if year <= 2017:
                        period = '2015_2017'
                    elif year <= 2020:
                        period = '2018_2020'
                    elif year <= 2023:
                        period = '2021_2023'
                    else:
                        period = '2024_2025'

            if period in RESEARCH_PERIODS:
                papers_by_period[period].append(paper)

        # Compute stats for each period
        for period in RESEARCH_PERIODS:
            papers = papers_by_period[period]
            stats = PeriodStats(period=period, n_papers=len(papers))

            if not papers:
                self.period_stats[period] = stats
                continue

            # Count ethical principles
            principle_counts = Counter()
            total_principles = 0

            for paper in papers:
                ethical_issues = paper.get('ethical_issues', {})
                for principle in ETHICS_PRINCIPLES:
                    principle_data = ethical_issues.get(principle, {})
                    if isinstance(principle_data, dict):
                        value = principle_data.get('value', principle_data)
                        if isinstance(value, dict) and value.get('mentioned', False):
                            principle_counts[principle] += 1
                            total_principles += 1

            stats.principle_counts = dict(principle_counts)
            stats.principle_rates = {
                p: c / len(papers) for p, c in principle_counts.items()
            }
            stats.avg_principles_per_paper = total_principles / len(papers) if papers else 0

            # Count stances
            stance_counts = Counter()
            for paper in papers:
                stance_data = paper.get('stance_classification', {})
                if isinstance(stance_data, dict):
                    value = stance_data.get('value', stance_data)
                    if isinstance(value, dict):
                        tone = value.get('overall_tone')
                        if tone:
                            stance_counts[tone] += 1

            stats.stance_counts = dict(stance_counts)
            stats.stance_rates = {
                s: c / len(papers) for s, c in stance_counts.items()
            }

            # Count solutions
            solution_counts = Counter()
            for paper in papers:
                solution_data = paper.get('solution_taxonomy', {})
                if isinstance(solution_data, dict):
                    value = solution_data.get('value', solution_data)
                    if isinstance(value, dict) and value.get('solutions_proposed', False):
                        if value.get('technical_solutions'):
                            solution_counts['technical'] += 1
                        if value.get('organizational_solutions'):
                            solution_counts['organizational'] += 1
                        if value.get('regulatory_solutions'):
                            solution_counts['regulatory'] += 1

            stats.solution_counts = dict(solution_counts)
            stats.solution_rates = {
                s: c / len(papers) for s, c in solution_counts.items()
            }

            # Count theoretical frameworks
            theory_counts = Counter()
            for paper in papers:
                theory_data = paper.get('theoretical_framework', {})
                if isinstance(theory_data, dict):
                    value = theory_data.get('value', theory_data)
                    if isinstance(value, dict) and value.get('applied', False):
                        theory_name = value.get('theory_name', 'Unknown')
                        if theory_name:
                            theory_counts[theory_name] += 1

            stats.theory_counts = dict(theory_counts)

            self.period_stats[period] = stats

        logger.info(f"Computed stats for {len(self.period_stats)} periods")
        return self.period_stats

    def analyze_trends(self) -> Dict[str, TrendAnalysis]:
        """Analyze trends for each concept across periods."""
        if not self.period_stats:
            self.compute_period_stats()

        # Ethical principle trends
        for principle in ETHICS_PRINCIPLES:
            values = [
                self.period_stats[p].principle_rates.get(principle, 0)
                for p in RESEARCH_PERIODS
            ]
            self.trends[f'principle_{principle}'] = self._compute_trend(
                principle, values
            )

        # Stance trends
        for stance in STANCE_TYPES:
            values = [
                self.period_stats[p].stance_rates.get(stance, 0)
                for p in RESEARCH_PERIODS
            ]
            self.trends[f'stance_{stance}'] = self._compute_trend(
                stance, values
            )

        # Solution trends
        for sol_type in SOLUTION_CATEGORIES:
            values = [
                self.period_stats[p].solution_rates.get(sol_type, 0)
                for p in RESEARCH_PERIODS
            ]
            self.trends[f'solution_{sol_type}'] = self._compute_trend(
                sol_type, values
            )

        logger.info(f"Analyzed {len(self.trends)} trends")
        return self.trends

    def _compute_trend(self, concept: str, values: List[float]) -> TrendAnalysis:
        """Compute trend analysis for a series of values."""
        # Linear regression for slope
        x = np.arange(len(values))
        y = np.array(values)

        if len(values) > 1 and np.std(y) > 0:
            slope = np.polyfit(x, y, 1)[0]
        else:
            slope = 0.0

        # Determine direction
        if slope > 0.02:  # Threshold for "increasing"
            direction = 'increasing'
        elif slope < -0.02:
            direction = 'decreasing'
        else:
            direction = 'stable'

        # Percent change
        first_val = values[0] if values else 0
        last_val = values[-1] if values else 0
        if first_val > 0:
            change_percent = ((last_val - first_val) / first_val) * 100
        elif last_val > 0:
            change_percent = 100.0  # From 0 to something
        else:
            change_percent = 0.0

        return TrendAnalysis(
            concept=concept,
            periods=RESEARCH_PERIODS,
            values=values,
            slope=float(slope),
            direction=direction,
            change_percent=float(change_percent)
        )

    def get_summary(self) -> Dict:
        """Generate summary of evolution analysis."""
        if not self.period_stats:
            self.compute_period_stats()
        if not self.trends:
            self.analyze_trends()

        # Papers per period
        papers_per_period = {
            p: stats.n_papers for p, stats in self.period_stats.items()
        }

        # Top increasing concepts
        increasing = [
            (k, t.change_percent) for k, t in self.trends.items()
            if t.direction == 'increasing'
        ]
        increasing.sort(key=lambda x: x[1], reverse=True)

        # Top decreasing concepts
        decreasing = [
            (k, t.change_percent) for k, t in self.trends.items()
            if t.direction == 'decreasing'
        ]
        decreasing.sort(key=lambda x: x[1])

        return {
            'papers_per_period': papers_per_period,
            'total_papers': sum(papers_per_period.values()),
            'top_increasing': increasing[:5],
            'top_decreasing': decreasing[:5],
            'period_summaries': {
                p: {
                    'n_papers': stats.n_papers,
                    'avg_principles': stats.avg_principles_per_paper,
                    'top_principles': sorted(
                        stats.principle_rates.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:3],
                    'dominant_stance': max(
                        stats.stance_rates.items(),
                        key=lambda x: x[1]
                    ) if stats.stance_rates else None
                }
                for p, stats in self.period_stats.items()
            }
        }


class EvolutionVisualizer:
    """Generates visualizations for evolution analysis."""

    def __init__(
        self,
        tracker: EvolutionTracker,
        output_dir: str = './output/analysis'
    ):
        """
        Initialize visualizer.

        Args:
            tracker: EvolutionTracker with computed stats
            output_dir: Directory for output files
        """
        self.tracker = tracker
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_ethics_evolution(
        self,
        figsize: Tuple[int, int] = (12, 6)
    ) -> str:
        """Plot evolution of ethical principles over time."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=figsize)

        x = np.arange(len(RESEARCH_PERIODS))
        width = 0.12

        colors = plt.cm.Set2(np.linspace(0, 1, len(ETHICS_PRINCIPLES)))

        for i, principle in enumerate(ETHICS_PRINCIPLES):
            values = [
                self.tracker.period_stats[p].principle_rates.get(principle, 0)
                for p in RESEARCH_PERIODS
            ]
            offset = (i - len(ETHICS_PRINCIPLES)/2 + 0.5) * width
            bars = ax.bar(
                x + offset,
                values,
                width,
                label=principle.replace('_', ' ').title(),
                color=colors[i]
            )

        ax.set_xlabel('Research Period')
        ax.set_ylabel('Proportion of Papers Mentioning')
        ax.set_title('Evolution of AI Ethics Principles in HR Literature')
        ax.set_xticks(x)
        ax.set_xticklabels([PERIOD_LABELS[p] for p in RESEARCH_PERIODS])
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax.set_ylim(0, 1)

        plt.tight_layout()

        output_path = self.output_dir / 'ethics_evolution.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved ethics evolution plot to {output_path}")
        return str(output_path)

    def plot_stance_evolution(
        self,
        figsize: Tuple[int, int] = (10, 6)
    ) -> str:
        """Plot evolution of stances over time."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=figsize)

        colors = {
            'AI_optimistic': '#2ecc71',
            'AI_critical': '#e74c3c',
            'balanced': '#3498db',
            'neutral': '#95a5a6'
        }

        for stance in STANCE_TYPES:
            values = [
                self.tracker.period_stats[p].stance_rates.get(stance, 0)
                for p in RESEARCH_PERIODS
            ]
            ax.plot(
                [PERIOD_LABELS[p] for p in RESEARCH_PERIODS],
                values,
                marker='o',
                linewidth=2,
                markersize=8,
                label=stance.replace('_', ' ').title(),
                color=colors.get(stance, 'gray')
            )

        ax.set_xlabel('Research Period')
        ax.set_ylabel('Proportion of Papers')
        ax.set_title('Evolution of Stances Toward AI in HR')
        ax.legend()
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        output_path = self.output_dir / 'stance_evolution.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved stance evolution plot to {output_path}")
        return str(output_path)

    def plot_solution_evolution(
        self,
        figsize: Tuple[int, int] = (10, 6)
    ) -> str:
        """Plot evolution of solution types over time."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=figsize)

        x = np.arange(len(RESEARCH_PERIODS))
        width = 0.25

        colors = ['#3498db', '#2ecc71', '#e74c3c']

        for i, sol_type in enumerate(SOLUTION_CATEGORIES):
            values = [
                self.tracker.period_stats[p].solution_rates.get(sol_type, 0)
                for p in RESEARCH_PERIODS
            ]
            offset = (i - 1) * width
            ax.bar(
                x + offset,
                values,
                width,
                label=sol_type.title(),
                color=colors[i]
            )

        ax.set_xlabel('Research Period')
        ax.set_ylabel('Proportion of Papers with Solutions')
        ax.set_title('Evolution of Solution Types in AI Ethics Literature')
        ax.set_xticks(x)
        ax.set_xticklabels([PERIOD_LABELS[p] for p in RESEARCH_PERIODS])
        ax.legend()
        ax.set_ylim(0, 1)

        plt.tight_layout()

        output_path = self.output_dir / 'solution_evolution.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved solution evolution plot to {output_path}")
        return str(output_path)

    def plot_papers_per_period(
        self,
        figsize: Tuple[int, int] = (8, 5)
    ) -> str:
        """Plot number of papers per research period."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=figsize)

        periods = RESEARCH_PERIODS
        counts = [self.tracker.period_stats[p].n_papers for p in periods]

        ax.bar(
            [PERIOD_LABELS[p] for p in periods],
            counts,
            color='steelblue',
            edgecolor='darkblue',
            linewidth=1.5
        )

        ax.set_xlabel('Research Period')
        ax.set_ylabel('Number of Papers')
        ax.set_title('Publication Volume Over Time')

        # Add value labels on bars
        for i, v in enumerate(counts):
            ax.text(i, v + 0.5, str(v), ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()

        output_path = self.output_dir / 'papers_per_period.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved papers per period plot to {output_path}")
        return str(output_path)

    def create_dashboard(self) -> str:
        """Create interactive dashboard with Plotly."""
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            logger.error("plotly not installed. Install with: pip install plotly")
            return ""

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Ethical Principles Evolution',
                'Stance Evolution',
                'Solution Types Evolution',
                'Publication Volume'
            )
        )

        periods_display = [PERIOD_LABELS[p] for p in RESEARCH_PERIODS]

        # Ethics evolution
        for principle in ETHICS_PRINCIPLES:
            values = [
                self.tracker.period_stats[p].principle_rates.get(principle, 0)
                for p in RESEARCH_PERIODS
            ]
            fig.add_trace(
                go.Scatter(
                    x=periods_display,
                    y=values,
                    name=principle.replace('_', ' ').title(),
                    mode='lines+markers'
                ),
                row=1, col=1
            )

        # Stance evolution
        stance_colors = {
            'AI_optimistic': 'green',
            'AI_critical': 'red',
            'balanced': 'blue',
            'neutral': 'gray'
        }
        for stance in STANCE_TYPES:
            values = [
                self.tracker.period_stats[p].stance_rates.get(stance, 0)
                for p in RESEARCH_PERIODS
            ]
            fig.add_trace(
                go.Scatter(
                    x=periods_display,
                    y=values,
                    name=stance.replace('_', ' ').title(),
                    mode='lines+markers',
                    line=dict(color=stance_colors.get(stance))
                ),
                row=1, col=2
            )

        # Solution evolution
        for sol_type in SOLUTION_CATEGORIES:
            values = [
                self.tracker.period_stats[p].solution_rates.get(sol_type, 0)
                for p in RESEARCH_PERIODS
            ]
            fig.add_trace(
                go.Scatter(
                    x=periods_display,
                    y=values,
                    name=sol_type.title(),
                    mode='lines+markers'
                ),
                row=2, col=1
            )

        # Publication volume
        counts = [self.tracker.period_stats[p].n_papers for p in RESEARCH_PERIODS]
        fig.add_trace(
            go.Bar(
                x=periods_display,
                y=counts,
                name='Papers',
                marker_color='steelblue'
            ),
            row=2, col=2
        )

        fig.update_layout(
            height=800,
            title_text="AI Ethics in HR: Evolution Dashboard",
            showlegend=True
        )

        output_path = self.output_dir / 'evolution_dashboard.html'
        fig.write_html(str(output_path))

        logger.info(f"Saved interactive dashboard to {output_path}")
        return str(output_path)

    def save_analysis(self) -> str:
        """Save evolution analysis to JSON."""
        summary = self.tracker.get_summary()

        # Add trend data
        summary['trends'] = {
            k: {
                'concept': t.concept,
                'values': t.values,
                'slope': t.slope,
                'direction': t.direction,
                'change_percent': t.change_percent
            }
            for k, t in self.tracker.trends.items()
        }

        output_path = self.output_dir / 'evolution_analysis.json'
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Saved evolution analysis to {output_path}")
        return str(output_path)


def load_coded_data(input_path: str) -> List[Dict]:
    """Load coded paper data from JSON file."""
    with open(input_path) as f:
        return json.load(f)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Concept Evolution Tracking"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input JSON file with coded papers"
    )
    parser.add_argument(
        "--output", "-o",
        default="./output/analysis/evolution",
        help="Output directory for results"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Generate interactive dashboard"
    )

    args = parser.parse_args()

    # Load data
    logger.info(f"Loading coded data from {args.input}")
    coded_data = load_coded_data(args.input)

    # Track evolution
    tracker = EvolutionTracker(coded_data)
    tracker.compute_period_stats()
    tracker.analyze_trends()

    # Visualize
    visualizer = EvolutionVisualizer(tracker, args.output)
    visualizer.plot_ethics_evolution()
    visualizer.plot_stance_evolution()
    visualizer.plot_solution_evolution()
    visualizer.plot_papers_per_period()

    if args.interactive:
        visualizer.create_dashboard()

    visualizer.save_analysis()

    # Print summary
    summary = tracker.get_summary()

    print("\n" + "="*60)
    print("EVOLUTION ANALYSIS COMPLETE")
    print("="*60)
    print(f"Total papers analyzed: {summary['total_papers']}")
    print("\nPapers per period:")
    for period, count in summary['papers_per_period'].items():
        print(f"  {PERIOD_LABELS[period]}: {count}")
    print("\nTop increasing trends:")
    for concept, change in summary['top_increasing'][:3]:
        print(f"  {concept}: +{change:.1f}%")
    print("\nTop decreasing trends:")
    for concept, change in summary['top_decreasing'][:3]:
        print(f"  {concept}: {change:.1f}%")
    print("="*60)
    print(f"Output saved to: {args.output}")
    print("="*60)


if __name__ == "__main__":
    main()
