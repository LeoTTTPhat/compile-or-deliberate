"""Results loading and paired statistics for the arm comparisons (DESIGN.md section 6)."""
from .load import load_episodes
from .stats import (clopper_pearson, cluster_bootstrap_ci, cochran_q, continuous_vs, holm, mcnemar_exact,
                    newcombe_paired_counts, newcombe_paired_diff_ci, paired_wide, pareto_front, success_vs,
                    summary_table, wilcoxon_paired, wilson_ci)

__all__ = ["load_episodes", "clopper_pearson", "cluster_bootstrap_ci", "cochran_q", "continuous_vs", "holm",
           "mcnemar_exact", "newcombe_paired_counts", "newcombe_paired_diff_ci", "paired_wide", "pareto_front",
           "success_vs", "summary_table", "wilcoxon_paired", "wilson_ci"]
