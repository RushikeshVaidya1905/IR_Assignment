from scipy import stats

class AblationStudy:
    @staticmethod
    def calculate_significance(baseline_scores: list[float], ablated_scores: list[float]) -> float:
        """
        Performs a paired t-test between two arrays of MAP/NDCG scores.
        Returns the p-value. A p-value < 0.05 indicates statistical significance.
        """
        if len(baseline_scores) != len(ablated_scores) or len(baseline_scores) < 2:
            raise ValueError("Score lists must be of equal length and contain at least 2 queries.")
            
        # Calculate the T-statistic and two-tailed p-value
        t_stat, p_value = stats.ttest_rel(baseline_scores, ablated_scores)
        return p_value