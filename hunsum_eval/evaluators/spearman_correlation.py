from evaluators.base_evaluator import BaseEvaluator
from scipy import stats


class SpearmanCorrelation(BaseEvaluator):
    def evaluate(self, results_1, results_2):
        tau, p_value = stats.spearmanr(results_1, results_2)
        return tau
