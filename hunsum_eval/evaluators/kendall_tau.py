from evaluators.base_evaluator import BaseEvaluator
from scipy import stats


class KendallTau(BaseEvaluator):
    def evaluate(self, results_1, results_2):
        tau, p_value = stats.kendalltau(results_1, results_2)
        return tau
