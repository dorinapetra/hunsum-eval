import utils.keywords as kw
from evaluators.base_evaluator import BaseEvaluator
from evaluators.kendall_tau import KendallTau
from evaluators.pearson_correlation import PearsonCorrelation
from evaluators.spearman_correlation import SpearmanCorrelation


class EvaluatorFactory:
    evaluators = {
        kw.KENDALL_TAU: KendallTau,
        kw.PEARSON_CORRELATION: PearsonCorrelation,
        kw.SPEARMAN_CORRELATION: SpearmanCorrelation,
    }

    @classmethod
    def get_evaluator(cls, name) -> BaseEvaluator:
        return cls.evaluators[name]()
