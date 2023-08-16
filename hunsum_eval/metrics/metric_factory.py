from summ_eval.metric import Metric

import utils.keywords as kw
from metrics.bert_score import BertScore
from metrics.blanc import Blanc
from metrics.rouge import Rouge
from metrics.rouge_we import RougeWE


class MetricFactory:
    metrics = {
        kw.ROUGE: Rouge,
        kw.ROUGE_WE: RougeWE,
        kw.BLANC: Blanc,
        kw.BERT_SCORE: BertScore
    }

    @classmethod
    def get_metric(cls, name) -> Metric:
        return cls.metrics[name]()
