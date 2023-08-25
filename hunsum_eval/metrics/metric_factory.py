from summ_eval.metric import Metric

import utils.keywords as kw
from metrics.bert_score import BertScore
from metrics.blanc import Blanc
from metrics.bleu import Bleu
from metrics.meteor import Meteor
from metrics.mover_score import MoverScore
from metrics.rouge import Rouge
from metrics.rouge_we import RougeWE


class MetricFactory:
    metrics = {
        kw.ROUGE: Rouge,
        kw.ROUGE_WE: RougeWE,
        kw.BLANC: Blanc,
        kw.BERT_SCORE: BertScore,
        kw.BLEU: Bleu,
        kw.METEOR: Meteor,
        kw.MOVER_SCORE: MoverScore
    }

    @classmethod
    def get_metric(cls, name) -> Metric:
        return cls.metrics[name]()
