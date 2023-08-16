from typing import Optional, Any, List

from summ_eval.bert_score_metric import BertScoreMetric
from utils import keywords as kw


class BertScore(BertScoreMetric):
    def __init__(self, model_type='SZTAKI-HLT/hubert-base-cc'):
        super().__init__(model_type=model_type, nthreads=1)

    def evaluate_batch(self, summaries: List[str], references: List[str] = [], **kwargs):
        result = super().evaluate_batch(summaries, references, **kwargs)
        if kwargs['aggregate']:
            return result
        return {key: [i[key] for i in result] for key in result[0].keys()}

    def evaluate_example(self, summary: str, reference: str, **kwargs):
        result = super().evaluate_batch([summary], [reference], **kwargs)
        return result

