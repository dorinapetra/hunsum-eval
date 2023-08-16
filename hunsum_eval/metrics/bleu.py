from summ_eval.bleu_metric import BleuMetric
from typing import List


class Bleu(BleuMetric):
    def __init__(self, bleu_args=None):
        super().__init__(n_workers=1)

    def evaluate_batch(self, summaries: List[str], references: List[str] = [], **kwargs):
        result = super().evaluate_batch(summaries, references, **kwargs)
        if kwargs['aggregate']:
            return result
        return {key: [i[key] for i in result] for key in result[0].keys()}
