from typing import Optional, Any, List

from summ_eval.rouge_metric import RougeMetric
from utils import keywords as kw
from summ_eval.rouge_we_metric import RougeWeMetric


class Rouge(RougeMetric):
    # rouge_args definition: https://github.com/andersjo/pyrouge/tree/master/tools/ROUGE-1.5.5
    def __init__(self, rouge_args=None):
        super().__init__(rouge_args=rouge_args)

    def evaluate_example(self, summary, reference):
        return super().evaluate_example(summary, reference)['rouge']

    def evaluate_batch(self, summaries: List[str], references: List[str] = [], **kwargs):
        result = super().evaluate_batch(summaries, references, **kwargs)
        if kwargs['aggregate']:
            return result['rouge']
        return {key: [i['rouge'][key] for i in result] for key in result[0]['rouge'].keys() if key in kw.ROUGE_NAMES}


if __name__ == '__main__':
    r = RougeWeMetric()
