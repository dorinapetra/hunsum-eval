from typing import List

import torch
from blanc import BlancTune, BlancHelp
from summ_eval.blanc_metric import BlancMetric


class Blanc(BlancMetric):
    def __init__(self, model: str = 'SZTAKI-HLT/hubert-base-cc', inference_batch_size: int = 10,
                 finetune_batch_size: int = 24, use_tune: bool = False):
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.model = model
        super().__init__(device=device, inference_batch_size=inference_batch_size,
                         finetune_batch_size=finetune_batch_size, use_tune=use_tune)

    def evaluate_batch(self, summaries: List[str], references: List[str] = [], aggregate: bool = True):
        if self.use_tune:
            blanc_mod = BlancTune(device=self.device, inference_batch_size=self.inference_batch_size,
                                  finetune_batch_size=self.finetune_batch_size,
                                  model_name=self.model)
        else:
            blanc_mod = BlancHelp(device=self.device, inference_batch_size=self.inference_batch_size,
                                  model_name=self.model)

        scores = blanc_mod.eval_pairs(summaries, references)
        if aggregate:
            results = {"blanc": sum(scores) / len(scores)}
        else:
            results = {"blanc": scores}
        return results

    def evaluate_example(self, summary: str, reference: str, **kwargs):
        result = self.evaluate_batch([summary], [reference], **kwargs)
        return result
