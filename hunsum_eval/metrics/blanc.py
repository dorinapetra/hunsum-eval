from typing import List

import torch
from blanc import BlancTune, BlancHelp
from summ_eval.blanc_metric import BlancMetric


class Blanc(BlancMetric):
    def __init__(self, inference_batch_size: int = 128, finetune_batch_size: int = 24, use_tune: bool = False):
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        super().__init__(device=device, inference_batch_size=inference_batch_size,
                         finetune_batch_size=finetune_batch_size, use_tune=use_tune)

    def evaluate_batch(self, summaries: List[str], references: List[str] = [], **kwargs):
        if self.use_tune:
            blanc_mod = BlancTune(device='cuda', inference_batch_size=self.inference_batch_size,
                                  finetune_batch_size=self.finetune_batch_size,
                                  model_name='bert-base-multilingual-cased')
        else:
            blanc_mod = BlancHelp(device=self.device, inference_batch_size=self.inference_batch_size,
                                  model_name='bert-base-multilingual-cased')

        results = blanc_mod.eval_pairs(summaries, references)
        results = {"blanc": results}
        return results
