from typing import List

import torch
from blanc import BlancTune, BlancHelp
from summ_eval.blanc_metric import BlancMetric
from transformers import AutoModelForMaskedLM


class Blanc(BlancMetric):
    def __init__(self, model: str = 'SZTAKI-HLT/hubert-base-cc', inference_batch_size: int = 128,
                 finetune_batch_size: int = 24, use_tune: bool = False):
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.model = model
        super().__init__(device=device, inference_batch_size=inference_batch_size,
                         finetune_batch_size=finetune_batch_size, use_tune=use_tune)

    def evaluate_batch(self, summaries: List[str], references: List[str] = [], **kwargs):
        if self.use_tune:
            blanc_mod = BlancTune(device='cuda', inference_batch_size=self.inference_batch_size,
                                  finetune_batch_size=self.finetune_batch_size,
                                  model_name=self.model)
        else:
            blanc_mod = BlancHelp(device=self.device, inference_batch_size=self.inference_batch_size,
                                  model_name=self.model)

        blanc_mod.model = self.init_model()
        results = blanc_mod.eval_pairs(summaries, references)
        results = {"blanc": results}
        return results

    def init_model(self):
        """Initialize the language model and send it to the given device
        Note: Transformers v.4 and higher made default return_dict=True.
        Args:
            device (str): torch device (usually "cpu" or "cuda")

        Returns:
            model: a model for masked language modeling torch model
        """
        try:
            model = AutoModelForMaskedLM.from_pretrained(self.model, return_dict=False).to(self.device)
        except:
            model = AutoModelForMaskedLM.from_pretrained(self.model).to(self.device)
        model.eval()
        return model
