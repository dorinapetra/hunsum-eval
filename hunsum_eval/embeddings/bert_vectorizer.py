from typing import List

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from embeddings.base_vectorizer import BaseVectorizer


class BertVectorizer(BaseVectorizer):
    def __init__(self, model_name='SZTAKI-HLT/hubert-base-cc'):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def bert_tokenize(self, text) -> List[str]:
        return self.tokenizer(text, padding=True, truncation=True)

    def vectorize_text(self, text):
        if type(text) != list:
            text = [text]
        inputs = self.tokenizer(text, padding=True, truncation=True)
        output = self.model(input_ids=torch.tensor(inputs.input_ids),
                            attention_mask=torch.tensor(inputs.attention_mask))
        embedding = output.last_hidden_state[:, 0, :][0]
        return embedding.detach().numpy()

    def vectorize_words(self, words: List[str]) -> List:
        """
        Vectorize a list of words by averaging the subword embeddings.
        """
        tokens = self.tokenizer([words], padding=True, truncation=True, is_split_into_words=True)
        output = self.model(input_ids=torch.tensor(tokens.input_ids),
                            attention_mask=torch.tensor(tokens.attention_mask))
        return list(self._get_avg_subword_embeddings(output.last_hidden_state[0], tokens[0].word_ids))

    def _get_avg_subword_embeddings(self, embeddings, word_ids):
        sub_word_embeddings = []
        curr_word_id = 0
        for embedding, word_id in zip(embeddings[1:-1], word_ids[1:-1]):
            if word_id != curr_word_id:
                yield np.array(sub_word_embeddings).mean(axis=0)
                sub_word_embeddings = [embedding.detach().numpy()]
                curr_word_id = word_id
            else:
                sub_word_embeddings.append(embedding.detach().numpy())
        yield np.array(sub_word_embeddings).mean(axis=0)
