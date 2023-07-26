import collections
from typing import List

import huspacy
import six
import torch
from transformers import AutoModel, AutoTokenizer

from embeddings.base_embedding import BaseEmbedding
from embeddings.ngram_embedding import NgramEmbedding
from utils.ngrams import get_ngrams


class BertEmbedding(BaseEmbedding):
    def __init__(self, model_name='SZTAKI-HLT/hubert-base-cc'):
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        try:
            self.nlp = huspacy.load('hu_core_news_lg',
                                    exclude=['tagger', 'parser', 'ner', 'lemmatizer', 'textcat', 'custom'])
        except OSError:
            huspacy.download('hu_core_news_lg')
            self.nlp = huspacy.load('hu_core_news_lg',
                                    exclude=['tagger', 'parser', 'ner', 'lemmatizer', 'textcat', 'custom'])

    def tokenize(self, text) -> List[str]:
        return [token.text for token in self.nlp(text)]

    def vectorize_text(self, text):
        inputs = self.tokenizer([text], padding=True)
        output = self.model(input_ids=torch.tensor(inputs.input_ids),
                            attention_mask=torch.tensor(inputs.attention_mask))
        embedding = output.last_hidden_state[:, 0, :][0]
        return embedding.detach().numpy()

    def vectorize_ngrams(self, tokens: List, n: int = 3) -> List[NgramEmbedding]:
        ngrams = collections.Counter(get_ngrams(tokens, n))
        ngram_embeddings = []

        for ngram, count in six.iteritems(ngrams):
            embedding = self.vectorize_text(' '.join(ngram))
            ngram_embeddings.append(NgramEmbedding(embedding=embedding, ngram=ngram, count=count))

        return ngram_embeddings
