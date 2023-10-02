import collections
from typing import List

import huspacy
import numpy as np
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
            self.nlp = huspacy.load('hu_core_news_trf',
                                    exclude=['tagger', 'parser', 'ner', 'lemmatizer', 'textcat', 'custom'])
        except OSError:
            huspacy.download('hu_core_news_trf')
            self.nlp = huspacy.load('hu_core_news_lg',
                                    exclude=['tagger', 'parser', 'ner', 'lemmatizer', 'textcat', 'custom'])

    def tokenize_words(self, text) -> List[str]:
        return [token.text for token in self.nlp(text)]

    def tokenize_sentences(self, text) -> List[str]:
        return [str(sent) for sent in self.nlp(text).sents]

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

    def vectorize_ngrams(self, tokens: List, n: int = 3) -> List[NgramEmbedding]:
        ngrams = collections.Counter(get_ngrams(tokens, n))
        ngram_embeddings = []

        for ngram, count in six.iteritems(ngrams):
            embedding = self.vectorize_text(' '.join(ngram))
            ngram_embeddings.append(NgramEmbedding(embedding=embedding, ngram=ngram, count=count))

        return ngram_embeddings

    def vectorize_words(self, words: List[str]) -> List:
        tokens = self.tokenizer([' '.join(words)], padding=True, truncation=True)
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
