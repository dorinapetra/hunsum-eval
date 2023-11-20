import collections
from abc import ABC, abstractmethod

from typing import List

import six

from embeddings.ngram_embedding import NgramEmbedding
import huspacy

from utils.ngrams import get_ngrams


class BaseVectorizer(ABC):
    def __init__(self):
        try:
            self.nlp = huspacy.load('hu_core_news_lg',
                                    disable=['tagger', 'morphologizer', 'lookup_lemmatizer', 'trainable_lemmatizer',
                                             'ner', 'parser', 'expeerimental_arc_predicter',
                                             'expeerimental_arc_labeler'])
        except OSError:
            huspacy.download('hu_core_news_lg')
            self.nlp = huspacy.load('hu_core_news_lg',
                                    disable=['tagger', 'morphologizer', 'lookup_lemmatizer', 'trainable_lemmatizer',
                                             'ner', 'parser', 'expeerimental_arc_predicter',
                                             'expeerimental_arc_labeler'])

    def tokenize_words(self, text) -> List[str]:
        return [token.text for token in self.nlp(text)]

    def tokenize_sentences(self, text) -> List[str]:
        return [str(sent) for sent in self.nlp(text).sents]

    @abstractmethod
    def vectorize_text(self, text: str):
        raise NotImplementedError()

    @abstractmethod
    def vectorize_words(self, words: List[str]) -> List:
        raise NotImplementedError()

    def vectorize_ngrams(self, tokens: List[str], n: int = 3) -> List[NgramEmbedding]:
        ngrams = collections.Counter(get_ngrams(tokens, n))
        ngram_embeddings = []

        for ngram, count in six.iteritems(ngrams):
            embedding = self.vectorize_text(' '.join(ngram))
            ngram_embeddings.append(NgramEmbedding(embedding=embedding, ngram=ngram, count=count))

        return ngram_embeddings
