import collections
from abc import ABC, abstractmethod

from typing import List

from embeddings.ngram_embedding import NgramEmbedding


class BaseEmbedding(ABC):
    @abstractmethod
    def tokenize_words(self, text):
        raise NotImplementedError()

    @abstractmethod
    def vectorize_text(self, text: str):
        raise NotImplementedError()

    @abstractmethod
    def vectorize_ngrams(self, tokens: List[str]) -> List[NgramEmbedding]:
        raise NotImplementedError()
