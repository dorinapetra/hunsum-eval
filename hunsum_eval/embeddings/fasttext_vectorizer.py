import fasttext

from typing import List

from embeddings.base_vectorizer import BaseVectorizer


class FasttextVectorizer(BaseVectorizer):
    def __init__(self, model_name='cc.hu.300.bin'):
        super().__init__()
        self.model = fasttext.load_model(model_name)

    def vectorize_text(self, text: str):
        return self.model.get_sentence_vector(text)

    def vectorize_words(self, words: List[str]) -> List:
        return [self.model.get_word_vector(word) for word in words]
