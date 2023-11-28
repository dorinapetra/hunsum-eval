import math

from nltk.corpus import stopwords
from typing import List
import logging
import numpy as np
from pyemd import emd
from summ_eval.metric import Metric

from embeddings.bert_vectorizer import BertVectorizer


# https://github.com/hechmik/word_mover_distance

class SentenceMoverScore(Metric):
    def __init__(self, model_type='SZTAKI-HLT/hubert-base-cc', metric='wms'):
        self.vectorizer = BertVectorizer(model_type)
        self.stop_words = set(stopwords.words('hungarian'))
        self.metric = metric

    def evaluate_batch(self, summaries: List[str], references: List[str] = [], **kwargs):
        results = []
        for summary, reference in zip(summaries, references):
            res = self.wmdistance(summary, reference)
            results.append(res)
        if kwargs['aggregate']:
            results = sum(results) / len(results)
        return {f"sentence_movers_{self.metric}": results}

    def evaluate_example(self, summary, reference):
        score = self.wmdistance(summary, reference)
        score_dict = {f"sentence_movers_{self.metric}": score}
        return score_dict

    def wmdistance(self, document1: str, document2: str):
        sum_words, sum_embeddings, sum_d = self.get_words_and_embeddings(document1)
        ref_words, ref_embeddings, ref_d = self.get_words_and_embeddings(document2)

        vocab_len = len(sum_words) + len(ref_words)

        distance_matrix = np.zeros((vocab_len, vocab_len), dtype=np.double)
        for i, embedding1 in enumerate(sum_embeddings + ref_embeddings):
            for j, embedding2 in enumerate(sum_embeddings + ref_embeddings):
                # If the current cell is empty compute cosine distance between word vectors.
                try:
                    if not distance_matrix[i, j]:
                        distance_matrix[i, j] = 1 - np.dot(embedding1, embedding2) / (
                                np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
                        # Fill the specular cell for saving computation
                        distance_matrix[j, i] = distance_matrix[i, j]
                except Exception as e:
                    a = 1

        if np.sum(distance_matrix) == 0.0:
            # `emd` gets stuck if the distance matrix contains only zeros.
            logging.info('The distance matrix is all zeros. Aborting (returning inf).')
            return float('inf')

        sum_d.extend([0 for _ in range(len(ref_words))])
        ref_d = [0 for _ in range(len(sum_words))] + ref_d

        wmd = emd(np.array(sum_d), np.array(ref_d), distance_matrix)
        sim = math.exp(-wmd)
        return sim

    def get_words_and_embeddings(self, text):
        words = []
        embeddings = []
        nbow = []
        mul = 2 if self.metric == 's+wms' else 1

        A = len(self.vectorizer.tokenize_words(text))

        if self.metric != 'sms':
            words.extend(self.vectorizer.tokenize_words(text))
            embeddings.extend(self.vectorizer.vectorize_words(words))
            nbow.extend([1 / (mul * A) for _ in range(len(words))])
        if self.metric != 'wms':
            sentences = self.vectorizer.tokenize_sentences(text)
            words.extend(sentences)
            embeddings.extend([self.vectorizer.vectorize_text(sent) for sent in sentences])
            nbow.extend([len(self.vectorizer.tokenize_words(sent)) / (mul * A) for sent in sentences])

        return words, embeddings, nbow
