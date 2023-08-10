from typing import List

from scipy import spatial
from summ_eval.rouge_we_metric import RougeWeMetric
from summ_eval.s3_utils import _ngram_count, _safe_f1

from embeddings.base_embedding import BaseEmbedding
from embeddings.bert_embedding import BertEmbedding


class RougeWE(RougeWeMetric):
    def __init__(self, embedding_model=''):
        super().__init__(n_workers=1)
        self.embedding: BaseEmbedding = BertEmbedding()
        self.THRESHOLD = 0.8

    def evaluate_example(self, summary, reference):
        # if not isinstance(reference, list):
        #     reference = [reference]
        # if not isinstance(summary, list):
        #     summary = [summary]
        score = self.rouge_n_we(summary, reference, self.n_gram, return_all=True, tokenize=self.tokenize)
        score_dict = {f"rouge_we_{self.n_gram}_p": score[0], f"rouge_we_{self.n_gram}_r": score[1],
                      f"rouge_we_{self.n_gram}_f": score[2]}
        return score_dict

    def evaluate_batch(self, summaries: List[str], references: List[str] = [], **kwargs):
        results = {
            'rouge_we_3_p': [],
            'rouge_we_3_r': [],
            'rouge_we_3_f': [],
        }
        del kwargs['aggregate']
        for summary, reference in zip(summaries, references):
            res = self.evaluate_example(summary, reference, **kwargs)
            results['rouge_we_3_p'] += [res['rouge_we_3_p']]
            results['rouge_we_3_r'] += [res['rouge_we_3_r']]
            results['rouge_we_3_f'] += [res['rouge_we_3_f']]
        return results

    def rouge_n_we(self, summary, reference, n, alpha=0.5, return_all=False, tokenize=False):
        """
        Compute the ROUGE-N-WE score of a peer with respect to one or more models, for
        a given value of `n`.
        """

        summary_tokens = self.embedding.tokenize(summary)
        reference_tokens = self.embedding.tokenize(reference)

        matches = 0
        recall_total = 0

        summary_embeddings = self.embedding.vectorize_ngrams(summary_tokens)
        reference_embeddings = self.embedding.vectorize_ngrams(reference_tokens)

        matches += self._soft_overlap(summary_embeddings, reference_embeddings)
        recall_total += _ngram_count(reference_tokens, n)
        precision_total = _ngram_count(summary_tokens, n)

        return _safe_f1(matches, recall_total, precision_total, alpha, return_all)

    def _soft_overlap(self, summary_embeddings, reference_embeddings):
        result = 0

        for summary_embedding in summary_embeddings:
            idx, closest, count, sim = self._find_closest(summary_embedding, reference_embeddings)
            if sim < self.THRESHOLD:
                continue
            if count <= summary_embedding.count:
                del reference_embeddings[idx]
                result += count
            else:
                reference_embeddings[idx].count -= summary_embedding.count
                result += summary_embedding.count

        return result

    def _find_closest(self, sum_emb, ref_emb):
        ranking_list = []
        for i, reference_embedding in enumerate(ref_emb):
            ## soft matching based on embeddings similarity
            ranking_list.append((i, sum_emb.ngram, sum_emb.count,
                                 1 - spatial.distance.cosine(reference_embedding.embedding, sum_emb.embedding)))

        # Sort ranking list according to sim
        ranked_list = sorted(ranking_list, key=lambda tup: tup[3], reverse=True)

        # extract top item
        if ranked_list:
            return ranked_list[0]
        else:
            return -1, sum_emb.ngram, sum_emb.count, 0


if __name__ == '__main__':
    r = RougeWeMetric()
