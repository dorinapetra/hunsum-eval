from __future__ import absolute_import, division, print_function

import os
import string
from collections import defaultdict, Counter
from functools import partial
from itertools import chain
from math import log
from multiprocessing import Pool

import numpy as np
import torch
from pyemd import emd
from summ_eval.metric import Metric
from transformers import AutoModel, AutoTokenizer

from embeddings.bert_vectorizer import BertVectorizer

dirname = os.path.dirname(__file__)


class MoverScore(Metric):
    def __init__(self, model_type='SZTAKI-HLT/hubert-base-cc', n_gram=1, remove_subwords=True, batch_size=1,
                 stop_wordsf=os.path.join(dirname, '../../resources/stopwords.txt')):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(model_type).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_type, do_lower_case=True)

        self.bert_vectorizer = BertVectorizer()

        stop_words = []
        if stop_wordsf is not None:
            with open(stop_wordsf) as inputf:
                stop_words = inputf.read().strip().split(' ')
        self.stop_words = stop_words
        self.n_gram = n_gram
        self.remove_subwords = remove_subwords
        self.batch_size = batch_size

    def evaluate_example(self, summary, reference):
        idf_dict_summ = self.get_idf_dict([summary])
        idf_dict_ref = self.get_idf_dict([reference])
        score = self.word_mover_score([reference], [summary], idf_dict_ref, idf_dict_summ)
        score_dict = {"mover_score": score[0]}
        return score_dict

    def evaluate_batch(self, summaries, references, aggregate=True):
        idf_dict_summ = self.get_idf_dict(summaries)
        idf_dict_ref = self.get_idf_dict(references)

        scores = self.word_mover_score(references, summaries, idf_dict_ref, idf_dict_summ,
                                       batch_size=self.batch_size)
        if aggregate:
            return {"mover_score": sum(scores) / len(scores)}
        else:
            score_dict = {"mover_score": scores}
            return score_dict

    def truncate(self, tokens):
        if len(tokens) > 512 - 2:
            tokens = tokens[0:(512 - 2)]
        return tokens

    def process(self, a):
        a = ["[CLS]"] + self.truncate(self.tokenizer.tokenize(a)) + ["[SEP]"]
        a = self.tokenizer.convert_tokens_to_ids(a)
        return set(a)

    def word_mover_score(self, refs, hyps, idf_dict_ref, idf_dict_hyp, batch_size=1):
        preds = []
        for batch_start in range(0, len(refs), batch_size):
            batch_refs = refs[batch_start:batch_start + batch_size]
            batch_hyps = hyps[batch_start:batch_start + batch_size]

            ref_embedding, ref_idf, ref_tokens = self.get_bert_embedding(batch_refs, idf_dict_ref)
            hyp_embedding, hyp_idf, hyp_tokens = self.get_bert_embedding(batch_hyps, idf_dict_hyp)

            ref_embedding.div_(torch.norm(ref_embedding, dim=-1).unsqueeze(-1))
            hyp_embedding.div_(torch.norm(hyp_embedding, dim=-1).unsqueeze(-1))

            ref_embedding_max, _ = torch.max(ref_embedding[-5:], dim=0, out=None)
            hyp_embedding_max, _ = torch.max(hyp_embedding[-5:], dim=0, out=None)

            ref_embedding_min, _ = torch.min(ref_embedding[-5:], dim=0, out=None)
            hyp_embedding_min, _ = torch.min(hyp_embedding[-5:], dim=0, out=None)

            ref_embedding_avg = ref_embedding[-5:].mean(0)
            hyp_embedding_avg = hyp_embedding[-5:].mean(0)

            ref_embedding = torch.cat([ref_embedding_min, ref_embedding_avg, ref_embedding_max], -1)
            hyp_embedding = torch.cat([hyp_embedding_min, hyp_embedding_avg, hyp_embedding_max], -1)

            for i in range(len(ref_tokens)):
                if self.remove_subwords:
                    ref_ids = [k for k, w in enumerate(ref_tokens[i]) if
                               w not in set(string.punctuation) and '##' not in w and w not in self.stop_words]
                    hyp_ids = [k for k, w in enumerate(hyp_tokens[i]) if
                               w not in set(string.punctuation) and '##' not in w and w not in self.stop_words]
                else:
                    ref_ids = [k for k, w in enumerate(ref_tokens[i]) if
                               w not in set(string.punctuation) and w not in self.stop_words]
                    hyp_ids = [k for k, w in enumerate(hyp_tokens[i]) if
                               w not in set(string.punctuation) and w not in self.stop_words]

                ref_embedding_i, ref_idf_i = self.load_ngram(ref_ids, ref_embedding[i], ref_idf[i], 1)
                hyp_embedding_i, hyp_idf_i = self.load_ngram(hyp_ids, hyp_embedding[i], hyp_idf[i], 1)

                raw = torch.cat([ref_embedding_i, hyp_embedding_i], 0)
                raw.div_(torch.norm(raw, dim=-1).unsqueeze(-1) + 0.000001)

                distance_matrix = pairwise_distances(raw, raw)

                c1 = np.zeros(len(ref_idf_i) + len(hyp_idf_i), dtype=np.double)
                c2 = np.zeros(len(ref_idf_i) + len(hyp_idf_i), dtype=np.double)

                c1[:len(ref_idf_i)] = ref_idf_i
                c2[-len(hyp_idf_i):] = hyp_idf_i

                c1 = _safe_divide(c1, np.sum(c1))
                c2 = _safe_divide(c2, np.sum(c2))
                score = 1 - emd(c1, c2, distance_matrix.double().cpu().numpy())
                preds.append(score)
        return preds

    def get_idf_dict(self, arr, nthreads=1):
        idf_count = Counter()
        num_docs = len(arr)

        process_partial = partial(self.process)

        # with Pool(nthreads) as p:
        #     idf_count.update(chain.from_iterable(p.map(process_partial, arr)))

        idf_count.update(chain.from_iterable(self.process(a) for a in arr))

        idf_dict = defaultdict(lambda: log((num_docs + 1) / 1))
        idf_dict.update({idx: log((num_docs + 1) / (c + 1)) + 1 for (idx, c) in idf_count.items()})
        return idf_dict

    def collate_idf(self, arr, idf_dict, pad="[PAD]"):
        # tokens = [["[CLS]"] + self.truncate(self.tokenizer.tokenize(a)) + ["[SEP]"] for a in arr]

        result = self.bert_vectorizer.bert_tokenize(arr)
        token_ids = result["input_ids"]

        tokens = [self.bert_vectorizer.tokenizer.convert_ids_to_tokens(ids) for ids in token_ids]

        idf_weights = [[idf_dict[i] for i in a] for a in token_ids]

        pad_token = self.tokenizer.convert_tokens_to_ids([pad])[0]

        # padded, mask = padding(arr, pad_token, dtype=torch.long)

        mask = result["attention_mask"]
        padded_idf, _ = padding(idf_weights, pad_token, dtype=torch.float)

        padded = torch.tensor(token_ids).to(device=self.device)
        mask = torch.tensor(mask).to(device=self.device)

        # padded = padded.to(device=self.device)
        # mask = mask.to(device=self.device)
        return padded, padded_idf, mask, tokens

    def bert_encode(self, x, attention_mask):
        self.model.eval()
        with torch.no_grad():
            # last_hidden_state, pooler_output, hidden_states = self.model(x, attention_mask=attention_mask,
            #                                                              output_hidden_states=True)
            output = self.model(x, attention_mask=attention_mask,
                                output_hidden_states=True)
        return output['hidden_states']

    def get_bert_embedding(self, all_sens, idf_dict):
        padded_sens, padded_idf, mask, tokens = self.collate_idf(all_sens, idf_dict)

        embeddings = []
        with torch.no_grad():
            batch_embedding = self.bert_encode(padded_sens, attention_mask=mask)
            batch_embedding = torch.stack(batch_embedding)
            embeddings.append(batch_embedding)
            del batch_embedding

        total_embedding = torch.cat(embeddings, dim=-3)
        return total_embedding, padded_idf, tokens

    def load_ngram(self, ids, embedding, idf, o):
        new_a = []
        new_idf = []

        slide_wins = slide_window(np.array(ids), w=self.n_gram, o=o)
        for slide_win in slide_wins:
            new_idf.append(idf[slide_win].sum().item())
            scale = _safe_divide(idf[slide_win], idf[slide_win].sum(0)).unsqueeze(-1).to(self.device)
            tmp = (scale * embedding[slide_win]).sum(0)
            new_a.append(tmp)
        new_a = torch.stack(new_a, 0).to(self.device)
        return new_a, new_idf


def padding(arr, pad_token, dtype=torch.long):
    lens = torch.LongTensor([len(a) for a in arr])
    max_len = lens.max().item()
    padded = torch.ones(len(arr), max_len, dtype=dtype) * pad_token
    mask = torch.zeros(len(arr), max_len, dtype=torch.long)
    for i, a in enumerate(arr):
        padded[i, :lens[i]] = torch.tensor(a, dtype=dtype)
        mask[i, :lens[i]] = 1
    return padded, mask


# plus_mask = lambda x, m: x + (1.0 - m).unsqueeze(-1) * 1e30
# minus_mask = lambda x, m: x - (1.0 - m).unsqueeze(-1) * 1e30
# mul_mask = lambda x, m: x * m.unsqueeze(-1)
# masked_reduce_min = lambda x, m: torch.min(plus_mask(x, m), dim=1, out=None)
# masked_reduce_max = lambda x, m: torch.max(minus_mask(x, m), dim=1, out=None)
# masked_reduce_mean = lambda x, m: mul_mask(x, m).sum(1) / (m.sum(1, keepdim=True) + 1e-10)
# masked_reduce_geomean = lambda x, m: np.exp(mul_mask(np.log(x), m).sum(1) / (m.sum(1, keepdim=True) + 1e-10))
# idf_reduce_mean = lambda x, m: mul_mask(x, m).sum(1)
# idf_reduce_max = lambda x, m, idf: torch.max(mul_mask(minus_mask(x, m), idf), dim=1, out=None)
# idf_reduce_min = lambda x, m, idf: torch.min(mul_mask(plus_mask(x, m), idf), dim=1, out=None)


def pairwise_distances(x, y=None):
    x_norm = (x ** 2).sum(1).view(-1, 1)
    y_norm = (y ** 2).sum(1).view(1, -1)
    y_t = torch.transpose(y, 0, 1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    return torch.clamp(dist, 0.0, np.inf)


def slide_window(a, w=3, o=2):
    if a.size - w + 1 <= 0:
        w = a.size
    sh = (a.size - w + 1, w)
    st = a.strides * 2
    view = np.lib.stride_tricks.as_strided(a, strides=st, shape=sh)[0::o]
    return view.copy().tolist()


def _safe_divide(numerator, denominator):
    return numerator / (denominator + 0.00001)
