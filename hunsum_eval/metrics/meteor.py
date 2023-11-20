from itertools import chain

import huspacy
import nltk
from nltk import PorterStemmer, StemmerI
from summ_eval.meteor_metric import MeteorMetric
from typing import List, Iterable, Callable, Tuple
from nltk.corpus.reader import XMLCorpusReader, WordNetCorpusReader
from nltk.translate.meteor_score import meteor_score, _generate_enums, _count_chunks, _match_enums, _enum_stem_match
from nltk.corpus import wordnet as wn
from summ_eval.metric import Metric
from pywnxml.WNQuery import WNQuery


# based on nltk.translate.meteor_score

class Meteor(Metric):
    def __init__(self):
        super().__init__()
        # nltk.download("wordnet")
        # nltk.download("omw-1.4")
        # nltk.download("extended_omw")
        self.nlp = huspacy.load('hu_core_news_lg')
        self.pos_tags = ['n', 'v', 'a', 'b']
        self.query = WNQuery('/home/dorka/projects/hunsum-eval/resources/huwn.xml', log=open('log.txt', 'w'))

    def evaluate_batch(self, summaries: List[str], references: List[str] = [], aggregate=False):
        results = {'meteor': []}
        for summary, reference in zip(summaries, references):
            results['meteor'].append(self.evaluate_example(summary, reference)['meteor'])
        if aggregate:
            return {key: sum(results[key]) / len(results[key]) for key in results.keys()}
        return results

    def evaluate_example(self, summary, reference):
        summary_tokenized = [token.lemma_ for token in self.nlp(summary)]
        reference_tokenized = [token.lemma_ for token in self.nlp(reference)]
        return {'meteor': self.single_meteor_score(summary_tokenized, reference_tokenized)}

    def single_meteor_score(
            self,
            hypothesis: Iterable[str],
            reference: Iterable[str],
            preprocess: Callable[[str], str] = str.lower,
            stemmer: StemmerI = PorterStemmer(),
            alpha: float = 0.9,
            beta: float = 3.0,
            gamma: float = 0.5,
    ) -> float:

        enum_hypothesis, enum_reference = _generate_enums(
            hypothesis, reference, preprocess=preprocess
        )
        translation_length = len(enum_hypothesis)
        reference_length = len(enum_reference)
        matches, _, _ = self._enum_align_words(
            enum_hypothesis, enum_reference, stemmer=stemmer
        )
        matches_count = len(matches)
        try:
            precision = float(matches_count) / translation_length
            recall = float(matches_count) / reference_length
            fmean = (precision * recall) / (alpha * precision + (1 - alpha) * recall)
            chunk_count = float(_count_chunks(matches))
            frag_frac = chunk_count / matches_count
        except ZeroDivisionError:
            return 0.0
        penalty = gamma * frag_frac ** beta
        return (1 - penalty) * fmean

    def _enum_align_words(
            self,
            enum_hypothesis_list: List[Tuple[int, str]],
            enum_reference_list: List[Tuple[int, str]],
            stemmer: StemmerI = PorterStemmer(),
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, str]], List[Tuple[int, str]]]:
        """
        Aligns/matches words in the hypothesis to reference by sequentially
        applying exact match, stemmed match and wordnet based synonym match.
        in case there are multiple matches the match which has the least number
        of crossing is chosen. Takes enumerated list as input instead of
        string input

        :param enum_hypothesis_list: enumerated hypothesis list
        :param enum_reference_list: enumerated reference list
        :param stemmer: nltk.stem.api.StemmerI object (default PorterStemmer())
        :param wordnet: a wordnet corpus reader object (default nltk.corpus.wordnet)
        :return: sorted list of matched tuples, unmatched hypothesis list,
                 unmatched reference list
        """
        exact_matches, enum_hypothesis_list, enum_reference_list = _match_enums(
            enum_hypothesis_list, enum_reference_list
        )

        stem_matches, enum_hypothesis_list, enum_reference_list = _enum_stem_match(
            enum_hypothesis_list, enum_reference_list, stemmer=stemmer
        )

        wns_matches, enum_hypothesis_list, enum_reference_list = self._enum_wordnetsyn_match(
            enum_hypothesis_list, enum_reference_list
        )

        return (
            sorted(
                exact_matches + stem_matches + wns_matches, key=lambda wordpair: wordpair[0]
            ),
            enum_hypothesis_list,
            enum_reference_list,
        )

    def _enum_wordnetsyn_match(
            self,
            enum_hypothesis_list: List[Tuple[int, str]],
            enum_reference_list: List[Tuple[int, str]],
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, str]], List[Tuple[int, str]]]:
        """
        Matches each word in reference to a word in hypothesis
        if any synonym of a hypothesis word is the exact match
        to the reference word.

        :param enum_hypothesis_list: enumerated hypothesis list
        :param enum_reference_list: enumerated reference list
        :param wordnet: a wordnet corpus reader object (default nltk.corpus.wordnet)
        """
        word_match = []
        for i in range(len(enum_hypothesis_list))[::-1]:
            hypothesis_syns = set(
                chain.from_iterable(
                    # (
                    #     lemma.name()
                    #     for lemma in synset.lemmas()
                    #     if lemma.name().find("_") < 0
                    # )
                    # for synset in wordnet.synsets(enum_hypothesis_list[i][1], lang='hun_wikt')
                    [synset for synset in self.get_synsets(enum_hypothesis_list[i][1])]
                )
            ).union({enum_hypothesis_list[i][1]})
            for j in range(len(enum_reference_list))[::-1]:
                if enum_reference_list[j][1] in hypothesis_syns:
                    word_match.append(
                        (enum_hypothesis_list[i][0], enum_reference_list[j][0])
                    )
                    enum_hypothesis_list.pop(i)
                    enum_reference_list.pop(j)
                    break
        return word_match, enum_hypothesis_list, enum_reference_list

    def get_synsets(self, literal: str):
        literals = set()
        literal_lemma = None  # TODO lemma from huspacy
        for pos in self.pos_tags:
            synsets = self.query.lookUpLiteral(literal, pos=pos)
            for synset in synsets:
                literals = literals.union([s.literal for s in synset.synonyms])
        return literals
