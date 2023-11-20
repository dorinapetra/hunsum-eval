import random

import pandas as pd
from typing import List

from embeddings.bert_vectorizer import BertVectorizer


def replace_sentence(vectorizer, article_words, summary_sentences: List, sent_count, i):
    sum_sentences = summary_sentences.copy()
    to_replace = [i % len(summary_sentences) for i in range(i, i + sent_count)]
    for i in to_replace:
        original_length = len(vectorizer.tokenize_words(summary_sentences[i]))
        new_sentence = ' '.join(random.sample(article_words, original_length))
        sum_sentences[i] = new_sentence

    return ' '.join(sum_sentences)


def shuffle_words(vectorizer, summary_sentences: List, sent_count, i):
    sum_sentences = summary_sentences.copy()
    to_shuffle = [i % len(summary_sentences) for i in range(i, i + sent_count)]
    for i in to_shuffle:
        words = vectorizer.tokenize_words(sum_sentences[i])
        sum_sentences[i] = ' '.join(random.sample(words, len(words)))

    return ' '.join(sum_sentences)


def main(input_file, output_file):
    df = pd.read_csv(input_file, sep='\t')
    vectorizer = BertVectorizer()

    df['mt5_sentences'] = df.mt5_base.apply(lambda x: vectorizer.tokenize_sentences(x))
    df['b2b_sentences'] = df.b2b.apply(lambda x: vectorizer.tokenize_sentences(x))

    df['article_words'] = df.article.apply(lambda x: vectorizer.tokenize_words(x))

    df = df[df.mt5_sentences.map(len) == 3]
    df = df[df.b2b_sentences.map(len) == 3]

    for model in ['b2b', 'mt5']:
        # random sentences
        for sent in range(1, 4):
            r = 3 if sent == 1 or sent == 2 else 1
            for i in range(r):
                df[f'{model}_{sent}_{i}_sentence'] = df.apply(
                    lambda x: replace_sentence(vectorizer, x['article_words'], x[f'{model}_sentences'], sent, i),
                    axis=1)
        # shuffle sentences
        df[f'{model}_shuffle_sentences'] = df.apply(
            lambda x: ' '.join(random.sample(x[f'{model}_sentences'], len(x[f'{model}_sentences']))), axis=1)
        # shuffle words
        for sent in range(1, 4):
            r = 3 if sent == 1 or sent == 2 else 1
            for i in range(r):
                df[f'{model}_shuffle_{i}_sentence'] = df.apply(
                    lambda x: shuffle_words(vectorizer, x[f'{model}_sentences'], sent, i), axis=1)

        # TODO: entity replacement

    df.to_csv(output_file, sep='\t', index=False)


if __name__ == '__main__':
    main('/home/dorka/projects/hunsum-eval/resources/test_set_generated.tsv',
         '/home/dorka/projects/hunsum-eval/resources/test_set_generated_manipulated.tsv')
