import unittest

from embeddings.bert_vectorizer import BertVectorizer


class BertEmbeddingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.embedding = BertVectorizer()

    def test_vectorize_ngrams(self):
        tokens = ['Ez', 'itt', 'egy', 'példa', 'mondat']

        ngram_embeddings = self.embedding.vectorize_ngrams(tokens, n=4)

        self.assertEquals(len(ngram_embeddings), 2)

        self.assertEquals(ngram_embeddings[0].embedding.shape, (768,))
        self.assertEquals(ngram_embeddings[0].count, 1)
        self.assertEquals(ngram_embeddings[0].ngram, ('Ez', 'itt', 'egy', 'példa'))

        self.assertEquals(ngram_embeddings[1].embedding.shape, (768,))
        self.assertEquals(ngram_embeddings[1].count, 1)
        self.assertEquals(ngram_embeddings[1].ngram, ('itt', 'egy', 'példa', 'mondat'))

    def test_tokenize(self):
        tokens = self.embedding.tokenize_words('Ez itt egy mondat.')

        self.assertEquals(tokens, ['Ez', 'itt', 'egy', 'mondat', '.'])

    def test_bert_tokenize(self):
        tokens = self.embedding.bert_tokenize(['Ez itt egy mondat.', 'Ez itt egy mondat.'])

        self.assertEquals(tokens, ['Ez', 'itt', 'egy', 'mondat', '.'])

    def test_vectorize_text(self):
        vector = self.embedding.vectorize_text('Ez itt egy mondat.')

        self.assertEquals(vector.shape, (768,))
