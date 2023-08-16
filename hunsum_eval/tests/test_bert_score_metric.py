import unittest

from metrics.bert_score import BertScore


class BertScoreMetricTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.metric = BertScore()

    def test_evaluate_batch(self):
        summary = 'A kutya elment a piacra.'
        reference = 'A kiskutya elsétált a boltba'

        score = self.metric.evaluate_batch([summary], [reference], aggregate=False)

        self.assertAlmostEqual(score['bert_score_precision'][0], 0.90774, places=4)
        self.assertAlmostEqual(score['bert_score_recall'][0], 0.87249, places=4)
        self.assertAlmostEqual(score['bert_score_f1'][0], 0.88977, places=4)

    def test_evaluate_batch_aggregate(self):
        summary = 'A kutya elment a piacra.'
        reference = 'A kiskutya elsétált a boltba'

        score = self.metric.evaluate_batch([summary], [reference], aggregate=True)

        self.assertAlmostEqual(score['bert_score_precision'], 0.90774, places=4)
        self.assertAlmostEqual(score['bert_score_recall'], 0.87249, places=4)
        self.assertAlmostEqual(score['bert_score_f1'], 0.88977, places=4)

    def test_evaluate_example(self):
        summary = 'A kutya elment a piacra.'
        reference = 'A kiskutya elsétált a boltba'

        score = self.metric.evaluate_example(summary, reference)

        self.assertAlmostEqual(score['bert_score_precision'], 0.90774, places=4)
        self.assertAlmostEqual(score['bert_score_recall'], 0.87249, places=4)
        self.assertAlmostEqual(score['bert_score_f1'], 0.88977, places=4)
