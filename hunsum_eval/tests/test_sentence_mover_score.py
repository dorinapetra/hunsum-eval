import unittest

from metrics.sentence_mover_score import SentenceMoverScore


class SentenceMoverScoreMetricTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.metric = SentenceMoverScore(metric='s+wms')

    def test_evaluate_batch(self):
        summary = 'A kutya elment a boltba'
        reference = 'A kiskutya elsétált a boltba'

        score = self.metric.evaluate_batch([summary], [reference], aggregate=False)

        self.assertAlmostEqual(score['sentence_movers_s+wms'][0], 0.7998078, places=4)

    def test_evaluate_batch_aggregate(self):
        summary = ['A kutya elment a boltba']
        reference = ['A kiskutya elsétált a boltba']

        score = self.metric.evaluate_batch(summary, reference, aggregate=True)

        self.assertAlmostEqual(score['sentence_movers_s+wms'], 0.7998078, places=4)

    def test_evaluate_example(self):
        summary = 'A kutya elment a boltba'
        reference = 'A kiskutya elsétált a boltba'

        score = self.metric.evaluate_example(summary, reference)

        self.assertAlmostEqual(score['sentence_movers_s+wms'], 0.7998078, places=4)
