import unittest

from metrics.mover_score import MoverScore


class MoverScoreMetricTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.metric = MoverScore()

    def test_evaluate_batch(self):
        summary = 'A kutya elment a boltba.'
        reference = 'A kiskutya elsétált a boltba'

        score = self.metric.evaluate_batch([summary], [reference], aggregate=False)

        self.assertAlmostEqual(score['mover_score'][0], 0.90961, places=4)

    def test_evaluate_batch_aggregate(self):
        summary = ['A kutya elment a boltba.']
        reference = ['A kiskutya elsétált a boltba']

        score = self.metric.evaluate_batch(summary, reference, aggregate=True)

        self.assertAlmostEqual(score['mover_score'], 0.90961, places=4)

    def test_evaluate_example(self):
        summary = 'A kutya elment a boltba.'
        reference = 'A kiskutya elsétált a boltba'

        score = self.metric.evaluate_example(summary, reference)

        self.assertAlmostEqual(score['mover_score'], 0.90961, places=4)
