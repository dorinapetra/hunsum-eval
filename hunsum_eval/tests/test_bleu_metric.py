import unittest

from metrics.bleu import Bleu


class TestBleuMetric(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.metric = Bleu()

    def test_evaluate_batch(self):
        summary = 'A kutya elment a boltba.'
        reference = 'A kiskutya elsétált a boltba'

        score = self.metric.evaluate_batch([summary], [reference], aggregate=False)

        self.assertAlmostEqual(score['bleu'][0], 17.9652, places=4)

    def test_evaluate_batch_aggregate(self):
        summary = 'A kutya elment a boltba.'
        reference = 'A kiskutya elsétált a boltba'

        score = self.metric.evaluate_batch([summary], [reference], aggregate=True)

        self.assertAlmostEqual(score['bleu'], 17.9652, places=4)

    def test_evaluate_example(self):
        summary = 'A kutya elment a boltba.'
        reference = 'A kiskutya elsétált a boltba'

        score = self.metric.evaluate_example(summary, reference)

        self.assertAlmostEqual(score['bleu'], 17.9652, places=4)
