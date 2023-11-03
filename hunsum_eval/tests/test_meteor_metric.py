import unittest

from metrics.blanc import Blanc
from metrics.meteor import Meteor


class BlancMetricTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.metric = Meteor()

    def test_evaluate_batch(self):
        summary = 'A kutya elment a boltba.'
        reference = 'A kiskutya elsétált a boltba'

        score = self.metric.evaluate_batch([summary], [reference], aggregate=False)

        self.assertAlmostEqual(score['blanc'][0], 0.66666, places=4)

    def test_evaluate_batch_aggregate(self):
        summary = ['A kutya elment a boltba.']
        reference = ['A kiskutya elsétált a boltba']

        score = self.metric.evaluate_batch(summary, reference, aggregate=True)

        self.assertAlmostEqual(score['blanc'], 0.66666, places=4)

    def test_evaluate_example(self):
        summary = 'A kutya elment a boltba.'
        reference = 'A kiskutya elsétált a boltba'

        score = self.metric.evaluate_example(summary, reference)

        self.assertAlmostEqual(score['blanc'], 0.66666, places=4)
