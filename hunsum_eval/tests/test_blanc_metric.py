import unittest

from metrics.blanc import Blanc


class BlancMetricTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.metric = Blanc()

    def test_evaluate_example(self):
        summary = 'A kutya elment a boltba.'
        reference = 'A kiskutya elsétált a piacra'

        score = self.metric.evaluate_example(summary, reference)

        self.assertAlmostEqual(score['blanc'], 0.3333, places=4)
