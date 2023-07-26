import unittest

from metrics.rouge_we import RougeWE


class RougeWEMetricTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.metric = RougeWE()

    def test_evaluate_example(self):
        summary = 'A kutya elment a boltba.'
        reference = 'A kiskutya elsétált a piacra'

        score = self.metric.evaluate_example(summary, reference)

        self.assertEqual(score['rouge_we_3_p'], 0.75)
        self.assertEqual(score['rouge_we_3_r'], 1)
        self.assertEqual(score['rouge_we_3_f'], 0.8571428571428571)
