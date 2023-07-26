import unittest

from metrics.rouge import Rouge


class RougeMetricTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.metric = Rouge()

    def test_evaluate_example(self):
        summary = 'A kutya elment a boltba.'
        reference = 'A kiskutya elsétált a piacra'

        score = self.metric.evaluate_example(summary, reference)['rouge']

        self.assertEqual(score['rouge_1_recall'], 0.28571)
        # self.assertEqual(score['rouge_2_recall'], 0.5)
        # self.assertEqual(score['rouge_l_recall'], 0.5)
        # self.assertEqual(score['rouge_1_precision'], 0.5)
        # self.assertEqual(score['rouge_2_precision'], 0.5)
        # self.assertEqual(score['rouge_l_precision'], 0.5)
        # self.assertEqual(score['rouge_1_f_score'], 0.5)
        # self.assertEqual(score['rouge_2_f_score'], 0.5)
        # self.assertEqual(score['rouge_l_f_score'], 0.5)
