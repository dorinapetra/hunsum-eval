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
        # self.assertEqual(score['rouge_2_recall'], 0)
        # self.assertEqual(score['rouge_l_recall'], 0.28571)
        # self.assertEqual(score['rouge_1_precision'], 0.4)
        # self.assertEqual(score['rouge_2_precision'], 0)
        # self.assertEqual(score['rouge_l_precision'], 0.5)
        # self.assertEqual(score['rouge_1_f_score'], 0.5)
        # self.assertEqual(score['rouge_2_f_score'], 0.5)
        # self.assertEqual(score['rouge_l_f_score'], 0.5)

    def test_evaluate_batch(self):
        summary = 'A kutya elment a boltba.'
        reference = 'A kiskutya elsétált a piacra'

        score = self.metric.evaluate_batch([summary], [reference], aggregate=False)

        self.assertEqual(score['rouge_1_recall'][0], 0.28571)
        # self.assertEqual(score['rouge_2_recall'], 0.5)
        # self.assertEqual(score['rouge_l_recall'], 0.5)
        # self.assertEqual(score['rouge_1_precision'], 0.5)
        # self.assertEqual(score['rouge_2_precision'], 0.5)
        # self.assertEqual(score['rouge_l_precision'], 0.5)
        # self.assertEqual(score['rouge_1_f_score'], 0.5)
        # self.assertEqual(score['rouge_2_f_score'], 0.5)
        # self.assertEqual(score['rouge_l_f_score'], 0.5)

    def test_evaluate_batch_aggregate(self):
        summary = 'A kutya elment a boltba.'
        reference = 'A kiskutya elsétált a piacra'

        score = self.metric.evaluate_batch([summary], [reference], aggregate=True)

        self.assertEqual(score['rouge_1_recall'], 0.28571)
        # self.assertEqual(score['rouge_2_recall'], 0.5)
        # self.assertEqual(score['rouge_l_recall'], 0.5)
        # self.assertEqual(score['rouge_1_precision'], 0.5)
        # self.assertEqual(score['rouge_2_precision'], 0.5)
        # self.assertEqual(score['rouge_l_precision'], 0.5)
        # self.assertEqual(score['rouge_1_f_score'], 0.5)
        # self.assertEqual(score['rouge_2_f_score'], 0.5)
        # self.assertEqual(score['rouge_l_f_score'], 0.5)
