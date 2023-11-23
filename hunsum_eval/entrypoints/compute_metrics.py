from collections import defaultdict
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

import utils.keywords as kw
from metrics.metric_factory import MetricFactory

RESOUCRES_DIR = Path(__file__).parents[2] / "resources"


def batch(i1, i2, i3, n=1):
    iterable = list(i1)
    l = len(iterable)
    for ndx in range(0, l, n):
        yield i1[ndx:min(ndx + n, l)], i2[ndx:min(ndx + n, l)], i3[ndx:min(ndx + n, l)]


def main(original_file, config_file):
    config = yaml.safe_load(Path(config_file).read_text())
    metric_names = config[kw.METRICS]

    df = pd.read_csv(original_file, sep="\t")

    articles = df["article"].tolist()
    reference = df["lead"].tolist()
    generated = df["generated_lead"].tolist()

    for name in metric_names:
        results = defaultdict(lambda: [])
        print(name)
        metric = MetricFactory.get_metric(name)
        for g, r, a in tqdm(batch(generated, reference, articles, 100)):
            if 'blanc' in name:
                result = metric.evaluate_batch(g, a, aggregate=False)
            else:
                result = metric.evaluate_batch(g, r, aggregate=False)
            for key, res in result.items():
                results[key].extend(res)

        for key, res in results.items():
            df[f'{key}'] = res
        df.to_csv(original_file, index=False, sep="\t")


if __name__ == '__main__':
    original_file = RESOUCRES_DIR / "test_set_generated_manipulated_to_eval.tsv"
    original_file = RESOUCRES_DIR / "test_set_for_eval.tsv"

    config_file = 'config.yaml'
    main(original_file, config_file)
