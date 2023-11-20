from collections import defaultdict
from pathlib import Path

import click
import pandas as pd
import yaml
import seaborn as sns
from pathlib import Path

from tqdm import tqdm

import utils.keywords as kw
from metrics.metric_factory import MetricFactory
from utils.data import load_dataset

RESOUCRES_DIR = Path(__file__).parents[2] / "resources"


def batch(i1, i2, i3, n=1):
    iterable = list(i1)
    l = len(iterable)
    for ndx in range(0, l, n):
        yield i1[ndx:min(ndx + n, l)], i2[ndx:min(ndx + n, l)], i3[ndx:min(ndx + n, l)]


def load_human_eval_datasets():
    df = pd.read_csv('/home/dorka/projects/hunsum-eval/resources/human_eval_result.csv', sep='\t')

    generated = df['mt5'].tolist()
    generated.extend(df['b2b'].tolist())

    reference = df['lead'].tolist()
    reference.extend(df['lead'].tolist())

    articles = df['article'].tolist()
    articles.extend(df['article'].tolist())

    return articles, reference, generated


# @click.command()
# @click.argument('generated_file')
# @click.argument('reference_file')
# @click.argument('output_file')
# @click.argument('config_file')
def main(generated_dir, reference_file, output_file, config_file, human_eval=False):
    config = yaml.safe_load(Path(config_file).read_text())
    metric_names = config[kw.METRICS]

    if human_eval:
        articles, reference, generated = load_human_eval_datasets()
    else:
        generated = pd.read_json(generated_dir / "mt5.jsonl", lines=True)[kw.GEN_LEAD].tolist()
        generated.extend(pd.read_json(generated_dir / "b2b.jsonl", lines=True)[kw.GEN_LEAD].tolist())

        test_df = load_dataset(reference_file)
        reference = test_df['lead'].tolist()
        reference.extend(test_df['lead'].tolist())

        articles = test_df['article'].tolist()
        articles.extend(test_df['article'].tolist())

    results = defaultdict(lambda: [])

    df = pd.read_csv(output_file)

    for name in metric_names:
        results = defaultdict(lambda: [])
        print(name)
        metric = MetricFactory.get_metric(name)
        # for g, r, a in tqdm(zip(generated, reference, articles)):
        #     if 'blanc' in name:
        #         result = metric.evaluate_example(g, a)
        #     else:
        #         result = metric.evaluate_example(g, r)
        #     for key, res in result.items():
        #         results[key].append(res)
        for g, r, a in tqdm(batch(generated, reference, articles, 100)):
            if 'blanc' in name:
                result = metric.evaluate_batch(g, a, aggregate=False)
            else:
                result = metric.evaluate_batch(g, r, aggregate=False)
            for key, res in result.items():
                results[key].extend(res)

        for key, res in results.items():
            df[f'{key}'] = res
        df.to_csv(output_file, index=False)

        # df = pd.DataFrame(results)
        # df.to_csv(output_file, index=False)


if __name__ == '__main__':
    generated_file = RESOUCRES_DIR / "human_eval"
    reference_file = RESOUCRES_DIR / "hunsum_2_test"
    output_file = 'test_final.csv'
    config_file = 'config.yaml'
    main(generated_file, reference_file, output_file, config_file, human_eval=False)
