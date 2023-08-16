from collections import defaultdict
from pathlib import Path

import click
import pandas as pd
import yaml
import seaborn as sns

import utils.keywords as kw
from metrics.metric_factory import MetricFactory
from utils.data import load_dataset


@click.command()
@click.argument('generated_file')
@click.argument('reference_file')
@click.argument('output_file')
@click.argument('config_file')
def main(generated_file, reference_file, output_file, config_file):
    config = yaml.safe_load(Path(config_file).read_text())
    metric_names = config[kw.METRICS]

    generated_df = pd.read_json(generated_file, lines=True)
    generated = generated_df[kw.GEN_LEAD].tolist()
    generated = generated[:10]

    # reference_df = pd.read_json(reference_file, lines=True)
    reference_df = load_dataset(reference_file)
    reference = reference_df['lead'].tolist()
    reference = reference[:10]

    results = defaultdict(lambda: [])

    for name in metric_names:
        metric = MetricFactory.get_metric(name)
        result = metric.evaluate_batch(generated, reference, aggregate=False)
        for key, res in result.items():
            results[key] = res

    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)


if __name__ == '__main__':
    main()
