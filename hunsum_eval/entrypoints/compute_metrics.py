import glob
import os
from collections import defaultdict
from pathlib import Path

import click
import pandas as pd
import yaml
from typing import List

import utils.keywords as kw
from metrics.metric_factory import MetricFactory


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

    #reference_df = pd.read_json(reference_file, lines=True)
    reference_df = load_dataset(reference_file)
    reference = reference_df['lead'].tolist()
    reference = reference[:10]

    results = defaultdict(lambda: [])

    for name in metric_names:
        metric = MetricFactory.get_metric(name)
        result: List = metric.evaluate_batch(generated, reference, aggregate=False)
        for key, res in result.items():
            results[key] = res




    a = 2


def load_dataset(data_dir, shuffle=False):
    files = [data_dir] if os.path.isfile(data_dir) else sorted(glob.glob(f'{data_dir}/*.jsonl.gz'))
    site_dfs = []
    for file in files:
        site_df = pd.read_json(file, lines=True)
        site_df = site_df[['lead', 'article', 'uuid']]
        site_df = site_df.dropna()
        site_df = site_df.astype('str')
        site_dfs.append(site_df)
    df = pd.concat(site_dfs)
    if shuffle:
        df = df.sample(frac=1, random_state=123)
    return df


if __name__ == '__main__':
    main()
