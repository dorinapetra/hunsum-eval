import glob
import os

import pandas as pd


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
