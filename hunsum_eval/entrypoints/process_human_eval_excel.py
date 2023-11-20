import functools as ft
import math
from pathlib import Path

import numpy as np
import pandas as pd


def main(original_file, eval_file, out_file):
    annotators = ['Lakatos', 'Geri', 'Dorina']
    annotator_dfs = []

    for annotator in annotators:
        df_original = pd.read_excel(original_file, sheet_name="annotator_1_2")
        df = pd.read_excel(eval_file, sheet_name=annotator)
        df = df.drop(['lead_0', 'lead_1', 'lead'], axis=1)
        df = df.add_suffix(f'_{annotator}')
        df['uuid'] = df_original['uuid']
        df[f'mt5_{annotator}'] = df_original['mt5_base']
        df[f'b2b_{annotator}'] = df_original['b2b']
        df[f'lead_{annotator}'] = df_original['lead']
        df[f'lead_sort_{annotator}'] = df_original['lead_sort']
        annotator_dfs.append(df)

    # shuffle_df = pd.read_csv(shuffle_file, sep='\t')
    # all_dfs = [shuffle_df]
    # all_dfs.extend(annotator_dfs)
    all_dfs = annotator_dfs

    df = ft.reduce(lambda left, right: pd.merge(left, right, on='uuid', how='outer'), all_dfs)

    models = ['mt5_base', 'b2b']
    # df_annotators = [prop.split('_')[-1] for prop in filter(lambda x: '_0_property_A' in x, df.columns.values.tolist())]

    properties = ['Konzisztencia', 'Relevancia', 'Folyékonyság', 'Kohézió']

    for model in models:
        # df[f'{model}_gen'] = df.apply(lambda x: find_model_pred(x, model), axis=1)
        for annotator in annotators:
            for prop in properties:
                df[f'{model}_{prop}_{annotator}'] = df.apply(lambda x: find_model_prop(x, model, annotator, prop),
                                                             axis=1)

    for idx in [0, 1]:
        # df = df.drop(f'pred_{model_to_idx[model]}', axis=1)
        # idx = model_to_idx[model]
        for annotator in annotators:
            for prop in properties:
                df = df.drop(f'{idx}_{prop}_{annotator}', axis=1)

    # majority voting
    for model in models:
        for prop in properties:
            cols = [f'{model}_{prop}_{a}' for a in annotators]
            # df[f'majority_{model}_{prop}'] = df[cols].mode(axis='columns')[0]
            df[f'avg_{model}_{prop}'] = df[cols].mean(axis='columns')

    # cols = [f'best_{a}' for a in annotators]
    # for col in cols:
    #    df[col] = df[col].apply(lambda x: 2 if x == 'ext' else x)
    # df['majority_best'] = df[cols].mode(axis='columns')[0]

    # TODO article, lead, mt5, b2b
    df['article'] = df.apply(
        lambda x: x['article_Dorina'] if isinstance(x['article_Dorina'], str) else x['article_Boti'], axis=1)
    df['lead'] = df.apply(lambda x: x['lead_Dorina'] if isinstance(x['lead_Dorina'], str) else x['lead_Boti'], axis=1)
    df['mt5'] = df.apply(lambda x: x['mt5_Dorina'] if isinstance(x['mt5_Dorina'], str) else x['mt5_Boti'], axis=1)
    df['b2b'] = df.apply(lambda x: x['b2b_Dorina'] if isinstance(x['b2b_Dorina'], str) else x['b2b_Boti'], axis=1)

    for a in annotators:
        df = df.drop(f'lead_{a}', axis=1)
        df = df.drop(f'mt5_{a}', axis=1)
        df = df.drop(f'article_{a}', axis=1)
        df = df.drop(f'b2b_{a}', axis=1)

    # saving
    df.to_csv(out_file, sep='\t')


def find_model_prop(a, model, annotator, char):
    try:
        idx = a[f'lead_sort_{annotator}'].strip('[]').replace("'", '').split(', ').index(model)
    except:
        return np.nan
    pred_str = str(idx) + '_' + char + '_' + annotator
    return a[pred_str]


if __name__ == '__main__':
    original_file = Path('/home/dorka/projects/hunsum-eval/resources/human_eval/annotator.xlsx')
    eval_file = Path('/home/dorka/projects/hunsum-eval/resources/human_eval/Human_eval_thesis.xlsx')
    output_file = Path('/home/dorka/projects/hunsum-eval/resources/human_eval/human_eval_processed.csv')
    main(original_file, eval_file, output_file)
