import os
import pickle

import numpy as np
import pandas as pd
from tabulate import tabulate

def to_table(proto_maml_version = 'inner_lr_1e-3_meta-lr_1e-3', proto_maml_path = './checkpoints/ProtoMAMLHParamv2',
             evaluation_version = 'evaluation', baseline_path='./checkpoints/Baselines',
             metric='acc_scaled', split='test', to=None):

    def sum_result(index):
        index_ = np.array([[float(i) for i in idx.replace('(', '').replace(')', '').rsplit(' ')]
                        for idx in index])

        mean = np.mean(index_[:, 0])
        stdev = np.sqrt(np.mean(index_[:, 1]**2))

        agg_result = "{:.2f} ({:.2f})".format(mean, stdev)

        return agg_result


    def rank(index):
        vals = [float(cell.rsplit(' ')[0]) for cell in index]

        order = np.argsort(-np.array(vals))
        ranks = np.argsort(order) + 1

        return ranks


    row_names = []
    records = []

    for d in os.listdir(baseline_path):
        if os.path.isdir(os.path.join(baseline_path, d)):
            with open(os.path.join(baseline_path, d, evaluation_version, 'results.pickle'), 'rb+') as file:
                row_names.append(d)
                results_dict = pickle.load(file)
                records.append(
                    {eval_set: results_dict[eval_set][split][metric] for eval_set in results_dict.keys()})

    with open(os.path.join(proto_maml_path, proto_maml_version, evaluation_version, 'results.pickle'), 'rb+') as file:
        row_names.append("ProtoMAML")
        results_dict = pickle.load(file)
        records.append(
            {eval_set: results_dict[eval_set][split][metric] for eval_set in results_dict.keys()})

    results_table = pd.DataFrame.from_records(records)
    results_table.index = row_names
    results_table = results_table[sorted(list(results_table.columns))]
    results_table['mean'] = results_table.apply(sum_result, axis=1)

    results_table = results_table.T

    MRR = np.mean(1/np.stack(results_table[:-1].apply(rank, axis=1)), axis=0)
    MRR_std = np.std(1/np.stack(results_table[:-1].apply(rank, axis=1)), axis=0)
    MRR = ["{:.2f} ({:.2f})".format(mrr, std) for (mrr, std) in zip(MRR, MRR_std)]
    MRR = pd.DataFrame(MRR, index=row_names).T
    MRR.index = ["MRR"]
    results_table = results_table.append(MRR)

    if to == 'latex':
        print(results_table.to_latex())
    elif to == 'table':
        print(tabulate(results_table, headers='keys', tablefmt='psql'))
    elif to == None:
        return results_table

if __name__ == "__main__":
    proto_maml_version = 'inner_lr_1e-3_meta-lr_1e-3'
    proto_maml_path = './checkpoints/ProtoMAMLHParamv2'
    evaluation_version = 'evaluation_limitedProtoMAML_k4_test'
    baseline_path = './checkpoints/Baselines'
    metric = 'acc_scaled'
    split = 'test'

    table = to_table(proto_maml_version, proto_maml_path,
                     evaluation_version, baseline_path,
                     metric, split, to='latex')
