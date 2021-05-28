import os
import pickle

import numpy as np
import pandas as pd
from tabulate import tabulate
import matplotlib
import matplotlib.pyplot as plt

def to_figure(proto_maml_version='inner_lr_1e-3_meta-lr_1e-3', proto_maml_path='./checkpoints/ProtoMAMLHParamv2',
              evaluation_version='evaluation', baseline_path='./checkpoints/Baselines',
              metric='acc_scaled', split='test', error_bars=True, title="k=4", figsize=(5.3, 17), legend=False):

    row_names = []
    records = []
    stds = []

    for d in os.listdir(baseline_path):
        if os.path.isdir(os.path.join(baseline_path, d)):
            with open(os.path.join(baseline_path, d, evaluation_version, 'results.pickle'), 'rb+') as file:
                row_names.append(d)
                results_dict = pickle.load(file)
                records.append({eval_set: float(results_dict[eval_set][split][metric].rsplit(
                    ' ')[0]) for eval_set in results_dict.keys()})
                stds.append({eval_set: float(results_dict[eval_set][split][metric].rsplit(
                    ' ')[1][1:-1]) for eval_set in results_dict.keys()})

    with open(os.path.join(proto_maml_path, proto_maml_version, evaluation_version, 'results.pickle'), 'rb+') as file:
        row_names.append("Proto-MAML")
        results_dict = pickle.load(file)

        records.append({eval_set: float(results_dict[eval_set][split][metric].rsplit(
            ' ')[0]) for eval_set in results_dict.keys()})
        stds.append({eval_set: float(results_dict[eval_set][split][metric].rsplit(
            ' ')[1][1:-1]) for eval_set in results_dict.keys()})

    results_table = pd.DataFrame.from_records(records)
    results_table.index = row_names

    std_table = pd.DataFrame.from_records(stds)
    std_table.index = row_names

    labels = []
    self_vals, max_vals, proto_maml_vals = [], [], []
    self_stds, max_stds, proto_maml_stds = [], [], []

    for r in sorted(results_table.columns):
        max_idx = np.argmax(results_table[r].drop([r, 'Proto-MAML']))

        self_val = results_table[r][r] - 1
        max_val = results_table[r].drop([r, 'Proto-MAML'])[max_idx] - 1
        proto_maml = results_table[r]['Proto-MAML'] - 1

        self_std = std_table[r][r]
        max_std = std_table[r].drop([r, 'Proto-MAML'])[max_idx]
        proto_maml_std = std_table[r]['Proto-MAML']

        labels.append(r)
        self_vals.append(self_val)
        max_vals.append(max_val)
        proto_maml_vals.append(proto_maml)

        self_stds.append(self_std / np.sqrt(10))
        max_stds.append(max_std / np.sqrt(10))
        proto_maml_stds.append(proto_maml_std / np.sqrt(10))

    num_labels = ["({:})".format(i+1) for i, l in enumerate(labels)]
    for l, i in zip(labels, num_labels):
        print(i, l)

    cmap = matplotlib.cm.get_cmap('tab20')

    x = np.arange(len(labels))  # the label locations
    width = 0.25  # the width of the bars

    fig, ax = plt.subplots(figsize=figsize)
    if error_bars:
        rects1 = ax.bar(x - width, self_vals, width,
                        yerr=self_stds, label='Fine-tuned', color=cmap(0))
        rects2 = ax.bar(x, max_vals, width, yerr=max_stds,
                        label='Best Transfer', color=cmap(1))
        rects3 = ax.bar(x + width, proto_maml_vals, width,
                        yerr=proto_maml_stds, label='Proto-MAML', color=cmap(2))
    else:
        rects1 = ax.bar(x - width, self_vals, width,
                        label='Fine-tuned', color=cmap(0))
        rects2 = ax.bar(x, max_vals, width,
                        label='Best Transfer', color=cmap(1))
        rects3 = ax.bar(x + width, proto_maml_vals, width,
                        label='Proto-MAML', color=cmap(2))

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(metric, fontsize=24)
    ax.set_title(title, fontsize=24)
    ax.set_xticks(x)
    ax.set_xticklabels(num_labels,
                       fontsize=24)
    ax.set_ylim((-0.5, 3.0))
    #plt.xticks(rotation=90)

    plt.yticks([-0.25, 0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
               [0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])

    if legend:
        plt.legend(loc="upper right", fontsize=24)

    fig.tight_layout()

    plt.savefig(f"./figures/{evaluation_version}_{title}_{metric}_{split}.pdf")

    plt.show()

if __name__ == "__main__":
    proto_maml_version = 'full_adamv2'
    proto_maml_path = './checkpoints/ProtoMAMLv3'
    evaluation_version = 'evaluation_full_adamv2_k=16'
    baseline_path = './checkpoints/Baselines'
    metric = 'acc_scaled'
    split = 'test'
    error_bars = False
    title = "k=16"
    figsize = (17, 5.3)
    legend = False

    to_figure(proto_maml_version, proto_maml_path,
              evaluation_version, baseline_path,
              metric, split, error_bars,
              title, figsize, legend)
