# %%
import os
import pickle

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def to_heatmap(dataAblationsfp, protoMAMLfp, split, metric):
    with open(os.path.join(protoMAMLfp, "results.pickle"), mode='rb+') as f:
        all_results = pickle.load(f)

    agg_results = {}

    for i, file in enumerate(sorted(os.listdir(dataAblationsfp))):
        omitted_dataset = file.rsplit('_')[0]

        with open(os.path.join(dataAblationsfp, file), mode='rb+') as f:
            results = pickle.load(f)

        agg_results["({:})".format(i+1)] = {"({:d})".format(i+1):
            float("{:.0f}".format((float(results[dataset][split][metric].rsplit(' ')[0]) -
            float(all_results[dataset][split][metric].rsplit(' ')[0]))/
            float(all_results[dataset][split][metric].rsplit(' ')[0]) * 100))
            for i, dataset in enumerate(sorted(results.keys()))}

    df = pd.DataFrame.from_dict(agg_results).T

    fig, ax = plt.subplots(figsize=(3.03, 2.5))

    sns.heatmap(df, annot=True, vmin=-50, vmax=50, center=0,
                cmap="RdBu", linewidths=0, ax=ax,
                cbar=False)

    ax.set_xlabel("Evaluation Dataset", fontsize=11)
    ax.set_ylabel("Omitted Dataset", fontsize=11)
    #ax.set_title("Percentual Change {:s} on {:s} split".format(metric, split))

    plt.tick_params(axis='both', which='both', bottom=False,
                    top=False, left=False, right=False,
                    labelsize=9)

    plt.tight_layout()

    plt.savefig(f"./figures/data_ablation_{metric}_{split}.pdf")

    plt.show()

if __name__ == "__main__":
    dataAblationsfp = "./checkpoints/DataAblations"
    protoMAMLfp = "./checkpoints/ProtoMAMLv3/full_adamv2/evaluation_full_adamv2_k=8"
    split = "test"
    metric = "acc_scaled"

    to_heatmap(dataAblationsfp, protoMAMLfp, split, metric)
