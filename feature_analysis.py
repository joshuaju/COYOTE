import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def run_script():
    path_to_org = "featuretable/features_org.csv"
    path_to_util = "featuretable/features_util.csv"
    path_to_neg = "featuretable/features_neg.csv"
    # load feature tables
    org_frame = pd.read_csv(path_to_org, index_col=0).fillna(0)
    util_frame = pd.read_csv(path_to_util, index_col=0).fillna(0)
    neg_frame = pd.read_csv(path_to_neg, index_col=0).fillna(0)
    # create ground truth labels
    LABEL_PROJECT = "P"
    LABEL_NON_PROJECT = "NP"
    org_labels = pd.Series(LABEL_PROJECT, index=org_frame.index)
    util_labels = pd.Series(LABEL_PROJECT, index=util_frame.index)
    neg_labels = pd.Series(LABEL_NON_PROJECT, index=neg_frame.index)
    # combine data sets
    org_neg_frame = pd.concat([org_frame, neg_frame])
    util_neg_frame = pd.concat([util_frame, neg_frame])
    org_neg_labels = pd.concat([org_labels, neg_labels])
    util_neg_labels = pd.concat([util_labels, neg_labels])
    del org_frame, util_frame, neg_frame, org_labels, util_labels, neg_labels
    # find and drop correlating features
    drop_org = get_correlating_features(org_neg_frame)
    drop_util = get_correlating_features(util_neg_frame)
    org_neg_frame.drop(drop_org, axis=1, inplace=True)
    util_neg_frame.drop(drop_util, axis=1, inplace=True)
    # scatter the features
    scatter_plot(org_neg_frame, org_neg_labels)
    scatter_plot(util_neg_frame, util_neg_labels)

    plt.show()


def get_correlating_features(features, drop_feature = [], corr_threshold = 0.9):
    corr_frame = features.drop(drop_feature, axis=1).corr()
    for i, row_index in enumerate(corr_frame.index):
        for j, col_index in enumerate(corr_frame.columns):
            if i < j: # iterate over lower triangular matrix
                absolute_corr_value = np.absolute(corr_frame.loc[row_index][col_index])
                if absolute_corr_value >= corr_threshold:
                    drop_feature.append(row_index)
                    return get_correlating_features(features, drop_feature, corr_threshold)
    return drop_feature


def scatter_plot(features, labels):
    fig = plt.figure(figsize=(5,5))
    n_features = len(features.columns)
    grid = gridspec.GridSpec(n_features, n_features, wspace=0, hspace=0)
    for i, col1 in enumerate(features.columns):
        for j, col2 in enumerate(features.columns):
            if i < j:
                continue
            ax = fig.add_subplot(grid[i, j])
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)

            for lbl in labels.unique():
                x, y = features[labels == lbl][col1], features[labels == lbl][col2]
                ax.scatter(x, y, alpha=0.5, marker='o', s=10, label=lbl)

            if i - j == 0:
                ax.set_title(col2, fontsize=12, rotation=90)
            if j == 0:
                ax.set_yticklabels([])
                ax.set_yticks([])
                ax.yaxis.set_visible(True)
                ax.set_ylabel(col1, fontsize=12, rotation=0, labelpad=35)
    fig.legend(labels.unique(), loc='upper right', prop={'size': 12})
    plt.tight_layout(pad=1.5, h_pad=0, w_pad=0)

# ----------------------------------------------------------------------------------------------------------------------
pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_columns", 500)
pd.set_option('display.expand_frame_repr', False)
run_script()