import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.manifold import TSNE


def get_correlating_features(features, corr_threshold, drop_feature=[]):
    ''' Identifies the correlating features. Returns a list of features with a correlation greater or equal
        to the threshold. The correlation is measured as an absolute value, so both positve and negative correlations
        are handled the same way. '''
    assert isinstance(features, pd.DataFrame)
    corr_frame = features.drop(drop_feature, axis=1).corr(method='pearson')
    for i, row_index in enumerate(corr_frame.index):
        for j, col_index in enumerate(corr_frame.columns):
            if i >= j:  # ignore diagonals and upper triangular matrix
                continue
            absolute_corr_value = np.absolute(corr_frame.loc[row_index][col_index])
            if absolute_corr_value >= corr_threshold:
                drop_feature.append(row_index)
                return get_correlating_features(features, corr_threshold, drop_feature)
    return drop_feature


def scatter_plot(features, labels):
    fig = plt.figure(figsize=(9, 9))
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
                ax.scatter(x, y, alpha=0.7, marker='x', s=5, label=lbl)

            if i - j == 0:
                #ax.set_title(col2, fontsize=12, rotation=90)
                ax.set_xticklabels([])
                ax.set_xticks([])
                ax.set_xlabel(col2, fontsize=10, rotation=90, labelpad=80)
                ax.xaxis.set_label_position('top')
                ax.xaxis.set_visible(True)
            if j == 0:
                ax.set_yticklabels([])
                ax.set_yticks([])
                ax.yaxis.set_visible(True)
                ax.set_ylabel(col1, fontsize=10, rotation=0, labelpad=40)
    fig.legend(labels.unique(), loc='upper right', prop={'size': 12})
    plt.tight_layout(pad=2, h_pad=0, w_pad=0)



def tsne(frame, n_components):
    model = TSNE(n_components=n_components, random_state=0)
    transformed = model.fit_transform(frame)
    return pd.DataFrame(transformed, index=frame.index)

def scatter_features_in_2d(features, labels, markers):
    assert isinstance(features, pd.DataFrame)
    assert isinstance(labels, pd.Series)
    assert len(features.columns) == 2
    assert len(features) == len(labels)
    assert len(markers) == len(labels.unique())

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)

    col0 = features.columns[0]
    col1 = features.columns[1]
    for idx, lbl in enumerate(labels.unique()):
        label_idx = (labels == lbl)
        m = markers[idx]
        ax.scatter(features[labels == lbl][col0], features[labels == lbl][col1], marker=m, label=lbl)

    plt.legend(loc='best')
    plt.tight_layout(pad=3, h_pad=0, w_pad=0)