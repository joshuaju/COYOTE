import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from dataset_utils import DATASET_ORG, DATASET_UTIL
import json


def generate_config_from_exploratoin_file(path_explore, path_config):
    df = pd.read_csv(path_explore)
    df = df[df.label == 'P']
    config = walk_measures(df)
    with open(path_config, 'w') as config_file:
        json.dump(config, config_file)


def walk_measures(df, make_plots=False):
    config = {}
    config[DATASET_ORG] = {}
    config[DATASET_UTIL] = {}
    for measure, series in df.groupby('measure'):
        series = series.drop('measure', axis=1)

        training = series[series.context == 'training']
        validation = series[series.context == 'validation']

        best_org = find_best(validation[validation.dataset == 'org'])
        best_util = find_best(validation[validation.dataset == 'util'])

        config[DATASET_ORG][measure] = float("%.2f" % best_org)
        config[DATASET_UTIL][measure] = float("%.2f" % best_util)

        if make_plots:
            fig, ax = plt.subplots(1, 2, sharex=True, sharey=False)
            ax[0].set_title('Training')
            accuracy_plot(ax[0], training[training.dataset == 'org'])
            ax[1].set_title('Validation')
            accuracy_plot(ax[1], validation[validation.dataset == 'org'])
            plt.suptitle("%s: %s" % ("Org", measure))

            fig, ax = plt.subplots(1, 2, sharex=True, sharey=False)
            ax[0].set_title('Training')
            accuracy_plot(ax[0], training[training.dataset == 'util'])
            ax[1].set_title('Validation')
            accuracy_plot(ax[1], validation[validation.dataset == 'util'])
            plt.suptitle("%s: %s" % ("Util", measure))
    plt.show()
    return config


def find_best(frame):
    frame = frame[frame.fmeasure == frame.fmeasure.max()]
    frame = frame[frame.precision == frame.precision.max()]
    frame = frame[frame.recall == frame.recall.max()]
    frame = frame[frame.threshold == frame.threshold.min()]
    print frame
    return frame.threshold.values[0]


def accuracy_plot(ax, frame):
    frame.plot(kind='scatter', x='threshold', y='precision', ax=ax, color='r', alpha=0.5, label='Precision')
    frame.plot(kind='scatter', x='threshold', y='recall', ax=ax, color='g', alpha=0.5, label='Recall')
    frame.plot(kind='scatter', x='threshold', y='fmeasure', color='b', ax=ax, label='F-Measure')

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xticks(np.arange(0, 1.05, 0.1))
    ax.set_yticks(np.arange(0, 1.05, 0.1))
    ax.grid(color='k', linestyle='--', linewidth=1, alpha=0.2)

# pd.set_option("display.max_rows", 500)
# pd.set_option('display.expand_frame_repr', False)
