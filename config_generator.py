import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from dataset_utils import DATASET_ORG, DATASET_UTIL
from matplotlib.ticker import AutoMinorLocator
import json


def generate_config_from_exploratoin_file(path_explore, path_config):
    df = pd.read_csv(path_explore)
    df = df[df.label == 'P']
    config = walk_measures(df)
    with open(path_config, 'w') as config_file:
        json.dump(config, config_file)


def walk_measures(df, make_plots=True):
    config = {}
    config[DATASET_ORG] = {}
    config[DATASET_UTIL] = {}

    pagewidth_in_cm = 14.99786*2
    size=(pagewidth_in_cm/2.54, pagewidth_in_cm/2.54/5*2)
    fig_training, ax_training = plt.subplots(2, 5, sharex=False, sharey=True, figsize=size)
    #plt.suptitle("Training")
    fig_validation, ax_validation = plt.subplots(2, 5, sharex=False, sharey=True, figsize=size)
    #plt.suptitle("Validation")
    ax_index = 0
    ax_org = 0
    ax_util = 1
    for measure, series in df.groupby('measure'):
        series = series.drop('measure', axis=1)

        training = series[series.context == 'training']
        validation = series[series.context == 'validation']
        print "*** %s *****************************************" % measure
        print "Best Training:"
        find_best(training[training.dataset == 'org'], measure)
        find_best(training[training.dataset == 'util'], measure)

        print "\nBest Validation:"
        best_org = find_best(validation[validation.dataset == 'org'], measure)
        best_util = find_best(validation[validation.dataset == 'util'], measure)

        print "\n\n"

        config[DATASET_ORG][measure] = float("%.2f" % best_org)
        config[DATASET_UTIL][measure] = float("%.2f" % best_util)

        if make_plots:
            ax_training[ax_org][ax_index].set_title(measure)
            accuracy_plot(ax_training[ax_org][ax_index], training[training.dataset == 'org'], 'Organisation')
            accuracy_plot(ax_training[ax_util][ax_index], training[training.dataset == 'util'], 'Utility')

            ax_validation[ax_org][ax_index].set_title(measure)
            accuracy_plot(ax_validation[ax_org][ax_index], validation[validation.dataset == 'org'], 'Organisation')
            accuracy_plot(ax_validation[ax_util][ax_index], validation[validation.dataset == 'util'], 'Utility')
            ax_index = ax_index + 1

    handle_f, label_f = ax_training[0][0].get_legend_handles_labels()
    handle_p = plt.Line2D([],[], color='r', marker='^', linestyle='', label='Precision')
    handle_r = plt.Line2D([], [], color='g', marker='v', linestyle='', label='Recall')

    handles = [handle_p, handle_r, handle_f[0]]
    labels = [handle_p.get_label(), handle_r.get_label(), label_f[0]]
    fig_training.legend(handles, labels, 'lower center', ncol=3, prop={'size': 12})
    fig_validation.legend(handles, labels, 'lower center', ncol=3, prop={'size': 12})
    plt.tight_layout(pad=2.75)
    plt.savefig("/home/joshua/Desktop/validation_accuracy.svg")
    plt.close()
    plt.tight_layout(pad=2.75)
    plt.savefig("/home/joshua/Desktop/training_accuracy.svg")
    plt.close()
    exit() # TODO REMOVE
    return config


def find_best(frame, measure):
    frame = frame[frame.fmeasure == frame.fmeasure.max()]
    frame = frame[frame.precision == frame.precision.max()]
    frame = frame[frame.recall == frame.recall.max()]
    frame = frame[frame.threshold == frame.threshold.min()]
    print_latex_table(measure, frame)
    return frame.threshold.values[0]

def print_latex_table(measure, frame):
    dataset = frame.dataset.values[0]
    threshold = frame.threshold.values[0]
    p = frame.precision.values[0] * 100
    r = frame.recall.values[0] * 100
    f = frame.fmeasure.values[0] * 100
    dropped = frame.dropped.values[0]
    latex = "%s&%s&%.2f&%.2f&%.2f&%.2f&%s" % (dataset, measure, threshold, p, r, f, dropped)
    print latex

def accuracy_plot(ax, frame, dataset):
    ax = frame.plot(kind='scatter', x='threshold', y='precision', ax=ax, marker='^', color='r', alpha=0.5, label='Precision', legend=False)
    ax = frame.plot(kind='scatter', x='threshold', y='recall', ax=ax, marker='v', color='g', alpha=0.5, label='Recall', legend=False)
    ax = frame.plot(kind='line', x='threshold', y='fmeasure', markersize=3, marker='o', color='b', ax=ax, label='F-Measure', legend=False)
    ax = frame.plot(kind='line', x='threshold', y='MCC', markersize=5, marker='x', color='y', ax=ax, label='MCC', legend=False)
    ax.set_xlim([0, 1])
    ax.set_ylim([-1, 1])
    ax.set_xticks(np.arange(0, 1.05, 0.2))
    ax.set_yticks(np.arange(-1, 1.05, 0.2))
    ax.set_ylabel(dataset, size='large')

    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    ax.grid(color='k', which='both', linestyle='--', linewidth=1, alpha=0.2)

pd.set_option("display.max_rows", 500)
pd.set_option('display.expand_frame_repr', False)
