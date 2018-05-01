import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def walk_measures(df):
    for key, series in df.groupby('measure'):
        print ">>>Measure: ", key
        series = series.drop('measure', axis=1)

        training = series[series.context == 'training']
        validation = series[series.context == 'validation']

        find_best(validation[validation.dataset == 'org'])
        find_best(validation[validation.dataset == 'util'])
        continue

        fig, ax = plt.subplots(1, 2, sharex=True, sharey=False)
        ax[0].set_title('Training')
        accuracy_plot(ax[0], training[training.dataset == 'org'])
        ax[1].set_title('Validation')
        accuracy_plot(ax[1], validation[validation.dataset == 'org'])
        plt.suptitle("%s: %s" % ("Org", key))

        fig, ax = plt.subplots(1, 2, sharex=True, sharey=False)
        ax[0].set_title('Training')
        accuracy_plot(ax[0], training[training.dataset == 'util'])
        ax[1].set_title('Validation')
        accuracy_plot(ax[1], validation[validation.dataset == 'util'])
        plt.suptitle("%s: %s" % ("Util", key))

    plt.show()

def find_best(frame):
    frame = frame[frame.fmeasure == frame.fmeasure.max()]
    frame = frame[frame.precision == frame.precision.max()]
    frame = frame[frame.recall == frame.recall.max()]
    frame = frame[frame.threshold == frame.threshold.min()]
    print frame




def accuracy_plot(ax, frame):
    frame.plot(kind='scatter', x='threshold', y='precision', ax=ax, color='r', alpha=0.5, label='Precision')
    frame.plot(kind='scatter', x='threshold', y='recall', ax=ax, color='g', alpha=0.5, label='Recall')
    frame.plot(kind='scatter', x='threshold', y='fmeasure', color='b', ax=ax, label='F-Measure')

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xticks(np.arange(0, 1.05, 0.1))
    ax.set_yticks(np.arange(0, 1.05, 0.1))
    ax.grid(color='k', linestyle='--', linewidth=1, alpha=0.2)


df = pd.read_csv('plots/accuracy.csv')
df = df[df.label == 'P']
walk_measures(df)


