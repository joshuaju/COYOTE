"""
Usage:
    coyote.py extract <TIMESERIES> <FEATURETABLE> --measure=<MEASURE>
    coyote.py cluster [--tsne]

"""
import feature_extraction
import feature_analysis
import clustering
import dataset_utils

from docopt import docopt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys, os


def __remove_features_and_cluster__(corr_threshold, save_tsne):
    # load datasets and labels -----------------------------------------------------------------------------------------
    org_neg_frame, org_neg_labels = dataset_utils.load_org_and_neg_combined_with_labels()
    util_neg_frame, util_neg_labels = dataset_utils.load_util_and_neg_combined_with_labels()
    val_frame, val_labels = dataset_utils.load_validation_combined_with_labels()
    # drop features ----------------------------------------------------------------------------------------------------
    print "\nCorrelation threshold is set to %s.\n" % corr_threshold
    dropped_features_org = __drop_features__(org_neg_frame, corr_threshold)
    val_org = val_frame.drop(dropped_features_org, axis=1)
    # cluster organisation and negative instances
    print "Organisation and Negative Instance (%s features dropped)" % len(dropped_features_org)
    train_and_validate(
        features=org_neg_frame, true_labels=org_neg_labels,
        features_validation=val_org, true_labels_validation=val_labels
    )
    # cluster utility and negative instances ---------------------------------------------------------------------------
    print
    dropped_features_util = __drop_features__(util_neg_frame, corr_threshold)
    val_util = val_frame.drop(dropped_features_util, axis=1)
    print "Utility and Negative Instance (%s features dropped)" % len(dropped_features_util)
    train_and_validate(
        features=util_neg_frame, true_labels=util_neg_labels,
        features_validation=val_util, true_labels_validation=val_labels
    )
    # save tsne --------------------------------------------------------------------------------------------------------
    if save_tsne:
        save_tsne_plot(org_neg_frame, org_neg_labels, 'plots/tsne_org_neg.png')
        save_tsne_plot(util_neg_frame, util_neg_labels, 'plots/tsne_util_neg.png')


def extract(path_to_timeseries, path_to_featuretable, measure):
    ''' Extracts features from the specified time-series file and stores them in the specified featuretable file.

        The time-series is expected to be a csv file with the following columns:
        filename, date, merges, commits, integrations, commiters, integrators
    '''
    path_to_timeseries = os.path.expanduser(path_to_timeseries)
    path_to_featuretable = os.path.expanduser(path_to_featuretable)
    if not os.path.isfile(path_to_timeseries):
        print "Path to timeseries file does not exist: %s" % path_to_timeseries
        exit(1)
    if os.path.isfile(path_to_featuretable):
        print "Featuretable already exists: %s" % path_to_featuretable
        exit(1)

    COL_REPO = 'filename'
    COL_DATE = 'date'
    COL_MEASURE = measure

    frame = pd.read_csv(
        path_to_timeseries,
        index_col=[COL_REPO, COL_DATE], parse_dates=[COL_DATE],
        usecols=[COL_REPO, COL_DATE, COL_MEASURE],
        #dtype={COL_MEASURE: np.float64}
    ).dropna()
    feature_extraction.transform_timeseries_frame_to_featuretable(frame, path_to_featuretable, measure=COL_MEASURE)


def train_and_validate(features, true_labels, features_validation, true_labels_validation):
    ''' Trains and validates a K-means classifier. Prints results to stdout.'''
    model, nan_columns = clustering.train(features, true_labels)
    clustering.validate(model, features_validation.drop(nan_columns, axis=1), true_labels_validation)


def __drop_features__(frame, corr_threshold):
    ''' Drops features exceeding the correlatoin threshold permanently from the frame.

        Returns the dropped features (column names)
    '''
    to_drop = feature_analysis.get_correlating_features(frame, corr_threshold=corr_threshold)
    frame.drop(to_drop, axis=1, inplace=True)
    return to_drop


def save_tsne_plot(frame, labels, save_to_path):
    transformed = feature_analysis.tsne(frame, n_components=2)
    feature_analysis.scatter_features_in_2d(transformed, labels, ['x', 'o'])
    plt.savefig(save_to_path)
    plt.close()


# Parse command line arguments -----------------------------------------------------------------------------------------
args = docopt(__doc__)
if args['extract']:
    path_to_timeseries = args['<TIMESERIES>']
    path_to_featuretable = args['<FEATURETABLE>']
    measure = args['--measure']
    extract(path_to_timeseries, path_to_featuretable, measure)
elif args['cluster']:
    save_tsne = args['--tsne']
    __remove_features_and_cluster__(corr_threshold=0.85, save_tsne=save_tsne)
else:
    print "UNDEFINED COMMAND LINE ARGUMENTS"
    exit(1)
