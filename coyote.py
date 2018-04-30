"""
Usage:
    coyote.py extract <TIMESERIES> <FEATURETABLE> --measure=<MEASURE>
    coyote.py cluster --corr=<THRESHOLD> --measure=<MEASURE> [--out=<FILE>] [--tsne]
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


def __remove_features_and_cluster__(corr_threshold, save_tsne, measure):
    assert isinstance(corr_threshold, float)
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
    org_nan, org_train_p, org_train_r, org_train_f, org_val_p, org_val_r, org_val_f = train_and_validate(
        features=org_neg_frame, true_labels=org_neg_labels,
        features_validation=val_org, true_labels_validation=val_labels
    )
    # cluster utility and negative instances ---------------------------------------------------------------------------
    print
    dropped_features_util = __drop_features__(util_neg_frame, corr_threshold)
    val_util = val_frame.drop(dropped_features_util, axis=1)
    print "Utility and Negative Instance (%s features dropped)" % len(dropped_features_util)
    util_nan, util_train_p, util_train_r, util_train_f, util_val_p, util_val_r, util_val_f = train_and_validate(
        features=util_neg_frame, true_labels=util_neg_labels,
        features_validation=val_util, true_labels_validation=val_labels
    )

    frame = pd.DataFrame(
        columns=['measure', 'dataset', 'context', 'threshold', 'label', 'precision', 'recall', 'f-measure', 'dropped'])
    org_dropped_total = len(dropped_features_org) + len(org_nan)
    frame.loc[len(frame)] = [measure, "org", "training", corr_threshold, "P", org_train_p[0], org_train_r[0],
                             org_train_f[0], org_dropped_total]
    frame.loc[len(frame)] = [measure, "org", "training", corr_threshold, "NP", org_train_p[1], org_train_r[1],
                             org_train_f[1], org_dropped_total]
    frame.loc[len(frame)] = [measure, "org", "validation", corr_threshold, "P", org_val_p[0], org_val_r[0],
                             org_val_f[0], org_dropped_total]
    frame.loc[len(frame)] = [measure, "org", "validation", corr_threshold, "NP", org_val_p[1], org_val_r[1],
                             org_val_f[1], org_dropped_total]

    util_dropped_total = len(dropped_features_util) + len(util_nan)
    frame.loc[len(frame)] = [measure, "util", "training", corr_threshold, "P", util_train_p[0], util_train_r[0],
                             util_train_f[0], util_dropped_total]
    frame.loc[len(frame)] = [measure, "util", "training", corr_threshold, "NP", util_train_p[1], util_train_r[1],
                             util_train_f[1], util_dropped_total]
    frame.loc[len(frame)] = [measure, "util", "validation", corr_threshold, "P", util_val_p[0], util_val_r[0],
                             util_val_f[0], util_dropped_total]
    frame.loc[len(frame)] = [measure, "util", "validation", corr_threshold, "NP", util_val_p[1], util_val_r[1],
                             util_val_f[1], util_dropped_total]
    return np.round(frame, decimals=2)
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
        # dtype={COL_MEASURE: np.float64}
    ).dropna()
    feature_extraction.transform_timeseries_frame_to_featuretable(frame, path_to_featuretable, measure=COL_MEASURE)


def train_and_validate(features, true_labels, features_validation, true_labels_validation):
    ''' Trains and validates a K-means classifier. Prints results to stdout.'''
    model, nan_columns, train_p, train_r, train_f = clustering.train(features, true_labels)
    val_p, val_r, val_f = clustering.validate(model, features_validation.drop(nan_columns, axis=1),
                                              true_labels_validation)
    return nan_columns, train_p, train_r, train_f, val_p, val_r, val_f


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
pd.set_option("display.max_rows", 500)
pd.set_option('display.expand_frame_repr', False)
args = docopt(__doc__)
if args['extract']:
    path_to_timeseries = args['<TIMESERIES>']
    path_to_featuretable = args['<FEATURETABLE>']
    measure = args['--measure']
    extract(path_to_timeseries, path_to_featuretable, measure)
elif args['cluster']:
    save_tsne = args['--tsne']
    corr_threshold = args['--corr']
    measure = args['--measure']
    frame = __remove_features_and_cluster__(corr_threshold=float(corr_threshold), save_tsne=save_tsne, measure=measure)

    output = args['--out']
    if output:
        output = os.path.expanduser(output)
        write_header = True
        open_mode = 'w'
        if os.path.isfile(output):
            write_header = False
            open_mode = 'a'
        frame.to_csv(output, header=write_header, mode=open_mode)
else:
    print "UNDEFINED COMMAND LINE ARGUMENTS"
    exit(1)
