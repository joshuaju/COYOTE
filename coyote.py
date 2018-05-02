"""
Usage:
    coyote.py extract <TIMESERIES> <FEATURETABLE>
    coyote.py cluster --corr=<THRESHOLD> --measure=<MEASURE> [--out=<FILE>]
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


def cluster_train(features, true_labels, corr_threshold):
    assert len(features) == len(true_labels)
    features, dropped_features = __drop_features__(features, corr_threshold)
    scaler = clustering.get_scaler(features)
    scaled_features = clustering.scale_data(features, scaler)
    model, accuracy = clustering.train(scaled_features, true_labels)
    return model, scaler, dropped_features, accuracy


def cluster_validate(features, true_labels, to_drop, scaler, model):
    scaled_features = clustering.scale_data(features.drop(to_drop, axis=1), scaler)
    accuracy = clustering.validate(model, scaled_features, true_labels)
    return accuracy


def cluster_predict(features, to_drop, scaler, model):
    scaled_features = clustering.scale_data(features.drop(to_drop, axis=1), scaler)
    labels = clustering.predict(scaled_features, model)
    return labels


class ClusterPipelineOutput():
    def __init__(self, model, scaler, dropped_features, accuracy_frame):
        self.model = model
        self.scaler = scaler
        self.dropped_features = dropped_features
        self.accuracy_frame = accuracy_frame

    def get_model(self):
        return self.model

    def get_scaler(self):
        return self.scaler

    def get_dropped_features(self):
        return self.dropped_features

    def get_accuracy_frame(self):
        return self.accuracy_frame


def cluster_pipeline(dataset, validate, corr_threshold):
    assert dataset in [dataset_utils.DATASET_ORG, dataset_utils.DATASET_UTIL]
    assert isinstance(validate, bool)
    assert isinstance(corr_threshold, float)
    accuracy_frame = clustering.create_accuracy_frame()

    all_measures, all_labels = dataset_utils.load_dataset(dataset)
    result_map = {}
    for measure in all_measures.index.levels[1].unique():
        print "*** ", measure, " ***"
        features = all_measures.xs(measure, level='measure')
        labels = all_labels.xs(measure, level='measure')
        model, scaler, dropped_features, training_accuracy = cluster_train(features, labels, corr_threshold)
        clustering.append_to_accuracy_frame(
            frame=accuracy_frame, accuracy=training_accuracy,
            measure=measure, dataset=dataset, context="training",
            corr_threshold=corr_threshold, dropped=len(dropped_features)
        )

        if validate:
            all_measures_validation, all_labels_validation = dataset_utils.load_dataset(dataset_utils.DATASET_VALIDATION)
            features = all_measures_validation.xs(measure, level='measure')
            labels = all_labels_validation.xs(measure, level='measure')
            validation_accuracy = cluster_validate(features, labels, dropped_features, scaler, model)
            clustering.append_to_accuracy_frame(frame=accuracy_frame, accuracy=validation_accuracy,
                                                measure=measure, dataset=dataset, context="validation",
                                                corr_threshold=corr_threshold, dropped=len(dropped_features)
                                                )
        result_map[measure] = ClusterPipelineOutput(model, scaler, dropped_features, accuracy_frame)
    return result_map


def __remove_features_and_cluster__(corr_threshold, measure):
    assert isinstance(corr_threshold, float)
    # load datasets and labels -----------------------------------------------------------------------------------------
    org_neg_frame, org_neg_labels = dataset_utils.load_org_and_neg_combined_with_labels()
    util_neg_frame, util_neg_labels = dataset_utils.load_util_and_neg_combined_with_labels()
    val_frame, val_labels = dataset_utils.load_validation_combined_with_labels()
    # drop features ----------------------------------------------------------------------------------------------------
    print "\nCorrelation threshold is set to %s.\n" % corr_threshold
    _, dropped_features_org = __drop_features__(org_neg_frame, corr_threshold)
    val_org = val_frame.drop(dropped_features_org, axis=1)
    # cluster organisation and negative instances
    print "Organisation and Negative Instance (%s features dropped)" % len(dropped_features_org)
    org_nan = []  # TODO remove
    org_train_p, org_train_r, org_train_f, org_val_p, org_val_r, org_val_f = train_and_validate(
        features=org_neg_frame, true_labels=org_neg_labels,
        features_validation=val_org, true_labels_validation=val_labels
    )
    # cluster utility and negative instances ---------------------------------------------------------------------------
    print
    _, dropped_features_util = __drop_features__(util_neg_frame, corr_threshold)
    val_util = val_frame.drop(dropped_features_util, axis=1)
    print "Utility and Negative Instance (%s features dropped)" % len(dropped_features_util)
    util_nan = []  # TODO remove
    util_train_p, util_train_r, util_train_f, util_val_p, util_val_r, util_val_f = train_and_validate(
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


def extract(path_to_timeseries, path_to_featuretable):
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

    measures = ["integrations", "integrators", "commits", "commiters", "merges"]
    COL_REPO = 'filename'
    COL_DATE = 'date'

    timeseries = pd.read_csv(path_to_timeseries, index_col=[COL_REPO, COL_DATE], parse_dates=[COL_DATE])
    for COL_MEASURE in measures:
        feature_extraction.transform_timeseries_frame_to_featuretable(timeseries[COL_MEASURE].dropna(),
                                                                      path_to_featuretable, measure=COL_MEASURE)


def train_and_validate(features, true_labels, features_validation, true_labels_validation):
    ''' Trains and validates a K-means classifier. Prints results to stdout.'''

    scaler = clustering.get_scaler(features)

    scaled_features = clustering.scale_data(features, scaler)
    scaled_features_validation = clustering.scale_data(features_validation, scaler)

    assert len(features.columns) == (scaled_features.shape[1])
    assert len(features_validation.columns) == (scaled_features_validation.shape[1])

    model, train_p, train_r, train_f = clustering.train(scaled_features, true_labels)
    val_p, val_r, val_f = clustering.validate(model, scaled_features_validation, true_labels_validation)
    return train_p, train_r, train_f, val_p, val_r, val_f


def __drop_features__(frame, corr_threshold):
    ''' Drops features exceeding the correlatoin threshold permanently from the frame.

        Returns the dropped features (column names)
    '''
    to_drop = feature_analysis.get_correlating_features(frame, corr_threshold=corr_threshold)
    return frame.drop(to_drop, axis=1), to_drop


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
    extract(path_to_timeseries, path_to_featuretable)
elif args['cluster']:
    corr_threshold = float(args['--corr'])
    measure = args['--measure']

    results_org = cluster_pipeline(dataset_utils.DATASET_ORG, validate=True, corr_threshold=corr_threshold)
    results_util = cluster_pipeline(dataset_utils.DATASET_UTIL, validate=True, corr_threshold=corr_threshold)

    output = args['--out']
    if output:
        output = os.path.expanduser(output)
        write_header = True
        open_mode = 'w'
        if os.path.isfile(output):
            write_header = False
            open_mode = 'a'
        # TODO
        #frame = pd.concat([accuracy_frame_org, accuracy_frame_util])
        #frame.to_csv(output, header=write_header, mode=open_mode)
else:
    print "UNDEFINED COMMAND LINE ARGUMENTS"
    exit(1)
