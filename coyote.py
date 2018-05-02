"""
Usage:
    coyote.py extract <TIMESERIES> <FEATURETABLE>
    coyote.py cluster --config=<FILE> [--out=<FILE>]
    coyote.py explore --explore=<FILE> --config=<File>
"""
import feature_extraction
import feature_analysis
import clustering
import dataset_utils
import config_generator

from docopt import docopt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys, os
import subprocess


# ----------------------------------------------------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------------------------------------------------

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


def cluster_pipeline(dataset, validate, config):
    assert dataset in [dataset_utils.DATASET_ORG, dataset_utils.DATASET_UTIL]
    assert isinstance(validate, bool)

    all_measures, all_labels = dataset_utils.load_dataset(dataset)
    result_map = {}
    measure_names = all_measures.index.levels[1].unique()  # level[1] is the 'measure' column
    for measure in measure_names:
        accuracy_frame = clustering.create_accuracy_frame()

        corr_threshold = config[dataset][measure]
        assert not corr_threshold == None

        features = all_measures.xs(measure, level='measure')
        labels = all_labels.xs(measure, level='measure')
        model, scaler, dropped_features, training_accuracy = cluster_train(features, labels, corr_threshold)
        clustering.append_to_accuracy_frame(
            frame=accuracy_frame, accuracy=training_accuracy,
            measure=measure, dataset=dataset, context="training",
            corr_threshold=corr_threshold, dropped=len(dropped_features)
        )

        if validate:
            all_measures_validation, all_labels_validation = dataset_utils.load_dataset(
                dataset_utils.DATASET_VALIDATION)
            features = all_measures_validation.xs(measure, level='measure')
            labels = all_labels_validation.xs(measure, level='measure')
            validation_accuracy = cluster_validate(features, labels, dropped_features, scaler, model)
            clustering.append_to_accuracy_frame(frame=accuracy_frame, accuracy=validation_accuracy,
                                                measure=measure, dataset=dataset, context="validation",
                                                corr_threshold=corr_threshold, dropped=len(dropped_features)
                                                )
        result_map[measure] = ClusterPipelineOutput(model, scaler, dropped_features, accuracy_frame)
    return result_map


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
    ''' Drops features exceeding the correlation threshold permanently from the frame.

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
    config_path = os.path.expanduser(args['--config'])
    import json
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
    results_org = cluster_pipeline(dataset=dataset_utils.DATASET_ORG, validate=True, config=config)
    results_util = cluster_pipeline(dataset=dataset_utils.DATASET_UTIL, validate=True, config=config)

    output = args['--out']
    if output:
        output = os.path.expanduser(output)
        write_header = True
        open_mode = 'w'
        if os.path.isfile(output):
            write_header = False
            open_mode = 'a'
        frames = []
        for key in results_org:
            frames.append(results_org[key].get_accuracy_frame())
        for key in results_util:
            frames.append(results_util[key].get_accuracy_frame())
        pd.concat(frames, ignore_index=True).to_csv(output, header=write_header, mode=open_mode)
elif args['explore']:
    out_explore = os.path.expanduser(args['--explore'])
    out_config = os.path.expanduser(args['--config'])

    if not os.path.isfile(out_explore):
        subprocess.call(['./explore.sh', out_explore])

    config_generator.generate_config_from_exploratoin_file(out_explore, out_config)

else:
    print "UNDEFINED COMMAND LINE ARGUMENTS"
    exit(1)
