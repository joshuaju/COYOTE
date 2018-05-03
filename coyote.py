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
    def __init__(self, model, scaler, dropped_features, accuracy_frame, label_converter, predicted_labels):
        self.model = model
        self.scaler = scaler
        self.dropped_features = dropped_features
        self.accuracy_frame = accuracy_frame
        self.label_converter = label_converter
        self.predicted_labels = predicted_labels

    def get_model(self):
        return self.model

    def get_scaler(self):
        return self.scaler

    def get_dropped_features(self):
        return self.dropped_features

    def get_accuracy_frame(self):
        return self.accuracy_frame

    def get_label_converter(self):
        return self.label_converter

    def get_predicted_labels(self):
        return self.predicted_labels


# ----------------------------------------------------------------------------------------------------------------------

def cluster_train(features, true_labels, corr_threshold):
    assert len(features) == len(true_labels)
    features, dropped_features = __drop_features__(features, corr_threshold)
    scaler = clustering.get_scaler(features)
    scaled_features = clustering.scale_data(features, scaler)
    model, label_converter, accuracy = clustering.train(scaled_features, true_labels)
    return model, scaler, dropped_features, accuracy, label_converter


def cluster_validate(features, true_labels, to_drop, scaler, model, label_converter):
    scaled_features = clustering.scale_data(features.drop(to_drop, axis=1), scaler)
    accuracy = clustering.validate(model, scaled_features, true_labels, label_converter)
    return accuracy


def cluster_predict(features, to_drop, scaler, model, label_converter):
    features = features.drop(to_drop, axis=1)
    scaled_features = clustering.scale_data(features, scaler)
    labels = clustering.predict(scaled_features, model)
    return label_converter.convert_to_strings(labels)


def cluster_pipeline(dataset, validate, config, dataset_to_predict=None):
    assert dataset in [dataset_utils.DATASET_ORG, dataset_utils.DATASET_UTIL]
    assert dataset_to_predict in [None, dataset_utils.DATASET_VALIDATION, dataset_utils.DATASET_18M]
    assert isinstance(validate, bool)

    all_measures, all_labels = dataset_utils.load_dataset(dataset)
    result_map = {}
    measure_names = all_measures.index.levels[1].unique()  # level[1] is the 'measure' column
    for measure in measure_names:
        accuracy_frame = clustering.create_accuracy_frame()

        threshold = config[dataset][measure]
        assert not threshold == None

        features = all_measures.xs(measure, level='measure')
        labels = all_labels.xs(measure, level='measure')
        model, scaler, dropped_features, training_accuracy, label_converter = cluster_train(features, labels, threshold)
        clustering.append_to_accuracy_frame(
            frame=accuracy_frame, accuracy=training_accuracy,
            measure=measure, dataset=dataset, context="training",
            corr_threshold=threshold, dropped=len(dropped_features)
        )

        if validate:
            all_measures_validation, all_labels_validation = dataset_utils.load_dataset(
                dataset_utils.DATASET_VALIDATION)
            features = all_measures_validation.xs(measure, level='measure')
            labels = all_labels_validation.xs(measure, level='measure')
            validation_accuracy = cluster_validate(features, labels, dropped_features, scaler, model, label_converter)
            clustering.append_to_accuracy_frame(frame=accuracy_frame, accuracy=validation_accuracy,
                                                measure=measure, dataset=dataset, context="validation",
                                                corr_threshold=threshold, dropped=len(dropped_features)
                                                )
        predicted_series = None
        if not dataset_to_predict == None:
            print "Loading features for prediction..."
            measures_predict, _ = dataset_utils.load_dataset(dataset_to_predict)
            print "Finished loading features."

            features_predict = measures_predict.xs(measure, level='measure')
            predicted_labels = cluster_predict(features=features_predict, to_drop=dropped_features,
                                     scaler=scaler, model=model,
                                     label_converter=label_converter)
            predicted_series = pd.Series(data=predicted_labels, index=features_predict.index)
        result_map[measure] = ClusterPipelineOutput(model, scaler, dropped_features, accuracy_frame, label_converter, predicted_series)
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
    results_org = cluster_pipeline(dataset=dataset_utils.DATASET_ORG, validate=True, config=config, dataset_to_predict=dataset_utils.DATASET_VALIDATION)
    results_util = cluster_pipeline(dataset=dataset_utils.DATASET_UTIL, validate=True, config=config, dataset_to_predict=dataset_utils.DATASET_VALIDATION)

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
    import json

    out_explore = os.path.expanduser(args['--explore'])
    out_config = os.path.expanduser(args['--config'])

    thresholds = np.arange(0.05, 1.01, 0.05)
    for threshold in thresholds:
        config_map = {'util': {}, 'org': {}}
        for k1 in config_map:
            for k2 in ['merges', 'commits', 'commiters', 'integrations', 'integrators']:
                config_map[k1][k2] = threshold
            with open('.tmp_config.json', 'w') as tmp_config:
                json.dump(config_map, tmp_config)
        subprocess.call(['python', 'coyote.py', 'cluster', '--config=.tmp_config.json', '--out=%s' % out_explore])

    subprocess.call(['rm', '.tmp_config.json'])

    config_generator.generate_config_from_exploratoin_file(out_explore, out_config)

else:
    print "UNDEFINED COMMAND LINE ARGUMENTS"
    exit(1)
