from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from sklearn import cluster
import pandas as pd
import dataset_utils


class Accuracy:
    def __init__(self, precision, recall, fmeasure):
        self.precision = precision
        self.recall = recall
        self.fmeasure = fmeasure

    def get_precision(self):
        return self.precision

    def get_recall(self):
        return self.recall

    def get_fmeasue(self):
        return self.fmeasure


def create_accuracy_frame():
    return pd.DataFrame(
        columns=['measure', 'dataset', 'context', 'threshold', 'label', 'precision', 'recall', 'fmeasure', 'dropped']
    )

def append_to_accuracy_frame(frame, accuracy, measure, dataset, context, corr_threshold, dropped):
    assert isinstance(accuracy, Accuracy)
    assert dataset in [dataset_utils.DATASET_ORG, dataset_utils.DATASET_UTIL]
    assert context in ['training', 'validation']
    assert isinstance(dropped, int)

    frame.loc[len(frame)] = [measure, dataset, context, corr_threshold, "P",
                             accuracy.get_precision()[0], accuracy.get_recall()[0], accuracy.get_fmeasue()[0],
                             dropped]
    frame.loc[len(frame)] = [measure, dataset, context, corr_threshold, "NP",
                             accuracy.get_precision()[1], accuracy.get_recall()[1], accuracy.get_fmeasue()[1],
                             dropped]

def get_scaler(data):
    scaler = StandardScaler()
    scaler = scaler.fit(data)
    return scaler


def scale_data(data, scaler):
    return scaler.transform(data)


def train(features, true_labels):
    # after normalisation there can be NaN values, which have to be removed/replaced.

    # TODO remove
    # nan_columns = features.columns[features.isna().any()].tolist()
    # features = features.drop(nan_columns, axis=1)

    model = cluster.KMeans(n_clusters=2, init='k-means++', n_init=10, max_iter=300, random_state=10).fit(features)
    cluster_labels = model.labels_

    predicted_labels, label_converter = __convert_to_string_labels__(cluster_labels, true_labels)

    p, r, f = __precision_recall_fscore(true_labels, predicted_labels)
    return model, label_converter, Accuracy(p, r, f)


def validate(model, features, true_labels, label_converter):
    cluster_labels = model.predict(features)
    predicted_labels = label_converter.convert_to_strings(cluster_labels)
    #predicted_labels, _ = __convert_to_string_labels__(cluster_labels, true_labels)
    p, r, f = __precision_recall_fscore(true_labels, predicted_labels)
    return Accuracy(p, r, f)


def predict(features, model):
    cluster_labels = model.predict(features)
    return cluster_labels


def __precision_recall_fscore(true_labels, predicted_labels):
    p, r, f, s = precision_recall_fscore_support(y_true=true_labels, y_pred=predicted_labels)
    return p, r, f


def __convert_to_string_labels__(predicted_labels, true_labels):
    '''
        The labels assigned by k-means are numerical. To make them easier to read they are transformed to string labels.

        @predicted_labels: Predicted, numeric labels

        @true_labels: Ground-truth, string labels
    '''
    assert len(true_labels.unique()) == 2
    assert len(predicted_labels) == len(true_labels)

    first_label = true_labels.unique()[0]
    second_label = true_labels.unique()[1]
    sum0_labels = (predicted_labels[
                       true_labels == first_label] == 0).sum()  # number of predicted 0 labels where true label is 'first'
    sum1_labels = (predicted_labels[
                       true_labels == first_label] == 1).sum()  # number of predicted 1 labels where true label is 'second'

    # depending on which sum is greater the labels are assigned.
    if sum0_labels >= sum1_labels:
        value_for_first = 0
        value_for_second = 1
    else:
        value_for_first = 1
        value_for_second = 0

    converter = LabelConverter(num1=value_for_first, string1=first_label, num2=value_for_second, string2=second_label)
    return converter.convert_to_strings(predicted_labels), converter


class LabelConverter:
    def __init__(self, num1, string1, num2, string2):
        self.num1 = num1
        self.string1 = string1
        self.num2 = num2
        self.string2 = string2

    def convert_to_strings(self, numeric_values):
        numeric_series = pd.Series(numeric_values)
        numeric_series = numeric_series.replace(to_replace=self.num1, value=self.string1)
        numeric_series = numeric_series.replace(to_replace=self.num2, value=self.string2)
        return numeric_series

    def convert_to_numbers(self, string_series):
        assert isinstance(string_series, pd.Series)
        string_series = string_series.replace(to_replace=self.string1, value=self.num1)
        string_series = string_series.replace(to_replace=self.string2, value=self.num2)
        return string_series