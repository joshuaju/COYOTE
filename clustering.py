from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from sklearn import cluster
import pandas as pd


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

    model = cluster.KMeans(n_clusters=2, init='k-means++', n_init=10, max_iter=300).fit(features)
    cluster_labels = model.labels_

    predicted_labels = __convert_to_string_labels__(cluster_labels, true_labels)

    print "* TRAINING"
    p, r, f = __precision_recall_fscore(true_labels, predicted_labels)
    return model, Accuracy(p, r, f)


def validate(model, features, true_labels):
    cluster_labels = model.predict(features)
    predicted_labels = __convert_to_string_labels__(cluster_labels, true_labels)
    print "* VALIDATION"
    p, r, f = __precision_recall_fscore(true_labels, predicted_labels)
    return Accuracy(p, r, f)


def predict(features, model):
    cluster_labels = model.predict(features)
    return cluster_labels


def __precision_recall_fscore(true_labels, predicted_labels, print_to_stdout=True):
    p, r, f, s = precision_recall_fscore_support(y_true=true_labels, y_pred=predicted_labels)
    if print_to_stdout:
        print "Precision(P)  = %.2f" % p[0], "\t", "Recall(P)  = %.2f" % r[0], "\t", "F-Measure(P)  = %.2f" % f[0]
        print "Precision(NP) = %.2f" % p[1], "\t", "Recall(NP) = %.2f" % r[1], "\t", "F-Measure(NP) = %.2f" % f[1]
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

    predicted_labels = pd.Series(predicted_labels)
    # replace the numeric label with the determined string labels
    predicted_labels.replace(to_replace=value_for_first, value=first_label, inplace=True)
    predicted_labels.replace(to_replace=value_for_second, value=second_label, inplace=True)
    return predicted_labels
