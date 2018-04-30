from sklearn.metrics import precision_recall_fscore_support
from sklearn import cluster
from feature_analysis import get_correlating_features
import pandas as pd
import numpy as np
import dataset_utils as ds_util


def __normalise__(features):
    assert isinstance(features, pd.DataFrame)
    return (features - features.mean()) / features.std()


def run_script():
    CORR_THRESHOLD = 0.85
    print "\n>>>Note: The correlation threshold is %s. Features greater or equal to this are automatically removed.\n" % CORR_THRESHOLD
    # load datasets and labels
    org_neg_frame, org_neg_labels = ds_util.load_org_and_neg_combined_with_labels()
    util_neg_frame, util_neg_labels = ds_util.load_util_and_neg_combined_with_labels()
    val_frame, val_labels = ds_util.load_validation_combined_with_labels()
    # drop correlating features
    drop_org_features = get_correlating_features(org_neg_frame, corr_threshold=CORR_THRESHOLD)
    drop_util_features = get_correlating_features(util_neg_frame, corr_threshold=CORR_THRESHOLD)
    org_neg_frame.drop(drop_org_features, axis=1, inplace=True)
    util_neg_frame.drop(drop_util_features, axis=1, inplace=True)
    print "Removed %s features from the Org&Neg dataset. There are %s features left:" % (len(drop_org_features), len(org_neg_frame.columns))
    print org_neg_frame.columns.values
    print "Removed %s features from the Util&Neg dataset. There are %s features left.\n" % (len(drop_util_features), len(util_neg_frame.columns))
    print util_neg_frame.columns.values
    # TODO drop features from scatter plot analysis

    print ">>>Org Training"
    org_model = train(org_neg_frame, org_neg_labels)
    print ">>>Org Validation"
    validate(org_model, val_frame.drop(drop_org_features, axis=1), val_labels)
    print
    print ">>>Util Training"
    util_model = train(util_neg_frame, util_neg_labels)
    print ">>>Util Validation"
    validate(util_model, val_frame.drop(drop_util_features, axis=1), val_labels)


def train(features, true_labels):
    features = __normalise__(features)

    model = cluster.KMeans(n_clusters=2).fit(features.as_matrix())
    cluster_labels = model.labels_

    predicted_labels = __convert_to_string_labels__(cluster_labels, true_labels)

    __precision_recall_fscore(true_labels, predicted_labels)
    return model

def validate(model, features, true_labels):
    features = __normalise__(features)

    cluster_labels = model.predict(features.as_matrix())
    predicted_labels = __convert_to_string_labels__(cluster_labels, true_labels)
    __precision_recall_fscore(true_labels, predicted_labels)


def __precision_recall_fscore(true_labels, predicted_labels, print_to_stdout=True):
    p, r, f, s = precision_recall_fscore_support(y_true=true_labels, y_pred=predicted_labels)
    if print_to_stdout:
        print "Precision(P)  = %.2f" % p[0], "\t", "Recall(P)  = %.2f" % r[0], "\t", "F-Measure(P)  = %.2f" % f[0]
        print "Precision(NP) = %.2f" % p[1], "\t", "Recall(NP) = %.2f" % r[1], "\t", "F-Measure(NP) = %.2f" % f[1]
    return p, r, f

def __convert_to_string_labels__(predicted_labels, true_labels):
    assert len(true_labels.unique()) == 2
    first_label = true_labels.unique()[0]
    second_label = true_labels.unique()[1]

    sum0_labels = (predicted_labels[true_labels == first_label] == 0).sum()
    sum1_labels = (predicted_labels[true_labels == first_label] == 1).sum()

    if sum0_labels >= sum1_labels:
        value_for_first = 0
        value_for_second = 1
    else:
        value_for_first = 1
        value_for_second = 0

    predicted_labels = pd.Series(predicted_labels)
    predicted_labels.replace(to_replace=value_for_first, value=first_label, inplace=True)
    predicted_labels.replace(to_replace=value_for_second, value=second_label, inplace=True)
    return predicted_labels


run_script()
