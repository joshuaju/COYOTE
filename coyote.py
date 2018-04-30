import feature_extraction
import feature_analysis
import clustering
import dataset_utils

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys, os

def run_script():
    __cluster__()


def __extract__(path_to_timeseries, path_to_featuretable):
    path_to_timeseries = os.path.expanduser(sys.argv[1])
    path_to_featuretable = os.path.expanduser(sys.argv[2])
    assert os.path.isfile(path_to_timeseries)
    assert not os.path.isfile(path_to_featuretable)

    COL_REPO = 0
    COL_DATE = 1
    COL_INTEGRATIONS = 4

    frame = pd.read_csv(
        path_to_timeseries,
        index_col=[COL_REPO, COL_DATE], parse_dates=[COL_DATE],
        usecols=[COL_REPO, COL_DATE, COL_INTEGRATIONS],
        dtype={'integrations': np.float64}).dropna()
    feature_extraction.transform_timeseries_frame_to_featuretable(frame, path_to_featuretable)


def __cluster__():
    CORR_THRESHOLD = 0.85
    print "\n>>>Note: The correlation threshold is %s. Features greater or equal to this are automatically removed.\n" % CORR_THRESHOLD
    # load datasets and labels -----------------------------------------------------------------------------------------
    org_neg_frame, org_neg_labels = dataset_utils.load_org_and_neg_combined_with_labels()
    util_neg_frame, util_neg_labels = dataset_utils.load_util_and_neg_combined_with_labels()
    val_frame, val_labels = dataset_utils.load_validation_combined_with_labels()
    # drop correlating features ----------------------------------------------------------------------------------------
    drop_org_features = feature_analysis.get_correlating_features(org_neg_frame, corr_threshold=CORR_THRESHOLD)
    drop_util_features = feature_analysis.get_correlating_features(util_neg_frame, corr_threshold=CORR_THRESHOLD)
    org_neg_frame.drop(drop_org_features, axis=1, inplace=True)
    util_neg_frame.drop(drop_util_features, axis=1, inplace=True)
    print "Removed %s features from the Org&Neg dataset. There are %s features left:" % (
    len(drop_org_features), len(org_neg_frame.columns))
    print org_neg_frame.columns.values
    print "Removed %s features from the Util&Neg dataset. There are %s features left.\n" % (
    len(drop_util_features), len(util_neg_frame.columns))
    print util_neg_frame.columns.values
    # t-SNE plot
    transformed_org_neg = feature_analysis.tsne(org_neg_frame, n_components=2)
    transformed_util_neg = feature_analysis.tsne(util_neg_frame, n_components=2)

    feature_analysis.scatter_features_in_2d(transformed_org_neg, org_neg_labels, ['x', '.'])
    plt.savefig('plots/tsne_org_neg.png')
    plt.close()
    feature_analysis.scatter_features_in_2d(transformed_util_neg, util_neg_labels, ['x', '.'])
    plt.savefig('plots/tsne_util_neg.png')
    plt.close()
    # Cluster features -------------------------------------------------------------------------------------------------
    print ">>>Org Training"
    org_model = clustering.train(org_neg_frame, org_neg_labels)
    print ">>>Org Validation"
    #clustering.validate(org_model, val_frame.drop(drop_org_features, axis=1), val_labels)
    print
    print ">>>Util Training"
    util_model = clustering.train(util_neg_frame, util_neg_labels)
    print ">>>Util Validation"
    #clustering.validate(util_model, val_frame.drop(drop_util_features, axis=1), val_labels)


pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_columns", 500)
pd.set_option('display.expand_frame_repr', False)
run_script()
