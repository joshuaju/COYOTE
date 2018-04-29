import pandas as pd
def load_organisation(path = "featuretable/features_org.csv", label="P"):
    return __load_feature_table_and_labels(path, label)

def load_utility(path = "featuretable/features_util.csv", label="P"):
    return __load_feature_table_and_labels(path, label)

def load_negative_instances(path = "featuretable/features_neg.csv", label="NP"):
    return __load_feature_table_and_labels(path, label)

def load_validation_projects(path = "featuretable(features_val_p", label="P"):
    return __load_feature_table_and_labels(path, label)

def load_validation_projects(path = "featuretable(features_val_np", label="NP"):
    return __load_feature_table_and_labels(path, label)

def load_org_and_neg_combined_with_labels():
    org, org_labels = load_organisation()
    neg, neg_labels = load_negative_instances()
    org_neg_frame = pd.concat([org, neg])
    org_neg_labels = pd.concat([org_labels, neg_labels])
    del org, neg, org_labels, neg_labels
    return org_neg_frame, org_neg_labels

def load_util_and_neg_combined_with_labels():
    util, util_labels = load_utility()
    neg, neg_labels = load_negative_instances()
    util_neg_frame = pd.concat([util, neg])
    util_neg_labels = pd.concat([util_labels, neg_labels])
    del util, neg, util_labels, neg_labels
    return util_neg_frame, util_neg_labels

def __load_feature_table_and_labels(path, label, index_col=0, fillna_with=0):
    features = pd.read_csv(path, index_col=index_col).fillna(fillna_with)
    labels = pd.Series(label, index=features.index)
    return features, labels