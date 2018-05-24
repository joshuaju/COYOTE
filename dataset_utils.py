import pandas as pd

DATASET_ORG = 'org'
DATASET_UTIL = 'util'
DATASET_VALIDATION = 'val'
DATASET_18M = 'data18M'

def load_dataset(dataset):
    assert dataset in [DATASET_ORG, DATASET_UTIL, DATASET_VALIDATION, DATASET_18M]
    if dataset == DATASET_ORG:
        return load_org_and_neg_combined_with_labels()
    elif dataset == DATASET_UTIL:
        return load_util_and_neg_combined_with_labels()
    elif dataset == DATASET_VALIDATION:
        return load_validation_combined_with_labels()
    elif dataset == DATASET_18M:
        return load_18M(), None

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


def load_validation_combined_with_labels():
    val_p, val_p_labels = load_validation_projects()
    val_np, val_np_labels = load_validation_non_projects()
    val_frame = pd.concat([val_p, val_np])
    val_labels = pd.concat([val_p_labels, val_np_labels])
    del val_p, val_np, val_p_labels, val_np_labels
    return val_frame, val_labels

def load_18M():
    features, _ = __load_feature_table_and_labels("featuretable/features_18M.csv", label="-")
    return features

def load_organisation(path="featuretable/features_org.csv", label="P"):
    return __load_feature_table_and_labels(path, label)


def load_utility(path="featuretable/features_util.csv", label="P"):
    return __load_feature_table_and_labels(path, label)


def load_negative_instances(path="featuretable/features_neg.csv", label="NP"):
    return __load_feature_table_and_labels(path, label)


def load_validation_projects(path="featuretable/features_val_p.csv", label="P"):
    return __load_feature_table_and_labels(path, label)


def load_validation_non_projects(path="featuretable/features_val_np.csv", label="NP"):
    return __load_feature_table_and_labels(path, label)


def __load_feature_table_and_labels(path, label, index_col=[0, 1], fillna_with=0):
    features = pd.read_csv(path, index_col=index_col).fillna(fillna_with)
    labels = pd.Series(label, index=features.index)
    return features, labels
