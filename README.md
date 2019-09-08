# COYOTE
This tool is part of the master thesis "Clustering Software Projects at Large-scale using Time-Series". COYOTE is a clustering tool, that works on time-series created from git history logs. 

# Setup

## Install dependencies
sudo apt-get install python2.7 python-pip python-dev build-essential python-tk python-setuptools

## Install python modules with pip2
pip install pandas 
sudo pip install --upgrade setuptools
sudo pip install matplotlib
sudo pip install sklearn
sudo pip install docopt

## Required files

COYOTE requires some files to work. These files contain data collected during the thesis and are not published in this git repository. You may contact the authors to receive these files.

Make sure to copy the feature tables to your directory.

    * "featuretable/features_18M.csv" (large dataset)
    * "featuretable/features_org.csv" (organisation dataset)
    * "featuretable/features_util.csv" (utility dataset)
    * "featuretable/features_neg.csv" (negative instances dataset)
    * "featuretable/features_val_p.csv" (well-engineered project of the validation dataset)
    * "featuretable/features_val_np.csv" (not well-engineered project of the validation dataset)

You may changes the paths to these files in dataset_utils.py.
    
# Usage
Open a terminal in the root directory of COYOTE. Executing

    python coyote.py

will show you your options.

## COYOTE extract

Extract a featuretable from a timeseries. Executing the command

    python coyote.py ./timeseries.csv ./features.csv

will read timeseries (as created by RHINO) and transform them to features.

## COYOTE cluster

Clusters projects (represented by feature vectors) into two classes: P (Project) and NP (Non-Project). Executing

    python coyote.py cluster --config=./cfg.json --accuracy_file=./acc.csv --prediction_file=./pred.csv

will 
    1. train COYOTE using the organisation, utility and negative instances feature tables, 
    2. validate COYOTE against the validation feature tables
    3. predict the labels of the large dataset 
The result of step 2 are saved to the accuracy file, the result of step 3 to the prediction_file.

NOTE: Predicting the large dataset required more than 8GB of RAM (i.e. my Laptop has 8GB and will exit with a memory error).

### What is the configuration file?
COYOTE analyses the features from the feature tables before using them for training. The configuration file states correlation thresholds for the measures on the training datasets. Any feature violating the correlation threshold will not be used for training.

### Example configuration
Below you will find an example configuration. This configuration was created using COYOTE explore.
    
    {"util": {"merges": 0.85, "commits": 0.95, "commiters": 0.75, "integrations": 0.95, "integrators": 0.45}, "org": {"merges": 0.9, "commits": 0.9, "commiters": 0.75, "integrations": 0.9, "integrators": 0.75}}

You may save it and use it to cluster

### How to predict other datasets?
    1. use RHINO to create timeseries of the dataset.
    2. extract the features from the timeseries using COYOTE extract.

COYOTE is hard-coded against the large dataset. However, you may want to tweak the program to predict another dataset. 

    3. update dataset_utils.py

Find the following section in "dataset_utils.py" 

    def load_18M():
        features, _ = __load_feature_table_and_labels("featuretable/features_18M.csv", label="-")
        return features

and update the path to the feature table of your dataset, for example like this

    def load_18M():
        features, _ = __load_feature_table_and_labels("featuretable/my_dataset.csv", label="-")
        return features


## COYOTE explore

Explores a range of correlation thresholds to find the best threshold configuration. Executing

    python coyote.py --explore./exp.csv --config=./cfg.json

will
    1. train COYOTE several times with different thresholds
    2. store the accuracy (precision, recall, f-measure) of all trained classifiers in exp.csv
    3. determine the best configuration an save it in cfg.json


# Workflow example

1. Clone COYOTE
    
    git clone https://github.com/joshuaju/COYOTE.git

2. Go into directory 

    cd COYOTE
    mkdir featuretable

3. Place the required files in "featuretable" directory

4. Extract features from timeseries

    python coyote extract ./timeseries/example_ts.csv ./featuretable/example_ft.csv

5. Explore measures to find the best configuration

    python coyote.py --explore./exp.csv --config=./cfg.json

6. Analyse exp.csv to see all results

7. Cluster the large dataset with your configuration. You may change "dataset_utils.py" to point to your own dataset. See "How to predict other datasets?" above.

    python coyote.py cluster --config=cfg.json --accuracy_file=./acc.csv --prediction_file=./pred.csv

8. Analyse acc.csv to see the accuracy of the trained classifier

9. Analyse pred.csv to see how COYOTE classified the projects



