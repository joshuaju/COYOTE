#!/bin/bash

python coyote.py extract timeseries/timeseries_org.csv featuretable/features_org.csv
python coyote.py extract timeseries/timeseries_util.csv featuretable/features_util.csv
python coyote.py extract timeseries/timeseries_neg.csv featuretable/features_neg.csv
python coyote.py extract timeseries/timeseries_val_p.csv featuretable/features_val_p.csv
python coyote.py extract timeseries/timeseries_val_np.csv featuretable/features_val_np.csv



