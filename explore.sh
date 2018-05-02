#!/bin/bash
declare -a steps=(0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1)

#rm featuretable/*.csv
#python coyote.py extract timeseries/timeseries_org.csv featuretable/features_org.csv
#python coyote.py extract timeseries/timeseries_util.csv featuretable/features_util.csv
#python coyote.py extract timeseries/timeseries_neg.csv featuretable/features_neg.csv
#python coyote.py extract timeseries/timeseries_val_p.csv featuretable/features_val_p.csv
#python coyote.py extract timeseries/timeseries_val_np.csv featuretable/features_val_np.csv

for corr in "${steps[@]}"
do
    python coyote.py cluster --corr=$corr --out=$1
done


