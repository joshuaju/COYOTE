#!/bin/bash

printf "\n**************************************\n"
measure="integrations"
echo $measure
rm featuretable/*.csv
python coyote.py extract timeseries/timeseries_org.csv featuretable/features_org.csv --measure=$measure
python coyote.py extract timeseries/timeseries_util.csv featuretable/features_util.csv --measure=$measure
python coyote.py extract timeseries/timeseries_neg.csv featuretable/features_neg.csv --measure=$measure
python coyote.py extract timeseries/timeseries_val_p.csv featuretable/features_val_p.csv --measure=$measure
python coyote.py extract timeseries/timeseries_val_np.csv featuretable/features_val_np.csv --measure=$measure
python coyote.py cluster --corr=0.5
python coyote.py cluster --corr=0.6
python coyote.py cluster --corr=0.7
python coyote.py cluster --corr=0.8
python coyote.py cluster --corr=0.85
python coyote.py cluster --corr=0.9
python coyote.py cluster --corr=0.95

printf "\n**************************************\n"
measure="commits"
echo $measure
rm featuretable/*.csv
python coyote.py extract timeseries/timeseries_org.csv featuretable/features_org.csv --measure=$measure
python coyote.py extract timeseries/timeseries_util.csv featuretable/features_util.csv --measure=$measure
python coyote.py extract timeseries/timeseries_neg.csv featuretable/features_neg.csv --measure=$measure
python coyote.py extract timeseries/timeseries_val_p.csv featuretable/features_val_p.csv --measure=$measure
python coyote.py extract timeseries/timeseries_val_np.csv featuretable/features_val_np.csv --measure=$measure
python coyote.py cluster --corr=0.5
python coyote.py cluster --corr=0.6
python coyote.py cluster --corr=0.7
python coyote.py cluster --corr=0.8
python coyote.py cluster --corr=0.85
python coyote.py cluster --corr=0.9
python coyote.py cluster --corr=0.95

printf "\n**************************************\n"
measure="merges"
echo $measure
rm featuretable/*.csv
python coyote.py extract timeseries/timeseries_org.csv featuretable/features_org.csv --measure=$measure
python coyote.py extract timeseries/timeseries_util.csv featuretable/features_util.csv --measure=$measure
python coyote.py extract timeseries/timeseries_neg.csv featuretable/features_neg.csv --measure=$measure
python coyote.py extract timeseries/timeseries_val_p.csv featuretable/features_val_p.csv --measure=$measure
python coyote.py extract timeseries/timeseries_val_np.csv featuretable/features_val_np.csv --measure=$measure
python coyote.py cluster --corr=0.5
python coyote.py cluster --corr=0.6
python coyote.py cluster --corr=0.7
python coyote.py cluster --corr=0.8
python coyote.py cluster --corr=0.85
python coyote.py cluster --corr=0.9
python coyote.py cluster --corr=0.95

printf "\n**************************************\n"
measure="commiters"
echo $measure
rm featuretable/*.csv
python coyote.py extract timeseries/timeseries_org.csv featuretable/features_org.csv --measure=$measure
python coyote.py extract timeseries/timeseries_util.csv featuretable/features_util.csv --measure=$measure
python coyote.py extract timeseries/timeseries_neg.csv featuretable/features_neg.csv --measure=$measure
python coyote.py extract timeseries/timeseries_val_p.csv featuretable/features_val_p.csv --measure=$measure
python coyote.py extract timeseries/timeseries_val_np.csv featuretable/features_val_np.csv --measure=$measure
python coyote.py cluster --corr=0.5
python coyote.py cluster --corr=0.6
python coyote.py cluster --corr=0.7
python coyote.py cluster --corr=0.8
python coyote.py cluster --corr=0.85
python coyote.py cluster --corr=0.9
python coyote.py cluster --corr=0.95

printf "\n**************************************\n"
measure="integrators"
echo $measure
rm featuretable/*.csv
python coyote.py extract timeseries/timeseries_org.csv featuretable/features_org.csv --measure=$measure
python coyote.py extract timeseries/timeseries_util.csv featuretable/features_util.csv --measure=$measure
python coyote.py extract timeseries/timeseries_neg.csv featuretable/features_neg.csv --measure=$measure
python coyote.py extract timeseries/timeseries_val_p.csv featuretable/features_val_p.csv --measure=$measure
python coyote.py extract timeseries/timeseries_val_np.csv featuretable/features_val_np.csv --measure=$measure
python coyote.py cluster --corr=0.5
python coyote.py cluster --corr=0.6
python coyote.py cluster --corr=0.7
python coyote.py cluster --corr=0.8
python coyote.py cluster --corr=0.85
python coyote.py cluster --corr=0.9
python coyote.py cluster --corr=0.95



