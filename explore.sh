#!/bin/bash
declare -a steps=(0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1)
printf "\n**************************************\n"
measure="integrations"
echo $measure
rm featuretable/*.csv
python coyote.py extract timeseries/timeseries_org.csv featuretable/features_org.csv --measure=$measure
python coyote.py extract timeseries/timeseries_util.csv featuretable/features_util.csv --measure=$measure
python coyote.py extract timeseries/timeseries_neg.csv featuretable/features_neg.csv --measure=$measure
python coyote.py extract timeseries/timeseries_val_p.csv featuretable/features_val_p.csv --measure=$measure
python coyote.py extract timeseries/timeseries_val_np.csv featuretable/features_val_np.csv --measure=$measure
for corr in "${steps[@]}"
do
    python coyote.py cluster --corr=$corr --measure=$measure --out=~/Desktop/accuracy.csv
done

printf "\n**************************************\n"
measure="commits"
echo $measure
rm featuretable/*.csv
python coyote.py extract timeseries/timeseries_org.csv featuretable/features_org.csv --measure=$measure
python coyote.py extract timeseries/timeseries_util.csv featuretable/features_util.csv --measure=$measure
python coyote.py extract timeseries/timeseries_neg.csv featuretable/features_neg.csv --measure=$measure
python coyote.py extract timeseries/timeseries_val_p.csv featuretable/features_val_p.csv --measure=$measure
python coyote.py extract timeseries/timeseries_val_np.csv featuretable/features_val_np.csv --measure=$measure
for corr in "${steps[@]}"
do
    python coyote.py cluster --corr=$corr --measure=$measure --out=~/Desktop/accuracy.csv
done

printf "\n**************************************\n"
measure="merges"
echo $measure
rm featuretable/*.csv
python coyote.py extract timeseries/timeseries_org.csv featuretable/features_org.csv --measure=$measure
python coyote.py extract timeseries/timeseries_util.csv featuretable/features_util.csv --measure=$measure
python coyote.py extract timeseries/timeseries_neg.csv featuretable/features_neg.csv --measure=$measure
python coyote.py extract timeseries/timeseries_val_p.csv featuretable/features_val_p.csv --measure=$measure
python coyote.py extract timeseries/timeseries_val_np.csv featuretable/features_val_np.csv --measure=$measure
for corr in "${steps[@]}"
do
    python coyote.py cluster --corr=$corr --measure=$measure --out=~/Desktop/accuracy.csv
done

printf "\n**************************************\n"
measure="commiters"
echo $measure
rm featuretable/*.csv
python coyote.py extract timeseries/timeseries_org.csv featuretable/features_org.csv --measure=$measure
python coyote.py extract timeseries/timeseries_util.csv featuretable/features_util.csv --measure=$measure
python coyote.py extract timeseries/timeseries_neg.csv featuretable/features_neg.csv --measure=$measure
python coyote.py extract timeseries/timeseries_val_p.csv featuretable/features_val_p.csv --measure=$measure
python coyote.py extract timeseries/timeseries_val_np.csv featuretable/features_val_np.csv --measure=$measure
for corr in "${steps[@]}"
do
    python coyote.py cluster --corr=$corr --measure=$measure --out=~/Desktop/accuracy.csv
done

printf "\n**************************************\n"
measure="integrators"
echo $measure
rm featuretable/*.csv
python coyote.py extract timeseries/timeseries_org.csv featuretable/features_org.csv --measure=$measure
python coyote.py extract timeseries/timeseries_util.csv featuretable/features_util.csv --measure=$measure
python coyote.py extract timeseries/timeseries_neg.csv featuretable/features_neg.csv --measure=$measure
python coyote.py extract timeseries/timeseries_val_p.csv featuretable/features_val_p.csv --measure=$measure
python coyote.py extract timeseries/timeseries_val_np.csv featuretable/features_val_np.csv --measure=$measure
for corr in "${steps[@]}"
do
    python coyote.py cluster --corr=$corr --measure=$measure --out=~/Desktop/accuracy.csv
done



