import pandas as pd
import numpy as np
from peak import get_peak_series


def run_script():
    dates = [
        "2018-01-01", "2018-01-08", "2018-01-015", "2018-01-23", "2018-01-30",
        "2018-02-01", "2018-02-08", "2018-02-015", "2018-02-23",
        # "2018-03-01", "2018-03-08", "2018-03-015", "2018-03-23", "2018-03-30"
    ]
    s = pd.Series(np.array([0, 5, 6, 7, 8, 8, 6, 6 , 0]), dtype=np.int32, index=pd.DatetimeIndex(dates))
    print extract("the_name", s)


def extract(index_name, timeseries):
    assert isinstance(timeseries, pd.Series)
    max_y = timeseries.max()
    mean_y = timeseries.mean()

    peak_feature_map = __extract_peak_features(timeseries, mean_y=mean_y, max_y=max_y)
    gradient_feature_map = __extract_gradient_features(timeseries)

    feature_map = {
        'duration': timeseries.count(),
        'max_y': max_y,
        'max_y_pos': timeseries.values.argmax() + 1,
        'mean_y': mean_y,
        'sum_y': timeseries.sum(),
        'q25': timeseries.quantile(q=0.25),
        'q50': timeseries.quantile(q=0.5),
        'q75': timeseries.quantile(q=0.75),
        'std': timeseries.std()

    }
    feature_map.update(peak_feature_map)
    feature_map.update(gradient_feature_map)
    return pd.DataFrame(data=feature_map, index=[index_name])


def __extract_peak_features(timeseries, mean_y, max_y, resample_time=7):
    PEAK, NONE, VALLEY = 1, 0, -1
    peakseries = get_peak_series(timeseries, PEAK=PEAK, NONE=NONE, VALLEY=VALLEY)
    df = pd.DataFrame(data={'number': timeseries.values, 'peaks': peakseries.values}, index=timeseries.index)
    # counts
    peak_counts = df.groupby('peaks').count()
    peak_down, peak_none, peak_up = 0, 0, 0
    if VALLEY in peak_counts.index:
        peak_down = peak_counts.loc[VALLEY][0]
    if NONE in peak_counts.index:
        peak_none = peak_counts.loc[NONE][0]
    if PEAK in peak_counts.index:
        peak_up = peak_counts.loc[PEAK][0]
    # time between
    up_times = df[df['peaks'] == 1].index
    delta_ups = [(up_times[idx] - up_times[idx - 1]).days for idx in range(1, len(up_times))]
    atbp_up = np.NaN
    if len(delta_ups) > 0:
        atbp_up = (np.average(delta_ups) / resample_time)

    down_times = df[df['peaks'] == -1].index
    delta_downs = [(down_times[idx] - down_times[idx - 1]).days for idx in range(1, len(down_times))]
    atbp_down = np.NaN
    if len(delta_downs) > 0:
        atbp_down = (np.average(delta_downs) / resample_time)
    # amplitudes
    prev = df.number[0]
    amplitudes = []
    for row in df.values:
        if row[1] == -1:
            prev = row[0]
        elif row[1] == 1 and not prev == None:
            amplitudes.append(np.true_divide(row[0] - prev, max_y))

    min_amp, avg_amp, max_amp = (np.NaN, np.NaN, np.NaN)
    if len(amplitudes) > 0:
        min_amp, avg_amp, max_amp = (np.min(amplitudes), np.average(amplitudes), np.max(amplitudes))
    min_amp = min_amp
    avg_amp = avg_amp
    max_amp = max_amp
    #deviation from mean
    appd = (df[df.peaks == 1].number - mean_y).mean()
    anpd = (df[df.peaks == -1].number - mean_y).mean()
    return {'peak_down': peak_down, 'peak_none': peak_none, 'peak_up': peak_up,
            'atbp_up': atbp_up, 'atbp_down': atbp_down,
            'min_amp': min_amp, 'avg_amp': avg_amp, 'max_amp': max_amp,
            'APPD': appd, 'ANPD': anpd
            }


def __extract_gradient_features(timeseries):
    assert timeseries.count() >= 2
    gradients = []
    for idx in range(1, len(timeseries)):
        gradient = timeseries[idx] - timeseries[idx - 1]
        gradients.append(gradient)
    gradients = np.array(gradients)

    pos_gradients = gradients[np.where(gradients >= 0)]
    neg_gradients = gradients[np.where(gradients < 0)]
    #mean positive and negative gradient
    mpg = pos_gradients.mean() if len(pos_gradients) > 0 else np.NaN
    mng = neg_gradients.mean() if len(neg_gradients) > 0 else np.NaN
    #postive and negative gradient count
    pgc = len(pos_gradients)
    ngc = len(neg_gradients)
    # TODO positve and negative sequences (min, mean, max, total each)
    return {'MPG': mpg, 'MNG': mng,
            'PGC': pgc, 'NGC':ngc}


############
pd.set_option("display.max_rows", 500)
pd.set_option('display.expand_frame_repr', True)

run_script()
