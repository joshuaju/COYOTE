import pandas as pd
import numpy as np
from peak import get_peak_series

def run_script():
    dates = [
        "2018-01-01", "2018-01-08", "2018-01-015", "2018-01-23", "2018-01-30",
        "2018-02-01", "2018-02-08", "2018-02-015", "2018-02-23",
        #"2018-03-01", "2018-03-08", "2018-03-015", "2018-03-23", "2018-03-30"
        ]

    s = pd.Series(np.array([2, 0, 0, 5, 5, 5, 4, 3, 8]), dtype=np.int32, index=pd.DatetimeIndex(dates))
    df1 = extract("first", s)
    s = pd.Series(np.array([5, 2, 1, 2, 2, 10, 0, 1, 5]), dtype=np.int32, index=pd.DatetimeIndex(dates))
    df2 = extract("second", s)
    print pd.concat([df1, df2])

def extract(index_name, timeseries):
    assert isinstance(timeseries, pd.Series)

    max_y = timeseries.max()
    peakseries = get_peak_series(timeseries)

    peak_feature_map = __extract_peak_features(timeseries, peakseries, max_y)
    gradient_feature_map = __extract_gradient_features(timeseries)

    feature_map = {
        'duration':timeseries.count(),
        'max_y': max_y,
        'max_y_pos': timeseries.values.argmax() + 1,
        'mean_y': timeseries.mean(),
        'sum_y': timeseries.sum(),
        'q25': timeseries.quantile(q=0.25),
        'q50': timeseries.quantile(q=0.5),
        'q75': timeseries.quantile(q=0.75),
        'std': timeseries.std()

    }
    feature_map.update(peak_feature_map)
    feature_map.update(gradient_feature_map)
    return pd.DataFrame(data=feature_map, index=[index_name])

def __extract_peak_features(timeseries, peakseries, max_y, resample_time = 7):
    df = pd.DataFrame(data={'values': timeseries.values, 'peaks': peakseries.values}, index=timeseries.index)

    #counts
    peak_counts = df.groupby('peaks').count()
    peak_down = peak_counts.loc[-1][0] if -1 in peak_counts.index else 0
    peak_none = peak_counts.loc[0][0] if 0 in peak_counts.index else 0
    peak_up = peak_counts.loc[1][0] if 1 in peak_counts.index else 0

    #time between
    up_times = df[df['peaks'] == 1].index
    delta_ups = [(up_times[idx] - up_times[idx - 1]).days for idx in range(1, len(up_times))]
    atbp_up = (np.average(delta_ups) / resample_time) if len(delta_ups) > 0 else np.NaN

    down_times = df[df['peaks'] == -1].index
    delta_downs = [(down_times[idx] - down_times[idx - 1]).days for idx in range(1, len(down_times))]
    atbp_down = (np.average(delta_downs) / resample_time) if len(delta_downs) > 0 else np.NaN

    #amplitudes
    up_idx = np.where(df.peaks.values == 1)[0]
    vals = df['values'].values
    amplitudes = []
    for idx in up_idx:
        prev_val = vals[idx - 1]
        peak_val = vals[idx]
        diff = peak_val - prev_val
        amplitudes.append(np.true_divide(diff, max_y))
    min_amp, avg_amp, max_amp = (np.NaN, np.NaN, np.NaN)
    if len(amplitudes) > 0:
        min_amp, avg_amp, max_amp = (np.min(amplitudes), np.average(amplitudes), np.max(amplitudes))
    min_amp = min_amp
    avg_amp = avg_amp
    max_amp = max_amp

    # TODO Positive Peak Deviation (AVG(PEAK(i) - mean)
    # TODO Negative Peak Deviation (AVG(VALLEY(i) - mean)


    return {'peak_down': peak_down, 'peak_none': peak_none, 'peak_up': peak_up,
            'atbp_up': atbp_up, 'atbp_down': atbp_down,
            'min_amp': min_amp, 'avg_amp':avg_amp, 'max_amp': max_amp,
            }

def __extract_gradient_features(timeseries):
    gradients = []
    for idx in range(1, len(timeseries)):
        gradients.append(timeseries[idx] - timeseries[idx - 1])
    gradients = np.array(gradients)
    pos_gradients = gradients[np.where(gradients >= 0)]
    neg_gradients = gradients[np.where(gradients < 0)]
    mpg = pos_gradients.mean() if len(pos_gradients) > 0 else np.NaN
    mng = neg_gradients.mean() if len(neg_gradients) > 0 else np.NaN
    # TODO count positive and negative gradients
    # TODO count sequential positive and negative gradients (min, mean, max, total each)
    return {'MPG': mpg, 'MNG':mng}

############
pd.set_option("display.max_rows", 500)
pd.set_option('display.expand_frame_repr', True)

run_script()
