import pandas as pd
import numpy as np

def run_script():
    s = pd.Series([1, 2, 3, 4, 2, 0])
    print s[0]

def extract(name, timeseries):
    assert isinstance(timeseries, pd.Series)

    duration = len(timeseries)
    max_y = timeseries.max()
    max_y_pos = timeseries.values.argmax() + 1
    mean_y = timeseries.mean()
    median_y = timeseries.median()
    sum_y = timeseries.sum()

    peakseries = __peak_analysis__(timeseries)
    peak_feature_map = __extract_peak_features(timeseries, peakseries, max_y)
    gradient_feature_map = __extract_gradient_features(timeseries)

    feature_map = {
        'duration':duration,
        'max_y': max_y,
        'max_y_pos': max_y_pos,
        'mean_y': mean_y,
        'median_y': median_y,
        'sum_y': sum_y,
    }
    feature_map.update(peak_feature_map)
    feature_map.update(gradient_feature_map)

    # TODO return data frame with name as index


def __peak_analysis__(timeseries):
    pass # TODO

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
    return {'MPG': mpg, 'MNG':mng}

############
run_script()
