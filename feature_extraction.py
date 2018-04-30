import pandas as pd
import numpy as np


def transform_timeseries_frame_to_featuretable(frame, path_to_featuretable, measure):
    assert isinstance(frame, pd.DataFrame)
    write_header = True
    for filename, group in frame.groupby(level=0):
        feature_frame = extract(filename, group.reset_index("filename", drop=True)[measure])
        feature_frame.to_csv(path_to_featuretable, mode="a", header=write_header)
        write_header = False


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


def get_peak_series(timeseries, PEAK=1, NONE=0, VALLEY=-1):
    length = len(timeseries)
    # the booleans remember the trend (up or down), when iterating the timeseries
    trend_up = False
    trend_down = False
    # 'peakseries' stores the peak type. Initially, all points are set to NONE-Peaks
    peakseries = pd.Series(np.repeat(NONE, length), dtype=np.int8)
    for idx in range(1, length):
        previous = timeseries[idx - 1]
        current = timeseries[idx]
        if previous < current:
            trend_up = True
            if trend_down:  # if true, the trend shifted from downwards to upwards
                peakseries.at[idx - 1] = VALLEY  # set previous point to be a valley
                trend_down = False
        elif previous > current:
            trend_down = True
            if trend_up:  # if true, the trend shifted from upwards to downwards
                peakseries.at[idx - 1] = PEAK  # set previous point to be a peak
                trend_up = False
    return peakseries


def __extract_peak_features(timeseries, mean_y, max_y, resample_time=7):
    PEAK, NONE, VALLEY = 1, 0, -1
    peakseries = get_peak_series(timeseries, PEAK=PEAK, NONE=NONE, VALLEY=VALLEY)
    df = pd.DataFrame(data={'number': timeseries.values, 'peaks': peakseries.values}, index=timeseries.index)
    # counts -----------------------------------------------------------------------------------------------------------
    peak_counts = df.groupby('peaks').count()
    peak_down, peak_none, peak_up = 0, 0, 0
    if VALLEY in peak_counts.index:
        peak_down = peak_counts.loc[VALLEY][0]
    if NONE in peak_counts.index:
        peak_none = peak_counts.loc[NONE][0]
    if PEAK in peak_counts.index:
        peak_up = peak_counts.loc[PEAK][0]
    # time between -----------------------------------------------------------------------------------------------------
    up_times = df[df['peaks'] == 1].index
    delta_ups = [(up_times[idx] - up_times[idx - 1]).days for idx in range(1, len(up_times))]
    min_tbp_up, avg_tbp_up, max_tbp_up = np.NaN, np.NaN, np.NaN
    if len(delta_ups) > 0:
        min_tbp_up = np.min(delta_ups) / resample_time
        avg_tbp_up = (np.average(delta_ups) / resample_time)
        max_tbp_up = np.max(delta_ups) / resample_time

    down_times = df[df['peaks'] == -1].index
    delta_downs = [(down_times[idx] - down_times[idx - 1]).days for idx in range(1, len(down_times))]
    min_tbp_down, avg_tbp_down, max_tbp_down = np.NaN, np.NaN, np.NaN
    if len(delta_downs) > 0:
        min_tbp_down = np.min(delta_downs) / resample_time
        avg_tbp_down = (np.average(delta_downs) / resample_time)
        max_tbp_down = np.max(delta_downs) / resample_time
    # amplitudes -------------------------------------------------------------------------------------------------------
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
    # deviation from mean -----------------------------------------------------------------------------------------------
    positive_deviations = (df[df.peaks == 1].number - mean_y)
    negative_deviations = (df[df.peaks == -1].number - mean_y)

    min_ppd = positive_deviations.min()
    avg_ppd = positive_deviations.mean()
    max_ppd = positive_deviations.max()

    min_npd = negative_deviations.min()
    avg_npd = negative_deviations.mean()
    max_npd = negative_deviations.max()
    # positve and negative sequence counts (min, mean, max, total each) ------------------------------------------------
    min_ps, mean_ps, max_ps, sum_ps = np.NaN, np.NaN, np.NaN, np.NaN
    min_ns, mean_ns, max_ns, sum_ns = np.NaN, np.NaN, np.NaN, np.NaN
    if True in np.in1d([-1, 1], peakseries):
        pos_counts = []
        neg_counts = []
        counter = 0
        trend_up = None
        for val in peakseries:
            counter = counter + 1
            if val == 1:
                pos_counts.append(counter)
                counter = 0
                trend_up = True
            elif val == -1:
                neg_counts.append(counter)
                counter = 0
                trend_up = False
        neg_counts.append(counter) if trend_up else pos_counts.append(counter)
        pos_counts = np.array(pos_counts)
        neg_counts = np.array(neg_counts)
        min_ps, mean_ps, max_ps, sum_ps = np.min(pos_counts), np.mean(pos_counts), np.max(pos_counts), np.sum(
            pos_counts)
        min_ns, mean_ns, max_ns, sum_ns = np.min(neg_counts), np.mean(neg_counts), np.max(neg_counts), np.sum(
            neg_counts)
    # ------------------------------------------------------------------------------------------------------------------
    return {'peak_down': peak_down, 'peak_none': peak_none, 'peak_up': peak_up,  # peak counts
            'min_TBP_up': min_tbp_up, 'avg_TBP_up': avg_tbp_up, 'max_TBP_up': max_tbp_up,
            'min_TBP_down': min_tbp_down, 'avg_TBP_down': avg_tbp_down, 'max_TBP_down': max_tbp_down,
            'min_amp': min_amp, 'avg_amp': avg_amp, 'max_amp': max_amp,  # amplitures
            'min_PPD': min_ppd, 'avg_PPD': avg_ppd, 'max_PPD': max_ppd,
            'min_NPD': min_npd, 'avg_NPD': avg_npd, 'max_NPD': max_npd,
            # Average Positive/Negative Peak Deviation
            'min_PS': min_ps, 'mean_PS': mean_ps, 'max_PS': max_ps, 'sum_PS': sum_ps,
            'min_NS': min_ns, 'mean_NS': mean_ns, 'max_NS': max_ns, 'sum_NS': sum_ns,
            }


def __extract_gradient_features(timeseries):
    gradients = []
    for idx in range(1, len(timeseries)):
        gradient = timeseries[idx] - timeseries[idx - 1]
        gradients.append(gradient)
    gradients = np.array(gradients)
    pos_gradients = gradients[np.where(gradients >= 0)]
    neg_gradients = gradients[np.where(gradients < 0)]
    # mean positive and negative gradient -------------------------------------------------------------------------------
    min_PG = pos_gradients.min() if len(pos_gradients) > 0 else np.NaN
    avg_PG = pos_gradients.mean() if len(pos_gradients) > 0 else np.NaN
    max_PG = pos_gradients.max() if len(pos_gradients) > 0 else np.NaN
    min_NG = neg_gradients.min() if len(neg_gradients) > 0 else np.NaN
    avg_NG = neg_gradients.mean() if len(neg_gradients) > 0 else np.NaN
    max_NG = neg_gradients.max() if len(neg_gradients) > 0 else np.NaN
    # postive and negative gradient count -------------------------------------------------------------------------------
    pg_count = len(pos_gradients)
    ng_count = len(neg_gradients)
    # ------------------------------------------------------------------------------------------------------------------
    return {'min_PG': min_PG, 'avg_PG': avg_PG, 'max_PG': max_PG,
            'min_NG': min_NG, 'avg_NG': avg_NG, 'max_NG': max_NG,
            'PG_count': pg_count, 'NG_count': ng_count}

