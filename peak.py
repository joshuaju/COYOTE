import pandas as pd
import numpy as np

def get_peak_series(timeseries, PEAK=1, NONE=0, VALLEY=-1):
    length = len(timeseries)
    # the booleans remember the trend (up or down), when iterating the timeseries
    trend_up = False
    trend_down = False
    # 'peakseries' stores the peak type. Initially, all points are set to NONE-Peaks
    peakseries = pd.Series(np.repeat(NONE, length), dtype=np.int8)
    for idx in range(1, length):
        previous = timeseries[idx-1]
        current = timeseries[idx]
        if previous < current:
            trend_up = True
            if trend_down: # if true, the trend shifted from downwards to upwards
                peakseries.at[idx-1] = VALLEY # set previous point to be a valley
                trend_down = False
        elif previous > current:
            trend_down = True
            if trend_up: # if true, the trend shifted from upwards to downwards
                peakseries.at[idx-1] = PEAK # set previous point to be a peak
                trend_up = False
    return peakseries