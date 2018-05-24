import pandas as pd
import numpy as np

def print_stats(frame, dataset):
    print "***", dataset

    frame = frame[frame.index == dataset]

    project = frame[frame.commits == 'P']

    num_total = frame.count()[0]
    num_p = project.count()[0]
    print "%s / %s = %.2f" % (num_p, num_total, np.true_divide(num_p, num_total))

path_to_prediction_file = "/home/joshua/Desktop/largescale/predictions.csv"
df = pd.read_csv(path_to_prediction_file, index_col=[0,6], usecols=[1, 2, 3, 4, 5, 6, 7])

columns = df.columns

print "Measure        &\tDataset   &\t\\#P    &\tTotal    &\tPercentage \\\\"
for col in columns:
    for key, group in df[col].groupby(level='dataset'):
        num_total = group.count()
        num_projects = group[group == 'P'].count()
        percentage = np.true_divide(num_projects, num_total)
        print "%s&\t%s&\t%s&\t%s&\t%.2f\\\\" % (col.ljust(15), key.ljust(10), num_projects, num_total, percentage)


#print_stats(df, 'org')
#print_stats(df, 'util')

