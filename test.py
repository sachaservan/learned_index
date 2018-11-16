import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
import pandas as pd
from optparse import OptionParser
import random
import ast
import csv


if __name__ == "__main__": 
    parser = OptionParser()
    parser.add_option("-f", "--file", dest="file", type="str", default="data_2d_corr.csv", help="data file")

    (options, args) = parser.parse_args()
    df = pd.read_csv(options.file)

    ax1 = df.plot.scatter(x='attr1',
        y='attr2',
        c='DarkBlue')
    plt.show()
    if True:

        n = len(df)
        # gen query data
        attr1_bounds = [df['attr1'].min(), df['attr1'].max()]
        attr2_bounds = [df['attr2'].min(), df['attr2'].max()]

        with open('query_data.csv', 'w') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_NONE)
            wr.writerow(["attr1_l", "attr1_u", "attr2_l", "attr2_u", "cnt"])
            for i in range(10000000):
                attr1_filter = sorted([random.uniform(attr1_bounds[0], attr1_bounds[1]), random.uniform(attr1_bounds[0], attr1_bounds[1])])
                attr2_filter = sorted([random.uniform(attr2_bounds[0], attr2_bounds[1]), random.uniform(attr2_bounds[0], attr2_bounds[1])])
                cnt = df[(df['attr1'] >= attr1_filter[0]) & (df['attr1'] < attr1_filter[1]) & (df['attr2'] >= attr2_filter[0]) & (df['attr2'] < attr2_filter[1])].count()
                wr.writerow([attr1_filter[0], attr1_filter[1], attr2_filter[0], attr2_filter[1], cnt[0]])
            
            