import csv
from random import randint
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import scatter

rows = []
age = np.random.normal(5, 1, 30000) * 10.0 + 25
salary = np.random.normal(5, 1, 30000) * 200000.0 + 1000
for i in range(30000):
    rows.append([int(age[i]), int(salary[i])])

rows = sorted(rows, key=lambda k: [k[0], k[1]])
with open('data_2d.csv', 'w') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_NONE)
    wr.writerow(["attr1", "attr2", "pos"])
    for i in range(30000):
        wr.writerow([rows[i][0], rows[i][1], i])


age = np.random.normal(5, 1, 30000) 
salary = np.random.normal(5, 1, 30000) 
means = [age.mean(), salary.mean()]  
stds = [age.std() / 3, salary.std() / 3]
corr = 0.8         # correlation
covs = [[stds[0]**2          , stds[0]*stds[1]*corr], 
        [stds[0]*stds[1]*corr,           stds[1]**2]] 

m = np.random.multivariate_normal(means, covs, 30000).T

rows = []
for i in range(30000):
    rows.append([m[0][i] * 10.0, m[1][i] * 10.0])

rows = sorted(rows, key=lambda k: [k[0], k[1]])
with open('data_2d_corr.csv', 'w') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_NONE)
    wr.writerow(["attr1", "attr2"])
    for i in range(30000):
        wr.writerow([rows[i][0], rows[i][1]])