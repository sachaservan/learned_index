import csv
import numpy as np

rows = []
age = np.random.normal(5, 1, 30000)
salary = np.random.normal(5, 1, 30000)
for i in range(30000):
    rows.append([int(25 + age[i]*10.0), int(1000.0 + salary[i]*200000.0)])

rows = sorted(rows, key=lambda k: [k[0], k[1]])
with open('data_2d.csv', 'wb') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_NONE)
    wr.writerow(["attr1", "attr2", "pos"])
    for i in range(30000):
        wr.writerow([rows[i][0], rows[i][1], i])
