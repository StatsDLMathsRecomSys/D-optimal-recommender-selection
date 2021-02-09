import json 
import sys
from matplotlib import pyplot as plt

fpath = sys.argv[1]
f = open(fpath)
record = json.load(f)

plt.plot(record['epoch'], record['cumu_recall'])
plt.savefig(f'{record["bandit_algorithm"]}_plt.jpg')