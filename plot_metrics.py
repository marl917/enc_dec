import argparse
import json
import matplotlib.pyplot as plt
from pathlib import Path
import os
import numpy as np

parser = argparse.ArgumentParser('Plot Metrics from existing json file')
parser.add_argument('paths', nargs='+', type=str, help='directory of json file')
args = parser.parse_args()

dirs = list(args.paths)
while len(dirs) > 0:
    path = dirs.pop()
    print(path)
    with open(path) as json_file:
        data = json.load(json_file)
        data = {int(k) : v for k, v in data.items()}
        lists = sorted(data.items())  # sorted by key, return a list of tuples
        
        x, y = zip(*lists)  # unpack a list of pairs into two tuples

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title("FID plot")
    ax.set_ylim(0, 64 )
    plt.plot(x, y)
    c=0
    for i, j in zip(x, y):
        print(i,j)
        
            
        ax.annotate( "%.2f" % j, xy=(i, j-0.8+(-1)**c))
        c+=1
    plt.grid()
    path = Path(path)
    fig.savefig(os.path.join(path.parent, 'fid.png'))


