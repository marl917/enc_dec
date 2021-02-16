import argparse
import json
import matplotlib.pyplot as plt
from pathlib import Path
import os

parser = argparse.ArgumentParser('Plot Metrics from existing json file')
parser.add_argument('paths', nargs='+', type=str, help='directory of json file')
args = parser.parse_args()

dirs = list(args.paths)
while len(dirs) > 0:
    path = dirs.pop()
    print(path)
    x=[]
    y=[]
    with open(path) as json_file:
        data = json.load(json_file)
        for k,v in data.items():
            x.append(k)
            y.append(v)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title("FID plot")
    # ax.set_ylim(0, 10)
    plt.plot(x, y)
    for i, j in zip(x, y):
        ax.annotate( "%.2f" % j, xy=(i, j))
    plt.grid()
    path = Path(path)
    fig.savefig(os.path.join(path.parent, 'fid.png'))


