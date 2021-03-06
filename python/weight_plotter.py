import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from random import randint

import os, fnmatch

def plot_weight_chart(data, file_name="", x_label="", y_label="",dim_x=28, dim_y=28, count=100):
    vmin, vmax = data.min(), data.max()

    fig = plt.figure()
    # fig.suptitle(file_name, fontsize=14, fontweight='bold')
    plt.ylabel(y_label)
    plt.xlabel(x_label)

    fig = matplotlib.pyplot.gcf()

    row_count = (int(count/10))
    if row_count > 10:
        row_count = 10

    fig.set_size_inches(row_count, 10)

    index = 1
    for x in range(row_count):
	    for y in range(10):
	        ax = fig.add_subplot(row_count, 10, index)
	        ax.matshow(data[0:,index-1].reshape(dim_x, dim_y),
                       cmap=plt.cm.gray, vmin=.5 * vmin, vmax=.5 * vmax)
	        plt.yticks(np.array([]))
	        plt.xticks(np.array([]))
	        index = index + 1

    plt.savefig("output/" + file_name + ".png", dpi=100)
    plt.close()

def plot_weights(file_name, dim__x=28, dim__y=28, _x_label=""):
    f = open("input/" + file_name)
    lines = f.readlines();
    data = np.loadtxt(lines)
    plot_weight_chart(data, file_name, x_label=_x_label, dim_x=dim__x, dim_y=dim__y);

def plot_all_weights(weight_file_pattern):
    files = fnmatch.filter(os.listdir('input'), weight_file_pattern)
    hardcode_scenarios = ["No dropout", "Dropout with 0.5", "Dropout with 0.7"]
    index = 0
    for f in files:
        print(f)
        plot_weights(f, 28, 28, hardcode_scenarios[index])
        index+=1
    print(files)

plot_all_weights("W0_60000*.txt")
# plot_weights("W0_MNIST_3000_CONCAVE_DEC_DROPOUT_1.000000_0.500000.txt")
