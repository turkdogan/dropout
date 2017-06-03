import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from random import randint

import os, fnmatch

def plot_weights(data, file_name="", x_label="", y_label="",dim_x=28, dim_y=28, count=100):
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
	        plt.xticks(np.array([]))
	        plt.yticks(np.array([]))
	        index = index + 1

    plt.savefig(file_name + ".png", dpi=100)

def weights(file_name, dim_x=28, dim_y=28):
    f = open(file_name)
    lines = f.readlines();
    data = np.loadtxt(lines)
    print(data.shape)
    plot_weights(data, file_name, dim_x=dim_x, dim_y=dim_y);

def plot_iterations(iteration_errors, name):
    fit = plt.figure();
    for iteration_error in iteration_errors:
        plt.plot(iteration_error[0,:], iteration_error[1,:])
    # plt.show();
    plt.savefig(name)

def plot_charts(rates, name, legend_loc = 'lower right', save=True, x_label="", y_label=""):
    fig, ax = plt.subplots()
    # fit = plt.figure();
    for rate_name in rates:
        rate = rates[rate_name]
        ax.plot(rate[0,:], rate[1,:], label=rate_name)

    legend = ax.legend(loc=legend_loc, shadow=True)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.savefig(name) if save else plt.show()

def plot_err_validate_charts():
    files = [
        "NO_DROPOUT.txt",
        "CONCAVE_DROPOUT_0.550000_0.950000.txt",
        "CONVEX_DROPOUT_0.550000_0.950000.txt",
        "LINEAR_DROPOUT_0.550000_0.950000.txt",
        "CONSTANT_DROPOUT_0.500000.txt",
        "CONSTANT_DROPOUT_0.700000.txt",
        "CONSTANT_DROPOUT_0.800000.txt",
        "CONSTANT_DROPOUT_0.900000.txt"
    ]

    for file_name in files:
        print(file_name)
        for train_size in [10000]:
            lines = open("E_MNIST_" + str(train_size) + "_" + file_name).readlines();
            data = np.loadtxt(lines);
            it_count = data.shape[0]
            merged_errors = np.ndarray(shape=(2, it_count))
            merged_errors[1,:] = data
            merged_errors[0,:] = np.arange(0, it_count)

            lines = open("V_MNIST_" + str(train_size) + "_" + file_name).readlines();
            data = np.loadtxt(lines);
            it_count = data.shape[0]
            merged_validation = np.ndarray(shape=(2, it_count))
            merged_validation[1,:] = data
            merged_validation[0,:] = np.arange(0, it_count)

            plot_map = {}
            plot_map['Errors'] = merged_errors
            plot_map['Validation'] = merged_validation
            plot_charts(plot_map, str(train_size) + "_" + file_name + ".png", legend_loc="upper right", x_label="Iteratation " + str(train_size) + file_name, y_label = "Cross entropy error (MNIST 1000)")


def read_iterations(filename, transpose=False):
    f = open(filename)
    lines = f.readlines();
    if transpose:
        lines = lines.T
    data = np.loadtxt(lines);
    it_count = data.shape[0]
    merged = np.ndarray(shape=(2, it_count))
    merged[1,:] = data
    merged[0,:] = np.arange(0, it_count)
    return merged

def its():
    iterations = {}
    # iterations['Concave'] = read_iterations("MNIST_10000_2_CONVEX_DROPOUT_0.500000_0.990000.txt")
    # iterations['Convex'] = read_iterations("MNIST_10000_2_CONCAVE_DROPOUT_0.500000_0.990000.txt")
    # iterations['Linear'] = read_iterations("MNIST_10000_2_LINEAR_DROPOUT_0.500000_0.990000.txt")
    # iterations['Constant (0.8)'] = read_iterations("MNIST_10000_2_CONSTANT_DROPOUT_0.800000.txt")
    # iterations['Constant (0.5)'] = read_iterations("MNIST_10000_2_CONSTANT_DROPOUT_0.500000.txt")
    # iterations['Constant (0.7)'] = read_iterations("MNIST_10000_2_CONSTANT_DROPOUT_0.700000.txt")
    # iterations['Constant (0.9)'] = read_iterations("MNIST_10000_2_CONSTANT_DROPOUT_0.900000.txt")
    # iterations['No Dropout'] = read_iterations("MNIST_10000_2_NO_DROPOUT.txt")
    plot_charts(iterations, "iterations.png", "upper right", save=False, x_label="iteration",y_label="error")

def iterations():
    iterations = {}
    # iterations['Concave'] = read_iterations("of_concave_dropout_0_5_1_0_0.txt")
    # iterations['Convex'] = read_iterations("of_convex_dropout_0_5_1_0_0.txt")
    # iterations['Linear'] = read_iterations("of_linear_dropout_0_5_1_0_0.txt")
    # iterations['Constant (0.5)'] = read_iterations("dropout_0_5_0.txt")
    # iterations['Constant (0.7)'] = read_iterations("dropout_0_7_0.txt")
    # iterations['Constant (0.9)'] = read_iterations("of_dropout_0_9_0.txt")
    # iterations['No Dropout'] = read_iterations("of_no_dropout_0.txt")
    # plot_charts(iterations, "dropout.png", "upper right", save=False)
    plot_charts(iterations, "dropout.png", "upper right", save=True)

def rates():
    rates = {}

    epoch = 300

    convex_diff = epoch**2
    convex_divider = convex_diff / 0.5
    convex_equation = np.ndarray(shape=(2, epoch))
    convex_equation[1,:] = [0.5 + i**2 / convex_divider for i in range(epoch)]
    convex_equation[0,:] = np.arange(0, epoch)
    rates['Convex increase'] = convex_equation 

    concave_diff = epoch**0.5
    concave_divider = concave_diff / 0.5
    concave_equation = np.ndarray(shape=(2, epoch))
    concave_equation[1,:] = [0.5 + i**0.5 / concave_divider for i in range(epoch)]
    concave_equation[0,:] = np.arange(0, epoch)
    rates['Concave increase'] = concave_equation 

    constant_0_5 = np.ndarray(shape=(2, epoch))
    constant_0_5[1,:] = [0.5 for i in range(epoch)]
    constant_0_5[0,:] = np.arange(0, epoch)
    rates['Constant 0.5'] = constant_0_5

    constant_0_7 = np.ndarray(shape=(2, epoch))
    constant_0_7[1,:] = [0.7 for i in range(epoch)]
    constant_0_7[0,:] = np.arange(0, epoch)
    rates['Constant 0.7'] = constant_0_7

    constant_0_8 = np.ndarray(shape=(2, epoch))
    constant_0_8[1,:] = [0.8 for i in range(epoch)]
    constant_0_8[0,:] = np.arange(0, epoch)
    rates['Constant 0.8'] = constant_0_8

	
    constant_0_9 = np.ndarray(shape=(2, epoch))
    constant_0_9[1,:] = [0.9 for i in range(epoch)]
    constant_0_9[0,:] = np.arange(0, epoch)
    rates['Constant 0.9'] = constant_0_9

    uniform_0_5 = np.ndarray(shape=(2, epoch))
    uniform_0_5[1,:] = [0.5 + i * (0.5 / epoch) for i in range(epoch)]
    uniform_0_5[0,:] = np.arange(0, epoch)
    rates['Uniform increase'] = uniform_0_5

    plot_charts(rates, "rates.png", "lower right", save=False, x_label="epoch",y_label="keep rate")

def plot_equation():
    epoch = 60

    convex_diff = epoch**2
    convex_divider = convex_diff / 0.5
    convex_equation = np.ndarray(shape=(2, epoch))
    convex_equation[1,:] = [0.5 + i**2 / convex_divider for i in range(epoch)]
    convex_equation[0,:] = np.arange(0, epoch)

    concave_diff = epoch**0.5
    concave_divider = concave_diff / 0.5
    concave_equation = np.ndarray(shape=(2, epoch))
    concave_equation[1,:] = [0.5 + i**0.5 / concave_divider for i in range(epoch)]
    concave_equation[0,:] = np.arange(0, epoch)

    linear_diff = epoch
    linear_divider = linear_diff / 0.5
    linear_equation = np.ndarray(shape=(2, epoch))
    linear_equation[1,:] = [0.5 + i / linear_divider for i in range(epoch)]
    linear_equation[0,:] = np.arange(0, epoch)

    charts = {
        "Concave": concave_equation,
        "Convex": convex_equation,
        "Linear": linear_equation
    }
    plot_charts(charts, "rates.png", save=False,x_label="iteration", y_label="Dropout Keep Rate")

def plot_all_weights():
    files = fnmatch.filter(os.listdir('.'), 'W0*.txt')
    for f in files:
        weights(f,dim_x=28,dim_y=28)
    print(files)

def plotOverfit(dataset):
    fit = plt.figure();
    # fit.set_size_inches(8, 10)

    plot_list = []

    # ax1.plot(x, y, color = (0, i / 20.0, 0, 1)

    colors = ["g", "c", "y", "b", "r", "m", "k", "violet"]
    scenarios = []

    # axes = plt.gca()
    # axes.set_xlim([0, 10000])

    plt.xticks([0, 500, 1000, 2000, 3000, 5000, 10000])
    plt.yticks([v/100.0 for v in range(120) if v % 4 == 0])

    first_time = True
    for size in dataset:
        index=0
        for data in dataset[size]:
            plott, = plt.plot([data['size']], [data['overfit']/200.0], "x", color=colors[index])
            index=index+1
            if first_time:
                scenario_name = data['name'].split("_")
                scenario = scenario_name[2]
                if len(scenario_name) > 4:
                    scenario = scenario + "-" + scenario_name[4]
                    if "CONSTANT" not in scenario:
                        scenario = scenario + "-" + scenario_name[5]
                    if scenario.endswith(".txt"):
                        scenario = scenario[:-4]
                else:
                    scenario = scenario + "-" + scenario_name[3][:-4]
                scenarios.append(scenario)
                # print(scenario)
        first_time = False

    plt.ylabel("Average overfit per epoch")
    plt.xlabel("Dataset size")
    # print([dataset[a]['name'].split("_")[2] for a in range(len(dataset))])
    plt.legend(scenarios)
    plt.show();
    # plt.savefig(name)

def plotOverfits(file_name):
    f = open(file_name)
    lines = f.readlines();
    dataset = {}
    for line in lines:
        line_data = line.split(",")
        size = int(line_data[2])

        if size not in dataset:
            dataset[size] = []
        dataset[size].append({"name": line_data[0],
            "trial": int(line_data[1]),
            "size": int(line_data[2]),
            "correct": int(line_data[3]),
            "overfit": float(line_data[4])})
    plotOverfit(dataset)

# weights("W0_MNIST_60000_0_NO_DROPOUT.txt")
#iterations()
#rates()
#plot_equation()
# its()
# plot_all_weights()
# plot_err_validate_charts()
plotOverfits("mnist.txt")
