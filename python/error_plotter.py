import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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
        "NO_DROPOUT.txt.txt",
        "CONCAVE_DROPOUT_0.550000_0.950000.txt.txt",
        "CONVEX_DROPOUT_0.550000_0.950000.txt.txt",
        # "LINEAR_DROPOUT_0.550000_0.950000.txt.txt",
        "HALF-CONVEX_DROPOUT_0.550000_0.950000.txt.txt",
        "HALF-CONVEX_DROPOUT_0.550000_0.950000.txt.txt",
        "HALF-CONCAVE-DEC_DROPOUT_0.950000_0.550000.txt.txt",
        "HALF-CONVEX-DEC_DROPOUT_0.950000_0.550000.txt.txt",
        "CONSTANT_DROPOUT_0.500000.txt.txt",
        # "CONSTANT_DROPOUT_0.700000.txt",
        "CONSTANT_DROPOUT_0.800000.txt.txt",
        # "CONSTANT_DROPOUT_0.900000.txt"
    ]

    for file_name in files:
        print(file_name)
        for train_size in [400, 1000, 3200, 10000]:
            lines = open("input/E_MNIST_" + str(train_size) + "_" + file_name).readlines();
            data = np.loadtxt(lines);
            it_count = data.shape[0]
            merged_errors = np.ndarray(shape=(2, it_count))
            merged_errors[1,:] = data
            merged_errors[0,:] = np.arange(0, it_count)

            lines = open("input/V_MNIST_" + str(train_size) + "_" + file_name).readlines();
            data = np.loadtxt(lines);
            it_count = data.shape[0]
            merged_validation = np.ndarray(shape=(2, it_count))
            merged_validation[1,:] = data
            merged_validation[0,:] = np.arange(0, it_count)

            plot_map = {}
            plot_map['Training Errors'] = merged_errors
            plot_map['Validation Error'] = merged_validation
            plot_charts(plot_map, str(train_size) + "_" + file_name + ".png", legend_loc="upper right", x_label="Iteratation " + str(train_size) + file_name, y_label = "Cross entropy error (MNIST 1000)")

plot_err_validate_charts()
