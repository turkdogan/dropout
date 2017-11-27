import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plot_charts(rates, name, legend_loc = 'lower right', save=True, x_label="", y_label="", size=0):
    fig, ax = plt.subplots()
    ax.text(2, 6, "MNIST training size: " + str(size), fontsize=15)

    # fit = plt.figure();
    for rate_name in rates:
        rate = rates[rate_name]
        ax.plot(rate[0,:], rate[1,:], label=rate_name)
        ax.set_yscale('log')

    legend = ax.legend(loc=legend_loc, shadow=True)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.savefig(name) if save else plt.show()

def plot_err_validate_charts():
    files = [
        "no_drop_overfit.txt",
        "drop05_overfit.txt",
    ]

    for file_name in files:
        print(file_name)
        for train_size in [200, 1000, 2000, 10000, 20000]:
            f = open("input/E_" + str(train_size) + "_Mnist_" + file_name)
            count = f.readline()
            lines = f.readlines();
            data = np.loadtxt(lines);
            it_count = data.shape[0]
            merged_errors = np.ndarray(shape=(2, it_count))
            merged_errors[1,:] = data
            merged_errors[0,:] = np.arange(0, it_count)

            f = open("input/V_" + str(train_size) + "_Mnist_" + file_name)
            count = f.readline()
            lines = f.readlines();
            data = np.loadtxt(lines);
            it_count = data.shape[0]
            merged_validation = np.ndarray(shape=(2, it_count))
            merged_validation[1,:] = data
            merged_validation[0,:] = np.arange(0, it_count)

            plot_map = {}
            plot_map['Training Errors'] = merged_errors
            plot_map['Validation Error'] = merged_validation
            plot_charts(plot_map, str(train_size) + "_" + file_name + ".png", legend_loc="upper right", x_label="Epoch", y_label = "Cross entropy error", size=train_size)

plot_err_validate_charts()
