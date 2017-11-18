import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

colors = ["r", "c", "y", "b", "g", "m", "k", "violet", "lime", "dimgray", "yellow", "tan", "skyblue", "seashell", "seagreen"]
rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

def plot_rate_correct_charts(correct_values):
    # plt.set_yscale('log')

    index = 0

    errors = [100 * (10000 - int(x))/10000.0 for x in correct_values]
    print(errors)

    for rate in rates:
        plt.plot(rates, errors)
    # ax.plot(merged_errors[0,:], merged_errors[1,:], label=lbl, color=colors[index])

    # legend = ax.legend(loc="upper right", shadow=True)
    plt.ylabel("Misclassified rate (%)")
    plt.xlabel("Keep rates")
    plt.savefig("mnist_rates_correct.png")
    plt.close()


def plot_err_rate_charts():
    files = [
        "E_60000_C_0.100000.txt",
        "E_60000_C_0.200000.txt",
        "E_60000_C_0.300000.txt",
        "E_60000_C_0.400000.txt",
        "E_60000_C_0.500000.txt",
        "E_60000_C_0.600000.txt",
        "E_60000_C_0.700000.txt",
        "E_60000_C_0.800000.txt",
        "E_60000_C_0.900000.txt",
    ]

    index = 0

    fig, ax = plt.subplots()

    correct_values = []

    for file_name in files:
        print(file_name)
        f = open("input/"+file_name)
        correct = f.readline()
        correct_values.append(correct)
        lines = f.readlines();
        data = np.loadtxt(lines);
        it_count = data.shape[0]
        merged_errors = np.ndarray(shape=(2, it_count))
        merged_errors[1,:] = data
        merged_errors[0,:] = np.arange(0, it_count)

        lbl = "Keep rate: " + str(rates[index])
        ax.set_yscale('log')
        ax.plot(merged_errors[0,:], merged_errors[1,:], label=lbl, color=colors[index])
        index+=1

    legend = ax.legend(loc="upper right", shadow=True)
    plt.ylabel("Cross-entropy error")
    plt.xlabel("Epochs")
    plt.savefig("mnist_rates.png")
    plt.close()

    return correct_values


correct_values = plot_err_rate_charts()
plot_rate_correct_charts(correct_values)
