import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

colors = ["r", "c", "y", "b", "g", "m", "k", "violet", "lime", "dimgray", "yellow", "tan", "skyblue", "seashell", "seagreen"]

def plot_rate_correct_charts(correct_values):
    # plt.set_yscale('log')
    epochs = [10, 20, 30, 100, 200, 400, 800]

    print(correct_values)

    index = 0

    errors = [100 * (10000 - int(x))/10000.0 for x in correct_values]
    print(errors)

    plt.plot(epochs, errors)
    # ax.plot(merged_errors[0,:], merged_errors[1,:], label=lbl, color=colors[index])

    # legend = ax.legend(loc="upper right", shadow=True)
    plt.ylabel("Misclassified rate (%)")
    plt.xlabel("Keep rates")
    plt.savefig("mnist_epochs_correct.png")
    plt.close()


def get_err_epoch_correct():
    correct_values = []
    epochs = [10, 20, 30, 100, 200, 400, 800]
    for epoch in epochs:
        f = open("input/E_60000_C0.5_" + str(epoch)+".txt")
        correct_values.append(f.readline().strip())

    return correct_values


correct_values = get_err_epoch_correct()
plot_rate_correct_charts(correct_values)
