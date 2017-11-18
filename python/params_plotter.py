import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

colors = ["r", "c", "y", "b", "g", "m", "k", "violet", "lime", "dimgray", "yellow", "tan", "skyblue", "seashell", "seagreen"]
params = [10, 100, 300, 800]

def plot_rate_correct_charts(correct_values):
    # plt.set_yscale('log')

    index = 0

    errors = [100 * (10000 - int(x))/10000.0 for x in correct_values]
    print(errors)

    plt.plot(params, errors)
    # ax.plot(merged_errors[0,:], merged_errors[1,:], label=lbl, color=colors[index])

    # legend = ax.legend(loc="upper right", shadow=True)
    plt.ylabel("Misclassified rate (%)")
    plt.xlabel("Parameter count")
    plt.savefig("mnist_params_correct.png")
    plt.close()


def plot_err_params_charts():
    files = [
        "E_60000_10_C0.5_params.txt",
        "E_60000_100_C0.5_params.txt",
        "E_60000_300_C0.5_params.txt",
        "E_60000_800_C0.5_params.txt",
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

        lbl = "Param count: : " + str(params[index])
        index+=1
        ax.set_yscale('log')
        ax.plot(merged_errors[0,:], merged_errors[1,:], label=lbl, color=colors[index])
    legend = ax.legend(loc="lower left", shadow=True)

    plt.ylabel("Cross-entropy error")
    plt.xlabel("Epochs")
    plt.savefig("mnist_params.png")
    plt.close()

    return correct_values


correct_values = plot_err_params_charts()
plot_rate_correct_charts(correct_values)
