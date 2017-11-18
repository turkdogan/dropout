import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

colors = ["r", "c", "y", "b", "g", "m", "k", "violet", "lime", "dimgray", "yellow", "tan", "skyblue", "seashell", "seagreen"]
params = [#"No Dropout",
          "Linearly Incresing",
          "Linearly Decreasing",
          "Concave Incresing",
          "Concave Decreasing",
          "Convex Incresing",
          "Convex Decreasing",
          # "Half Concave Incresing",
          # "Half Concave Decreasing",
          # "Half Concave Incresing",
          # "Hal Concave Decreasing",
]
# params = [10, 100, 300, 800]

min_params = [
          "lin inc",
          "lin dec",
          "cnv inc",
          "cnv dec",
          "cnvx inc",
          "cnvx dec",
]

def plot_rate_correct_charts(correct_values):


    index = 0

    errors = [100 * (10000 - int(x))/10000.0 for x in correct_values]

    y_pos = np.arange(len(min_params))

    plt.barh(y_pos, errors, align='center', alpha=0.5)
    plt.yticks(y_pos, min_params)


    # plt.plot(params, errors)

    # legend = ax.legend(loc="upper right", shadow=True)
    plt.xlabel("Misclassified rate (%)")
    plt.savefig("mnist_dynamic_correct.png")
    plt.close()


def plot_err_dynamic_charts():
    files = [
        # "E_60000_NO-DROPOUT.txt",
        "E_60000_L055-095.txt",
        "E_60000_L095-055.txt",
        "E_60000_Concave055-095.txt",
        "E_60000_Concave095-055.txt",
        "E_60000_Convex055-095.txt",
        "E_60000_Convex095-055.txt",
        # "E_60000_HConcave055-095.txt",
        # "E_60000_HConcave095-055.txt",
        # "E_60000_HConvex055-095.txt",
        # "E_60000_HConvex095-055.txt",
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

        lbl = params[index]
        index+=1
        ax.set_yscale('log')
        ax.plot(merged_errors[0,:], merged_errors[1,:], label=lbl, color=colors[index])
    legend = ax.legend(loc="upper right", shadow=True, prop={'size':6})

    plt.ylabel("Cross-entropy error")
    plt.xlabel("Epochs")
    plt.savefig("mnist_dynamic.png")
    plt.close()

    return correct_values


correct_values = plot_err_dynamic_charts()
plot_rate_correct_charts(correct_values)
