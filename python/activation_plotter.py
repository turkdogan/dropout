import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

colors = ["r", "c", "y", "b", "g", "m", "k", "violet", "lime", "dimgray", "yellow", "tan", "skyblue", "seashell", "seagreen"]
activations = ["Sigmoid", "Tanh", "ReLU"]

def plot_rate_correct_charts(correct_values):

    index = 0

    errors = [100 * (10000 - int(x))/10000.0 for x in correct_values]

    y_pos = np.arange(len(activations))

    plt.barh(y_pos, errors, align='center', alpha=0.5)
    plt.yticks(y_pos, activations)

    plt.xlabel("Misclassified rate (%)")
    plt.savefig("mnist_activations_correct.png")
    plt.close()


def plot_err_rate_charts():
    files = [
        "E_60000_Mnist_act_sigmoid.txt",
        "E_60000_Mnist_act_Tanh.txt",
        "E_60000_Mnist_act_relu.txt",
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

        lbl = str(activations[index])
        ax.set_yscale('log')
        ax.plot(merged_errors[0,:], merged_errors[1,:], label=lbl, color=colors[index])
        index+=1

    legend = ax.legend(loc="upper right", shadow=True)
    plt.ylabel("Cross-entropy error")
    plt.xlabel("Epochs")
    plt.savefig("mnist_activations.png")
    plt.close()

    return correct_values


correct_values = plot_err_rate_charts()
plot_rate_correct_charts(correct_values)
