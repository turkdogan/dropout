import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

colors = ["r", "c", "y", "b", "g", "m", "k", "violet", "lime", "dimgray", "yellow", "tan", "skyblue", "seashell", "seagreen"]
params = [#"No Dropout",
          "No Dropout",
          "Constant Dropout with 0.5",
          "DropGrad",
]
# params = [10, 100, 300, 800]

def plot_err_mnist_charts():
    files = [
        "E_50000_Mnist_no_grad.txt",
        "E_50000_Mnist_grad_dropout05.txt",
        "E_50000_Mnist_grad.txt"
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
    legend = ax.legend(loc="upper right", shadow=True)
    # legend = ax.legend(loc="upper right", shadow=True, prop={'size':6})

    plt.ylabel("Cross-entropy error")
    plt.xlabel("MNIST Epochs")
    plt.savefig("mnist_drop_grad.png")
    plt.close()

    return correct_values


correct_values = plot_err_mnist_charts()
