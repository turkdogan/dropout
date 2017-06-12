import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from random import randint

import os, fnmatch

def plot_batch_overfit(dataset):

    fit = plt.figure();
    # fit.set_size_inches(8, 10)

    plot_list = []

    # ax1.plot(x, y, color = (0, i / 20.0, 0, 1)

    colors = ["r", "c", "y", "b", "g", "m", "k", "violet", "lime", "dimgray", "yellow"]
    scenarios = []

    # axes = plt.gca()
    # axes.set_xlim([0, 10000])

    # plt.xticks([0, 500, 1000, 2000, 3000, 5000, 10000])
    # plt.yticks([v/100.0 for v in range(120) if v % 4 == 0])

    first_time = True
    for size in dataset:
        index=0
        for data in dataset[size]:
            plott, = plt.plot([data['size']], [data['overfit']/120.0], "x", color=colors[index])
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

def plotOverfit(dataset):

    plot_list = []

    # ax1.plot(x, y, color = (0, i / 20.0, 0, 1)

    colors = ["g", "c", "y", "b", "r", "m", "k", "violet"]
    scenarios = []

    # axes = plt.gca()
    # axes.set_xlim([0, 10000])

    # plt.xticks([0, 500, 1000, 2000, 3000, 5000, 10000])
    # plt.yticks([v/100.0 for v in range(120) if v % 4 == 0])

    first_time = True
    for size in dataset:
        index=0
        fit = plt.figure();
        fit.set_size_inches(12, 4)
        for data in dataset[size]:
            plott, = plt.plot([data['size']], [data['overfit']/120.0], "x", color=colors[index])
            plt.ylabel("Average overfit per epoch")
            plt.xlabel("Dataset size")

            # plott, = plt.plot([data['size']], [10000-data['correct']], "x", color=colors[index])
            # plt.ylabel("Misclassified sample size of 10000")
            # plt.xlabel("Dataset size")
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
        plt.legend(scenarios)
        plt.savefig("output/" + str(size) +"-overfit.png")
        # plt.savefig(str(size) +"-misclassified.png")
        first_time = False

    # plt.ylabel("Average overfit per epoch")
    # plt.xlabel("Dataset size")
    # print([dataset[a]['name'].split("_")[2] for a in range(len(dataset))])
    # plt.legend(scenarios)
    # plt.show();
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
    plot_batch_overfit(dataset)

plotOverfits("input/mnist.txt")
