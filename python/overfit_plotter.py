import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from random import randint
import matplotlib.ticker as plticker

import os, fnmatch

def plot_batch_overfit(dataset, scenario_index_map):
    print(scenario_index_map)

    # fit = plt.figure();
    # plt.xticks([0, 1000])

    plot_list = []
    plt.xticks([-1000, 200, 1000, 2000, 5000, 10000, 20000, 22000], rotation='vertical')

    # plt.xticks(np.arange(-10, 30000, 100))
    axes = plt.gca()
    axes.set_xlim([-1000,22000])
    # axes.set_ylim([ymin,ymax])
    # colors = ["r", "c"]
    colors = ["r", "c", "y", "b", "g", "m", "k", "violet", "lime", "dimgray", "yellow", "tan", "skyblue", "seashell", "seagreen"]
    scenarios = []
    epochs = 120.0

    for data in dataset:
        label = data['size']
        value = data['overfit'] / epochs
        splitted = data['name'].split('_')
        scenario = splitted[2]
        # scenario = splitted[1] + splitted[2]
        print("Scenario: " + scenario + " " + str(label) + " " + str(value))
        if scenario not in scenarios:
            plott, = plt.plot(label, value, "o", markersize=10, color=colors[scenario_index_map[scenario]], label=scenario)
            scenarios.append(scenario)
        else:
            plott, = plt.plot(label, value, "o", markersize=10, color=colors[scenario_index_map[scenario]])

    category = dataset[0]['category']
    plt.ylabel("Standardized overfit")
    # plt.ylabel("Average overfit per epoch for category: " + category)
    plt.xlabel("MNIST Training Dataset size")
    plt.legend(loc=0)
    # plt.legend(scenarios)
    plt.show();

def parse_overfit_data(directory):
    dataset = {}
    file_names = fnmatch.filter(os.listdir(directory), "A_*")
    for file_name in file_names:
        f = open(directory+ "/" + file_name, 'r')
        line = f.readline().rstrip('\n')
        content = line.split(',')
        category = content[1]
        if category not in dataset:
            dataset[category] = []

        dataset[category].append({
            "name": content[0],
            "category": content[1],
            "trial": int(content[2]),
            "size": int(content[3]),
            "correct": int(content[4]),
            "overfit": float(content[5])})

        f.close()

    return dataset


def plot_overfits():
    directory = 'input'
    overfit_data = parse_overfit_data(directory)
    scenarios = {}

    index = 0
    for category in overfit_data:
        for experiment in overfit_data[category]:
            print(experiment['name'])
            splitted = experiment['name'].split('_')
            scenario = splitted[2]
            # scenario = splitted[1] + splitted[2]
            if scenario not in scenarios:
                scenarios[scenario] = index
                index = index + 1

    batch_overfit_data = []
    for category in overfit_data:
        batch_overfit_data.extend(overfit_data[category])
        # plot_batch_overfit(overfit_data[category], scenarios)
    plot_batch_overfit(batch_overfit_data, scenarios)

if __name__ == '__main__':
    plot_overfits()
