import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from random import randint

import os, fnmatch

def plot_batch_overfit(dataset, scenario_index_map):
    print(scenario_index_map)
    fit = plt.figure();
    plt.xticks([0, 400, 800, 1000, 2000, 3000, 400, 5000, 8000, 10000])

    plot_list = []
    colors = ["r", "c", "y", "b", "g", "m", "k", "violet", "lime", "dimgray", "yellow", "tan", "skyblue", "seashell", "seagreen"]
    scenarios = []
    iterations = 120.0

    for data in dataset:
        label = data['size']
        value = data['overfit'] / iterations
        scenario = data['name'].split('_')[1]
        if scenario not in scenarios:
            plott, = plt.plot(label, value, "o", markersize=15, color=colors[scenario_index_map[scenario]], label=scenario)
            scenarios.append(scenario)
        else:
            plott, = plt.plot(label, value, "o", markersize=15, color=colors[scenario_index_map[scenario]])

    category = dataset[0]['category']
    # plt.ylabel("Average overfit per epoch for all categories")
    plt.ylabel("Average overfit per epoch for category: " + category)
    plt.xlabel("Dataset size")
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
            scenario = experiment['name'].split('_')[1]
            if scenario not in scenarios:
                scenarios[scenario] = index
                index = index + 1

    batch_overfit_data = []
    for category in overfit_data:
        batch_overfit_data.extend(overfit_data[category])
        plot_batch_overfit(overfit_data[category], scenarios)
    # plot_batch_overfit(batch_overfit_data, scenarios)

if __name__ == '__main__':
    plot_overfits()
