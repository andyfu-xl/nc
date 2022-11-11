import matplotlib.pyplot as plt
import numpy as np
from results import *

# # there are 504 different parameter settings.
parameter_list = []
for c1c2_temp in range(1, 41):
    for w_temp in range(-9, 10):
        c1c2 = c1c2_temp / 10
        w = w_temp / 10
        if c1c2 < (24 * (1 - np.square(w)) / (7 - 5 * w)):
            parameter_list.append((w, c1c2))


def q1_plot():
    cmap = plt.colormaps["plasma_r"]
    fig, ax = plt.subplots()
    x = [x for (_, x) in parameter_list]
    y = [y for (y, _) in parameter_list]
    area = [1 / (a / 1000 + 0.05) for (a, _, _) in results_rastrigin]
    # normalize fitness values
    fit_raw = np.array([fit for (_, fit, _) in results_rastrigin])
    fit_range = max(fit_raw) - min(fit_raw)
    fit_min = min(fit_raw)
    fit = np.array([np.square((f - fit_min) / fit_range) for f in fit_raw])
    sc = ax.scatter(x, y, s=area, c=fit, cmap=cmap)
    plt.colorbar(sc)
    ax.set_title('Rastrigin Function Parameters')
    ax.set_xlabel('alpha1 + alpha2')
    ax.set_ylabel('omega')
    plt.show()

    fig, ax = plt.subplots()
    x = [x for (_, x) in parameter_list]
    y = [y for (y, _) in parameter_list]
    area = [1 / (a / 1000 + 0.05) for (a, _, _) in results_sphere]
    # normalize fitness values
    fit_raw = np.array([fit for (_, fit, _) in results_sphere])
    fit_range = max(fit_raw) - min(fit_raw)
    fit_min = min(fit_raw)
    fit = np.array([np.square((f - fit_min) / fit_range) for f in fit_raw])
    sc = ax.scatter(x, y, s=area, c=fit, cmap=cmap)
    plt.colorbar(sc)
    ax.set_title('Sphere Function Parameters')
    ax.set_xlabel('alpha1 + alpha2')
    ax.set_ylabel('omega')
    plt.show()


def q2_plot():
    fig, ax = plt.subplots()
    # # generate some random test data
    array1 = np.array(q2_result_sphere[:])
    all_data1 = []
    for box in q2_result_sphere[:]:
        temp = []
        for item in box:
            temp.append(item)  # / (array1.max() - array1.min()) + 1)
        all_data1.append(temp)
    n_list = [1, 5, 10, 20, 30, 50, 75, 100, 200, 500][:]
    # plot violin plot
    ax.violinplot(all_data1, showmeans=False, showmedians=True)
    ax.set_title('Fitness vs. Number of Particle \n Sphere Function')
    ax.yaxis.grid(True)
    ax.set_xticks([y + 1 for y in range(len(all_data1))], labels=n_list)
    ax.set_xlabel('Number of particles')
    ax.set_ylabel('Fitness values')
    plt.show()

    fig, ax = plt.subplots()
    # # generate some random test data
    array2 = np.array(q2_result_rastrigin[:])
    all_data2 = []
    for box in q2_result_rastrigin[:]:
        temp = []
        for item in box:
            temp.append(item)  # / (array1.max() - array1.min()) + 1)
        all_data2.append(temp)
    n_list = [1, 5, 10, 20, 30, 50, 75, 100, 200, 500][:]
    # plot violin plot
    ax.violinplot(all_data2, showmeans=False, showmedians=True)
    ax.set_title('Fitness vs. Number of Particle \n Rastrigin Function')
    ax.yaxis.grid(True)
    ax.set_xticks([y + 1 for y in range(len(all_data2))], labels=n_list)
    ax.set_xlabel('Number of particles')
    ax.set_ylabel('Fitness values')
    plt.show()
