import numpy as np
import random
import time

from phso import HPSO
from results import *

X_MIN = -5.12
X_MAX = 5.12


class PSO:
    def __init__(self, dimension, parameters, fitness_function, terminate, bound, max_iter=float('inf'),
                 number_particles=20, max_time=float('inf')):
        # I decided to terminate the search base the number of iterations with no change in the global best.
        # Other options like number of iterations are also feasible.
        self.terminate = terminate
        self.omega, self.alpha1, self.alpha2 = parameters
        # this should be a function
        self.fitness = fitness_function
        # bound should be 2D array in shape (2, dimension)
        self.x_min, self.x_max = bound
        self.number_particles = number_particles
        self.dimension = dimension
        self.iteration = 0
        self.max_iteration = max_iter
        self.time_start = time.time()
        self.max_time = max_time

        self.x_positions = np.zeros((self.number_particles, self.dimension))
        self.x_velocities = np.zeros((self.number_particles, self.dimension))
        self.x_bests = np.zeros((self.number_particles, self.dimension))
        self.global_best = np.zeros((1, self.dimension))

        max_fitness = float("-inf")
        # initializing particle positions
        for x in range(self.number_particles):
            for d in range(self.dimension):
                self.x_positions[x][d] = random.uniform(self.x_min[d], self.x_max[d])
            self.x_bests[x] = self.x_positions[x]
            temp_fitness = self.fitness(self.x_positions[x])
            if temp_fitness > max_fitness:
                self.global_best = self.x_positions[x]
                max_fitness = temp_fitness

        self.best_ever = self.global_best

    # asynchronous update
    def update(self):
        for x in range(self.number_particles):
            # update particle velocity
            self.x_velocities[x] = (self.omega * self.x_velocities[x] + self.alpha1 * random.uniform(0, 1) *
                                    (self.x_bests[x] - self.x_positions[x]) + self.alpha2 * random.uniform(0, 1) *
                                    (self.global_best - self.x_positions[x]))
            # update particle position
            self.x_positions[x] = self.x_positions[x] + self.x_velocities[x]
            for d in range(self.dimension):
                self.x_positions[x][d] = max(self.x_positions[x][d], self.x_min[d])
                self.x_positions[x][d] = min(self.x_positions[x][d], self.x_max[d])
            # update particle best
            if self.fitness(self.x_positions[x]) > self.fitness(self.x_bests[x]):
                self.x_bests[x] = self.x_positions[x]
            # update global best
            if self.fitness(self.x_positions[x]) > self.fitness(self.global_best):
                self.global_best = self.x_positions[x]
        # update best_ever, note that best_ever doesn't affect the search
        if self.fitness(self.best_ever) < self.fitness(self.global_best):
            self.best_ever = self.global_best

    def main(self):
        counter = 0
        previous_best = self.best_ever
        while (counter <= self.terminate) and (self.iteration < self.max_iteration) and \
                ((time.time() - self.time_start) < self.max_time):
            self.update()
            self.iteration += 1
            if self.fitness(previous_best) == self.fitness(self.best_ever):
                counter += 1
            else:
                previous_best = self.best_ever
                counter = 0
        return self.best_ever, self.fitness(self.best_ever), self.iteration



# input x should be a np.array
def fit1(x):
    return -sum(x ** 2)


def fit2(x):
    return -((10 * len(x)) + sum([(xi ** 2 - 10 * np.cos(2 * np.pi * xi)) for xi in x]))


def q1():
    # there are 504 different parameter settings.
    parameter_list = []
    for c1c2_temp in range(1, 41):
        for w_temp in range(-9, 10):
            c1c2 = c1c2_temp / 10
            w = w_temp / 10
            if c1c2 < (24 * (1 - np.square(w)) / (7 - 5 * w)):
                parameter_list.append((w, c1c2 / 2, c1c2 / 2))
    results = []
    for index in range(len(parameter_list)):
        dim = 6
        space = [[X_MIN for _ in range(dim)], [X_MAX for _ in range(dim)]]
        fit_sum = 0
        iter_sum = 0
        for i in range(5):
            # change fit1 to fit2 to run experiment for rastrigin function
            p1 = PSO(dim, parameter_list[index], fit1, 10, space, 1000)
            best, fit, iters = p1.main()
            fit_sum += fit
            iter_sum += iters
        results.append((iter_sum / 5, fit_sum / 5, index))
        print(index)
    print(results)


def q2():
    # there are 504 different parameter settings.
    parameter_list = []
    for c1c2_temp in range(1, 41):
        for w_temp in range(-9, 10):
            c1c2 = c1c2_temp / 10
            w = w_temp / 10
            if c1c2 < (24 * (1 - np.square(w)) / (7 - 5 * w)):
                parameter_list.append((w, c1c2 / 2, c1c2 / 2))
    top_parameters_sphere = [parameter_list[index] for (iters, fit, index) in
                             (sorted(results_sphere, key=lambda r: r[1], reverse=True)[:10])]
    top_parameters_rastrigin = [parameter_list[index] for (iters, fit, index) in
                                (sorted(results_rastrigin, key=lambda r: r[1], reverse=True)[:10])]
    q2_result_sphere = []
    q2_result_rastrigin = []
    dim = 6
    space = [[X_MIN for _ in range(dim)], [X_MAX for _ in range(dim)]]
    n_list = [1, 5, 10, 20, 30, 50, 75, 100, 200, 500]
    for n in n_list:
        fit_temp = []
        for params in top_parameters_sphere:
            for i in range(2):
                p1 = PSO(dim, params, fit1, 100, space, 9999999, number_particles=n, max_time=1)
                best, fit, iters = p1.main()
                print("done")
                fit_temp.append(fit)
        q2_result_sphere.append(fit_temp)

    for n in n_list:
        fit_temp = []
        for params in top_parameters_rastrigin:
            for i in range(2):
                p1 = PSO(dim, params, fit2, 100, space, 9999999, number_particles=n, max_time=1)
                best, fit, iters = p1.main()
                print("done")
                fit_temp.append(fit)
        q2_result_rastrigin.append(fit_temp)
    print(q2_result_sphere)
    print(q2_result_rastrigin)


if __name__ == '__main__':
    space = [[X_MIN for _ in range(6)], [X_MAX for _ in range(6)]]
    p1 = PSO(6, (0.6, 1.3, 1.3), fit2, 50, space, 1000, number_particles=60)
    print(p1.main())
    # parameter_list = []
    # for c1c2_temp in range(1, 41):
    #     for w_temp in range(-9, 10):
    #         c1c2 = c1c2_temp / 10
    #         w = w_temp / 10
    #         if c1c2 < (24 * (1 - np.square(w)) / (7 - 5 * w)):
    #             parameter_list.append((w, c1c2 / 2, c1c2 / 2))
    # top_parameters1 = [(0.5, 1.2, 1.2), (0.5, 1.1, 1.1), (0.4, 1.3, 1.3)]
    # top_parameters2 = [(0.9, 0.1, 0.1), (0.9, 0.1, 0.1), (0.9, 0.1, 0.1)]
    # top_parameters_rastrigin = [parameter_list[index] for (iters, fit, index) in
    #                             (sorted(results_rastrigin, key=lambda r: r[1], reverse=True)[:1])]
    # q2_result_rastrigin = []
    # dim = 6
    # space = [[X_MIN for _ in range(dim)], [X_MAX for _ in range(dim)]]
    # n_list = [20]
    # for n in n_list:
    #     fit_temp = []
    #     for p in range(len(top_parameters_rastrigin)):
    #         for i in range(50):
    #             w, a1, a2 = top_parameters_rastrigin[p]
    #             param1 = (0.6, 1.3, 1.3)
    #             print(w, a1, a2)
    #             p1 = PSO(dim, top_parameters_rastrigin[p], fit2, 100, space, 1000, number_particles=60)
    #             best, fit, iters = p1.main()
    #             print("done")
    #             fit_temp.append(fit)
    #     q2_result_rastrigin.append(fit_temp)

    # for n in n_list:
    #     fit_temp = []
    #     for p in range(len(top_parameters1)):
    #         for i in range(5):
    #             p1 = HPSO(dim, top_parameters1[p], top_parameters2[p], fit2, 10, space, 1000)
    #             best, fit, iters = p1.main()
    #             print("done")
    #             fit_temp.append(fit)
    #     q2_result_rastrigin.append(fit_temp)

    print(q2_result_rastrigin)
