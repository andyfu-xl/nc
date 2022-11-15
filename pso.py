import numpy as np
import time
from results import *

X_MIN = -5.12
X_MAX = 5.12


# each particle in the swarm
class Particle:
    def __init__(self, fitness, dimension, x_min, x_max):
        self.position = np.random.uniform(low=x_min, high=x_max, size=dimension)
        self.velocity = np.zeros(dimension)
        self.personal_best = self.position
        self.dim = dimension
        self.fitness = fitness(self.position)
        self.fitness_func = fitness
        self.best_fitness = self.fitness

    # update the particle's position and record the personal best solution
    def update_position(self, pos):
        self.position = pos
        self.fitness = self.fitness_func(self.position)
        if self.fitness > self.best_fitness:
            self.best_fitness = self.fitness
            self.personal_best = pos

    # update the particle's velocity
    def update_velocity(self, inertia, a1, a2, global_best):
        r1 = np.random.uniform(low=0, high=1, size=self.dim)
        r2 = np.random.uniform(low=0, high=1, size=self.dim)
        self.velocity = (inertia * self.velocity + a1 * r1 * (self.personal_best - self.position) +
                         a2 * r2 * (global_best - self.position))


# the pso algorithm
class PSO:
    def __init__(self, dimension, parameters, fitness_function, terminate, bound, max_iter=float('inf'),
                 number_particles=20, max_time=float('inf')):
        # I decided to terminate the search base the number of iterations with no change in the global best.
        # Other options like number of iterations are also feasible.
        self.terminate = terminate
        self.w, self.a1, self.a2 = parameters
        # this should be a function
        self.fitness = fitness_function
        self.x_min, self.x_max = bound
        self.number_particles = number_particles
        self.dimension = dimension
        self.iteration = 0
        self.max_iteration = max_iter
        self.max_time = max_time
        self.best_fitness = float("-inf")
        self.time_start = time.time()

        # initialization
        self.swarm = [Particle(fitness_function, dimension, self.x_min, self.x_max) for _ in range(number_particles)]
        self.global_best = self.swarm[0].personal_best

        for i in range(self.number_particles):
            if self.best_fitness <= self.swarm[i].best_fitness:
                self.best_fitness = self.swarm[i].best_fitness
                self.global_best = self.swarm[i].personal_best

    # asynchronous update
    def update(self):
        for x in range(self.number_particles):
            self.swarm[x].update_velocity(self.w, self.a1, self.a2, self.global_best)
            new_position = self.swarm[x].position + self.swarm[x].velocity
            # all particles in the bounded area
            for d in range(self.dimension):
                new_position[d] = max(new_position[d], self.x_min)
                new_position[d] = min(new_position[d], self.x_max)
            self.swarm[x].update_position(new_position)
            # update global best
            new_fitness = self.fitness(new_position)
            if new_fitness > self.best_fitness:
                self.best_fitness = new_fitness
                self.global_best = new_position

    def main(self):
        counter = 0
        previous_best = self.best_fitness
        while ((counter <= self.terminate) and (self.iteration < self.max_iteration) and
               ((time.time() - self.time_start) < self.max_time)):
            self.update()
            self.iteration += 1
            if previous_best == self.best_fitness:
                counter += 1
            else:
                previous_best = self.best_fitness
                counter = 0
        return self.global_best, self.best_fitness, self.iteration


# sphere function
def fit1(x):
    return -sum(x ** 2)


# rastrigin function
def fit2(x):
    return -((10 * len(x)) + sum([(xi ** 2 - 10 * np.cos(2 * np.pi * xi)) for xi in x]))


def q1(fitness_function):
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
        b = (-5.12, 5.12)
        fit_sum = 0
        iter_sum = 0
        for i in range(5):
            p1 = PSO(dim, parameter_list[index], fitness_function, 10, b, 1000)
            best, fit, iters = p1.main()
            fit_sum += fit
            iter_sum += iters
        results.append((iter_sum / 5, fit_sum / 5, index))
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
    b = (-5.12, 5.12)
    n_list = [1, 5, 10, 20, 30, 50, 75, 100, 200, 500]
    for n in n_list:
        fit_temp = []
        for params in top_parameters_sphere:
            for i in range(5):
                p1 = PSO(18, params, fit1, 99999999, b, 99999999, number_particles=n, max_time=0.2)
                best, fit, iters = p1.main()
                fit_temp.append(fit)
        q2_result_sphere.append(fit_temp)

    for n in n_list:
        fit_temp = []
        for params in top_parameters_rastrigin:
            for i in range(5):
                p1 = PSO(18, params, fit2, 99999999, b, 99999999, number_particles=n, max_time=0.2)
                best, fit, iters = p1.main()
                fit_temp.append(fit)
        q2_result_rastrigin.append(fit_temp)
    print(q2_result_sphere)
    print(q2_result_rastrigin)


if __name__ == '__main__':
    print('running')
    # uncomment to run code for question 1, you can change fit1 to fit2 for the rastrigin function
    # printed results have been copied to result.py
    # q1(fit1)
    q2()
