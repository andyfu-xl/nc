import math
import numpy as np
import random
import time
from results import *

X_MIN = -5.12
X_MAX = 5.12


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


class HPSO:
    def __init__(self, dimension, parameters1, parameters2, fitness_function, terminate, bound, max_iter=float('inf'),
                 number_particles=20, max_time=float('inf')):
        # I decided to terminate the search base the number of iterations with no change in the global best.
        # Other options like number of iterations are also feasible.
        self.terminate = terminate
        self.w_1, self.a1_1, self.a2_1 = parameters1
        self.w_2, self.a1_2, self.a2_2 = parameters2
        # this should be a function
        self.fitness = fitness_function
        # bound should be 2D array in shape (2, dimension)
        self.x_min, self.x_max = bound
        self.number_particles = number_particles
        self.dimension = dimension
        self.iteration = 0
        self.max_iteration = max_iter
        self.max_time = max_time
        self.best_fitness = float("-inf")
        self.time_start = time.time()
        self.half_population = int(number_particles / 2)

        self.swarm = [Particle(fitness_function, dimension, self.x_min, self.x_max) for _ in range(number_particles)]
        self.global_best = self.swarm[0].personal_best

        for i in range(self.number_particles):
            if self.best_fitness <= self.swarm[i].best_fitness:
                self.best_fitness = self.swarm[i].best_fitness
                self.global_best = self.swarm[i].personal_best

    # asynchronous update
    def update(self):
        for x in range(self.number_particles):
            if x < self.half_population:
                self.swarm[x].update_velocity(self.w_1, self.a1_1, self.a2_1, self.global_best)
            else:
                self.swarm[x].update_velocity(self.w_2, self.a1_2, self.a2_2, self.global_best)
            new_position = self.swarm[x].position + self.swarm[x].velocity
            for d in range(self.dimension):
                new_position[d] = max(new_position[d], self.x_min)
                new_position[d] = min(new_position[d], self.x_max)
            self.swarm[x].update_position(new_position)
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


def fit2(x):
    return -((10 * len(x)) + sum([(xi ** 2 - 10 * np.cos(2 * np.pi * xi)) for xi in x]))


def q3():
    parameter_list = []
    for c1c2_temp_1 in range(7):
        for c1c2_temp_2 in range(7):
            for w_1 in [-0.9, -0.6, -0.3, 0, 0.3, 0.6, 0.9]:
                for w_2 in [-0.9, -0.6, -0.3, 0, 0.3, 0.6, 0.9]:
                    if c1c2_temp_2 == c1c2_temp_1 or w_1 == w_2:
                        continue
                    c1c2_1 = c1c2_temp_1 * 0.5
                    c1c2_2 = c1c2_temp_2 * 0.5
                    parameter_list.append([(w_1, c1c2_1 / 2, c1c2_1 / 2), (w_2, c1c2_2 / 2, c1c2_2 / 2)])
    results = []
    for index in range(len(parameter_list)):
        dim = 6
        b = (-5.12, 5.12)
        fit_sum = 0
        iter_sum = 0
        for i in range(5):
            p1 = HPSO(dim, parameter_list[index][0], parameter_list[index][1], fit2, 10, b, 1000)
            best, fit, iters = p1.main()
            fit_sum += fit
            iter_sum += iters
        results.append((iter_sum / 5, fit_sum / 5, index))
    print(results)


if __name__ == '__main__':
    q3()
