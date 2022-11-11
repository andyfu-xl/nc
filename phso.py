import numpy as np
import random
import time
from results import *

X_MIN = -5.12
X_MAX = 5.12


class HPSO:
    def __init__(self, dimension, parameters1, parameters2, fitness_function, terminate, bound, max_iter=float('inf'),
                 number_particles=20, max_time=float('inf')):
        # I decided to terminate the search base the number of iterations with no change in the global best.
        # Other options like number of iterations are also feasible.
        self.terminate = terminate
        self.omega_1, self.alpha1_1, self.alpha2_1 = parameters1
        self.omega_2, self.alpha1_2, self.alpha2_2 = parameters2
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
        self.index_half = math.ceil(self.number_particles / 2)

        self.x_positions = np.zeros(self.number_particles, self.dimension)
        self.x_velocities = np.zeros(self.number_particles, self.dimension)
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
            if x <= self.index_half:
                # update particle velocity
                self.x_velocities[x] = (self.omega_1 * self.x_velocities[x] + self.alpha1_1 * random.uniform(0, 1) *
                                        (self.x_bests[x] - self.x_positions[x]) + self.alpha2_1 * random.uniform(0, 1) *
                                        (self.global_best - self.x_positions[x]))
                # update particle position
                self.x_positions[x] = self.x_positions[x] + self.x_velocities[x]
                for d in range(self.dimension):
                    self.x_positions[x][d] = max(self.x_positions[x][d], self.x_min[d])
                    self.x_positions[x][d] = min(self.x_positions[x][d], self.x_max[d])
            else:
                # update particle velocity
                self.x_velocities[x] = (self.omega_2 * self.x_velocities[x] + self.alpha1_2 * random.uniform(0, 1) *
                                        (self.x_bests[x] - self.x_positions[x]) + self.alpha2_2 * random.uniform(0, 1) *
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

def fit(x):
    return -((10 * len(x)) + sum([(xi ** 2 - 10 * np.cos(2 * np.pi * xi)) for xi in x]))
