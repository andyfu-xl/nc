import numpy as np
import random
import time

X_MIN = -5.12
X_MAX = 5.12


class Particle:
    def __init__(self, fitness, dimension, x_min, x_max):
        self.position = np.random.uniform(low=x_min, high=x_max, size=dimension)
        self.dim = dimension
        self.fitness = fitness(self.position)
        self.fitness_func = fitness

    # update the particle's position and record the personal best solution
    def selection(self, u):
        new_fitness = self.fitness_func(u)
        if self.fitness < new_fitness:
            self.fitness = new_fitness
            self.position = u

    def crossover(self, v, p):
        pd = np.random.uniform(0, 1, self.dim)
        u = []
        for d in range(self.dim):
            if pd[d] < p:
                u.append(v[d])
            else:
                u.append(self.position[d])
        return np.array(u)


class DE:
    def __init__(self, dimension, parameters, fitness_function, terminate, bound, max_iter=1000,
                 number_particles=20, max_time=float('inf')):
        self.f, self.p = parameters
        # this should be a function
        self.fitness = fitness_function
        self.x_min, self.x_max = bound
        self.number_particles = number_particles
        self.dimension = dimension
        self.iteration = 0
        self.max_iteration = max_iter
        self.max_time = max_time
        self.time_start = time.time()
        self.best_fitness = float('-inf')
        self.global_best = None
        self.terminate = terminate

        self.particles = [Particle(fitness_function, dimension, self.x_min, self.x_max) for _ in
                          range(number_particles)]

    def compute_v(self):
        v_list = []
        for _ in range(self.number_particles):
            q, r, s = random.sample(range(self.number_particles), 3)
            vi = self.particles[q].position + self.f * (self.particles[r].position - self.particles[s].position)
            v_list.append(vi)
        return v_list

    # asynchronous update
    def update(self):
        for x in range(self.number_particles):
            v_list = self.compute_v()
            u = self.particles[x].crossover(v_list[x], self.p)
            for d in range(self.dimension):
                u[d] = max(u[d], self.x_min)
                u[d] = min(u[d], self.x_max)
            self.particles[x].selection(u)
            new_fitness = self.particles[x].fitness
            if new_fitness > self.best_fitness:
                self.best_fitness = new_fitness
                self.global_best = self.particles[x].position

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


d = DE(6, (0.5, 0.5), fit2, 100, (-5.12, 5.12), 1000)
print(d.main())
