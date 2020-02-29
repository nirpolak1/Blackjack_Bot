# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 21:49:05 2020

@author: user
"""

import numpy as np
import geneticalgorithm as ga
import matplotlib.pyplot as plt
import blackjack as bj

#Defining the fitness of each individual by the expected value of winnings:
def fitness_calculator(population):
    # calulates the result of a linear problem (for now...)
    fitness_values = np.zeros(population.shape[0])
    for i in range (population.shape[0]):
        fitness_values[i] = bj.win_mean(population[i,:],100000)
    return fitness_values

#number of weights in the ANN:
number_of_parameters = 222

#defining mutation base rate:
base_rate=0.9
mutate_factor=0.075

#defining the size of the population:
number_of_individuals = 20
number_of_parents = int(number_of_individuals/5)
population_size = (number_of_individuals, number_of_parameters)

#defining the number of generations:
generations = 100
#defining wieghts range:
lower_bound = -1
upper_bound = 1
parameters_range = np.zeros((2,number_of_parameters))
parameters_range[:][0] = lower_bound
parameters_range[:][1] = upper_bound


b_fig = plt.figure(figsize=(15,10))
progress_max = np.zeros(generations)
progress_mean = np.zeros(generations)
progress_min = np.zeros(generations)
progress_generations = np.linspace(0,generations-1,generations)
progress_fig = b_fig.add_subplot(1,1,1)


#creating the first generation population:
population = np.random.uniform(lower_bound,upper_bound,population_size)

for generation in range(generations):
    print("Generation #", generation)
    #calculating the fitness value of each individual:
    fitness_values = fitness_calculator(population)
    progress_max[generation] = np.max(fitness_values)
    progress_mean[generation] = np.mean(fitness_values)
    progress_min[generation] = np.min(fitness_values)
    #selecting the parents:
    parents_index = ga.sus_selection(population, fitness_values,number_of_parents)
    #creating offsprings:
    offsprings = ga.sus_crossover(parents_index, population, ((number_of_individuals-number_of_parents),(number_of_parameters)))
    #mutating offsprings:
    offsprings_mutated = ga.sus_noise_mutate(offsprings,parameters_range,base_rate,mutate_factor)
    population = ga.sus_insertion(offsprings, population, number_of_parents, fitness_values)
    best_fitness_index = np.where(fitness_values == np.max(fitness_values))
    best_solution = population[best_fitness_index,:]

    #show the best result:
    print("Best Fitness:", np.max(fitness_calculator(population)))
    print("Generation Mean:", progress_mean[generation])

progress_fig.fill_between(progress_generations,progress_max,progress_min,edgecolor='black',facecolor='grey')
progress_fig.plot(progress_generations, progress_mean,c='black')
progress_fig.plot(progress_generations, progress_max,c='r',linewidth=2)
plt.show()

#show the best set and best result:
fitness_values = fitness_calculator(population)
best_fitness_index = np.where(fitness_values == np.max(fitness_values))
best_solution = population[best_fitness_index,:]

print("Best Solution:", population[best_fitness_index,:])
print("Best Solution Fitness:", fitness_values[best_fitness_index])
