
import numpy as np

def sus_selection(population, fitness_values,number_of_parents):
    #creating an empty array to hold the indexes of selected parents among the population
    parents_index = np.empty(number_of_parents)
    #making sure all fitness values for roulette sections are positive
    fitness_values = fitness_values - np.min(fitness_values)
    sum_of_fitness = np.sum(fitness_values)
    pointer_step = 1/number_of_parents
    pointer = np.random.uniform(0.0,1.0)%pointer_step
    roulette_i = 0.0
    selection_index = 0
    #selecting the individuals where a roulette pointer lies and recording their index:
    for i in range(0,fitness_values.shape[0]):
        roulette_f = roulette_i + fitness_values[i]/sum_of_fitness
        if (roulette_i <= pointer < roulette_f):
            parents_index[selection_index] = np.uint8(i)
            selection_index += 1
            pointer += pointer_step
        roulette_i = roulette_f
    
    return parents_index

def sus_crossover(parents_index, population, number_of_offsprings):
    offsprings = np.empty(number_of_offsprings)
    for index in range(number_of_offsprings[0]):
        #selecting the parents by order of fitness:
        first_parent = np.uint8(parents_index[(index%parents_index.shape[0])])
        second_parent = np.uint8(parents_index[(index+1)%parents_index.shape[0]])
        #randomly choosing which parameters would be taken from the first or second parent:
        crossover_list = np.random.randint(0,2,number_of_offsprings[1])
        #creating the offspring:
        offsprings[index,:] = population[first_parent,:]*crossover_list + population[second_parent,:]*(1-crossover_list)
    return offsprings

def sus_mutate(offsprings,parameters_range,rate):
    #mutating a random parameter in each offspring:
    for index in range(offsprings.shape[0]):
        mutation_vector = np.random.uniform(0.0,1.0,offsprings.shape[1])
        mutation_vector = np.where(mutation_vector <= rate, 1 ,0)
        offsprings[index,:] = mutation_vector*np.random.uniform(parameters_range[0,0],parameters_range[1,0],offsprings.shape[1]) + (1-mutation_vector)*offsprings[index,:]
    return offsprings

def sus_noise_mutate(offsprings,parameters_range,rate,factor):
    #mutating a random parameter in each offspring:
    for index in range(offsprings.shape[0]):
        mutation_vector = np.random.uniform(0.0,1.0,offsprings.shape[1])
        mutation_vector = np.where(mutation_vector <= rate, 1 ,0)
        offsprings[index,:] = mutation_vector*(offsprings[index,:]+np.random.uniform(-factor,factor,offsprings.shape[1]))+ (1-mutation_vector)*offsprings[index,:]
        offsprings[index,:] = np.where(offsprings[index,:]<1, offsprings[index,:], parameters_range[1,0] )
        offsprings[index,:] = np.where(offsprings[index,:]>-1, offsprings[index,:],  parameters_range[0,0] )
    return offsprings

def sus_insertion(offsprings, population, number_of_parents, fitness_values):
    parents = np.empty((number_of_parents, population.shape[1]))
    #selecting the parents by decreasing fitness values:
    for parent_index in range(number_of_parents):
        max_fitness_index = np.where(fitness_values == np.max(fitness_values))
        max_fitness_index = max_fitness_index[0][0]
        parents[parent_index, :] = population[max_fitness_index,:]
        #erasing the best current fitness to move to the next best fitness:
        fitness_values[max_fitness_index] = -999999999
    population[0:number_of_parents,:] = parents
    population[number_of_parents:,:] = offsprings
    return population