#Brittany Strong
#Sepetember 22, 2023
#Project 2- Traveling Salesman Problem using Genetic Algorithm
import random
import math
import itertools
import matplotlib.pyplot as plt

#This program will create a Genetic Algorithm for the Traveling Salesman Problem, taking a list of cities.The Genetic Algorithm takes all the elements from the theory of Evolution (mutation, crossover, and selection) and create a breakdown of each within a program that will give you the best general solution rather than the absolute best solution.

# Number of cities for Traveling Salesman
city_List = 25

# Generate random cities, this will give x and y coordinates
cities = [(random.randint(0, 100), random.randint(0, 100)) for _ in range(city_List)]

# Function to calculate distance between two cities
#use the distance formular d = ((x2 -x1)^2 + (y2-y1)^2)^1/2
def calculate_distance(city1, city2):
    x1, y1 = city1
    x2, y2 = city2
    distance_form = ((x1-x2) **2 + (y1 - y2) **2) ** 0.5
    return distance_form

# Function to find the fitness of all the individuals
def calculate_tour_distance(tour):
    total_distance = 0
    for i in range(len(tour)):
        #sums all the total distances and the indivudals in the city
        total_distance += calculate_distance(cities[tour[i]], cities[tour[(i + 1) % len(tour)]])
    return total_distance

# function for the initial popualtion, will be our first generation
def generate_initial_population(population_size):
    population = []
    all_cities = list(range(city_List))
    for _ in range(population_size):
        #shuffles and then adds the cities to the population
        random.shuffle(all_cities)
        population.append(all_cities[:])
    return population

# Perform one-point crossover on tours
def one_point_crossover(parent_a, parent_b):
    crossover_point = random.randint(0, len(parent_a) - 1)
    #crossing population points for all the points that are not the first and last in the list
    child1 = parent_a[:crossover_point] + [city for city in parent_b if city not in parent_a[:crossover_point]]
    child2 = parent_b[:crossover_point] + [city for city in parent_a if city not in parent_b[:crossover_point]]
    return child1, child2

# Mutate the children by shifting points in each array
def mutate(tour):
    index1, index2 = random.sample(range(len(tour)), 2)
    tour[index1], tour[index2] = tour[index2], tour[index1]
    return tour

# Reproduce the selected children
def reproduce_children(selected_individuals):
    children = []
    for i in range(0, len(selected_individuals), 2):
        parent1 = selected_individuals[i]
        parent2 = selected_individuals[(i + 1) % len(selected_individuals)]
        child1, child2 = one_point_crossover(parent1, parent2)
        children.append(mutate(child1))
        children.append(mutate(child2))
    return children


# Set the hyper parameters for the genetic algorithm
NUMBER_OF_GENERATIONS = 1000
INITIAL_POPULATION_SIZE = 1000
MUTATION_RATE = 0.1  # Mutation rate (probability of mutation)
NUMBER_OF_ITERATIONS = 5

    
# Run the genetic algorithm
def run_ga():
    best_tour = []
    best_global_distance = math.inf # Best distance found so far
    global_population = generate_initial_population(INITIAL_POPULATION_SIZE)

    for generation in range(NUMBER_OF_GENERATIONS):
        # Calculate fitness (total tour distance) for each individual in the population
        population_distances = [calculate_tour_distance(tour) for tour in global_population]

        # Find the best tour and its distance in this generation
        best_generation_distance = min(population_distances)
        if best_generation_distance < best_global_distance:
            best_global_distance = best_generation_distance
            best_tour = global_population[population_distances.index(best_generation_distance)]


        # Find Selection based on fitness (shorter distance is better)
        selected_individuals = [global_population[i] for i in range(len(global_population))
                                if population_distances[i] <= best_generation_distance]

        # Call to reproduce children function using selected individuals
        children = reproduce_children(selected_individuals)

        # children mix with selected children
        global_population = selected_individuals + children

        # Random mutation with the chosen population
        for i in range(len(global_population)):
            if random.random() < MUTATION_RATE:
                global_population[i] = mutate(global_population[i])
                
    #Global population, accuracy, final population and best distance, ex. from Grokking
    print('Global Population: ', global_population)
    print('Accuracy:', best_global_distance /city_List * 100)
    print('Final population size: ', len(global_population))
    print('Best tour distance: ', best_global_distance)

     # Plot the Traveling Salesman algorithm
    x_coords = [cities[i][0] for i in best_tour] + [cities[best_tour[0]][0]]
    y_coords = [cities[i][1] for i in best_tour] + [cities[best_tour[0]][1]]

    plt.figure()
    plt.plot(x_coords, y_coords, 'o-')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Best Tour')
    plt.show()

# Run the genetic algorithm
for i in range(0,NUMBER_OF_ITERATIONS):
    print('\n',i+1,'.\n')
    run_ga()
