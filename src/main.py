import time

from config import CROSSOVER_RATE, MUTATION_RATE, TOURNAMENT_SIZE
from genetic_program import *
from process import DataProcessor
from tree_util import copy_tree

if __name__ == "__main__":
    random.seed(91)
    start_time = time.time()

    processor = DataProcessor()
    processor.process()
    x_train, x_test, y_train, y_test = processor.get_data()
    features = processor.get_features()

    tree_generator = BinaryTreeGenerator(length_of_terminal_set=len(features))
    population = generate_initial_population(tree_generator)

    for tree in population:
        calculate_fitness(tree, x_train, y_train)

    population.sort(key=lambda t: t.fitness)
    best_tree = copy_tree(population[0])
    calculate_fitness(best_tree, x_train, y_train)

    num_of_generations = 0
    while num_of_generations < 500:
        first_tree = tournament_selection(population, TOURNAMENT_SIZE)
        second_tree = tournament_selection(population, TOURNAMENT_SIZE)

        if random.random() < CROSSOVER_RATE:
            crossover(first_tree, second_tree, tree_generator)

        if random.random() < MUTATION_RATE:
            mutation(first_tree, tree_generator)
            mutation(second_tree, tree_generator)

        first_offspring = copy_tree(first_tree)
        second_offspring = copy_tree(second_tree)

        calculate_fitness(first_offspring, x_train, y_train)
        calculate_fitness(second_offspring, x_train, y_train)

        population.append(first_offspring)
        population.append(second_offspring)

        population.sort(key=lambda t: t.fitness)
        population.pop()
        population.pop()

        if population[0].fitness < best_tree.fitness:
            best_tree = copy_tree(population[0])
            calculate_fitness(best_tree, x_train, y_train)

        num_of_generations += 1


print("[] Best Program After Training Fitness (Mean Absolute Percentage Error): " + str(best_tree.fitness) + "%\n")
calculate_fitness(best_tree, x_test, y_test)
print("[] Best Program After Testing Fitness (Mean Absolute Percentage Error): " + str(best_tree.fitness) + "%\n")
print("[] Time Elapsed: " + str(time.time() - start_time) + " seconds")
