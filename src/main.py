import time

from process import DataProcessor
from gp import *

random.seed(55)

start_time = time.time()

data_preprocessor = DataProcessor()
data_preprocessor.process()
x_train, x_test, y_train, y_test = data_preprocessor.get_split_dataset()
terminal_set = data_preprocessor.get_features()

tree_generator = BinaryTreeGenerator(length_of_terminal_set=len(terminal_set))

population = generate_initial_population(tree_generator=tree_generator)

for tree in population:
    tree.fitness = calculate_fitness(tree, x_train, y_train)

population = sorted(population, key=lambda t: t.fitness)

best_program = copy_tree(population[0])
best_program.fitness = calculate_fitness(best_program, x_train, y_train)
num_of_generations = 0
crossover_rate = CROSSOVER_RATE
mutation_rate = MUTATION_RATE

numRetires = 0

# perform generational population replacement
while num_of_generations < 5000:
    first_offspring = tournament_selection(population=population)
    second_offspring = tournament_selection(population=population)

    if random.random() < crossover_rate:
        first_offspring, second_offspring = crossover(first_offspring, second_offspring)

    if random.random() < mutation_rate:
        first_offspring = mutation(tree_generator, first_offspring)
        second_offspring = mutation(tree_generator, second_offspring)

    first_offspring.fitness = calculate_fitness(first_offspring, x_train, y_train)
    second_offspring.fitness = calculate_fitness(first_offspring, x_train, y_train)

    population.append(first_offspring)
    population.append(second_offspring)

    population = sorted(population, key=lambda t: t.fitness)
    population.pop()
    population.pop()

    if population[0].fitness < best_program.fitness:
        best_program = copy_tree(population[0])
        best_program.fitness = calculate_fitness(best_program, x_train, y_train)
        numRetires = 0
    else:
        numRetires += 1
        if numRetires > 25:
            new_population = generate_initial_population(tree_generator=tree_generator)

            for tree in new_population:
                tree.fitness = calculate_fitness(tree, x_train, y_train)

            # Merge the new population with the old one
            population.extend(new_population)

            # Sort the population
            population = sorted(population, key=lambda t: t.fitness)

            # Remove the worst half of the population
            population = population[:len(population)//2]

            crossover_rate = 0.5
            mutation_rate = 0.5
            numRetires = 0

    print("[] BEST PROGRAM FITNESS AFTER GENERATION %s: %s" % (num_of_generations, best_program.fitness))
    num_of_generations += 1

print("[] BEST PROGRAM FITNESS AFTER TRAINING: %s" % best_program.fitness)
print("[] TESTING BEST PROGRAM ON TEST DATA")
print("[] BEST PROGRAM FITNESS ON TEST DATA: %s" % calculate_fitness(best_program, x_test, y_test))
print("[] EXECUTION TIME: %s seconds" % (time.time() - start_time))
