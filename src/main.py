import time

from process import DataProcessor
from gp import *

random.seed(1)

start_time = time.time()

data_preprocessor = DataProcessor()
data_preprocessor.process()
x_train, y_train, x_test, y_test = data_preprocessor.get_split_dataset()
terminal_set = data_preprocessor.get_features()

tree_generator = BinaryTreeGenerator(length_of_terminal_set=len(terminal_set))

population = generate_initial_population(tree_generator=tree_generator)

# for tree in population:
#     tree.fitness = calculate_fitness(tree, x_train, y_train, terminal_set)
#
# population = sorted(population, key=lambda t: t.fitness)
#
# best_program = population[0]
# num_of_generations = 0
#
# # perform generational population replacement
# while num_of_generations < 500:
#     new_population = []
#     for i in range(int(len(population) / 2)):
#         first_parent = tournament_selection(population)
#         second_parent = tournament_selection(population)
#
#         first_child, second_child = crossover(first_parent, second_parent)
#
#         first_child = mutation(tree_generator=tree_generator, tree=first_child)
#         second_child = mutation(tree_generator=tree_generator, tree=second_child)
#
#         first_child.fitness = calculate_fitness(first_child, x_train, y_train, terminal_set)
#         second_child.fitness = calculate_fitness(second_child, x_train, y_train, terminal_set)
#
#         new_population.append(first_child)
#         new_population.append(second_child)
#
#     population = new_population
#     population = sorted(population, key=lambda t: t.fitness)
#
#     best_program = population[0]
#     num_of_generations += 1
#
# print("[] BEST PROGRAM FITNESS AFTER TRAINING: %s" % best_program.fitness)
# print("[] TESTING BEST PROGRAM ON TEST DATA")
# print("[] BEST PROGRAM FITNESS ON TEST DATA: %s" % calculate_fitness(best_program, x_test, y_test, terminal_set))
# print("[] EXECUTION TIME: %s seconds" % (time.time() - start_time))
