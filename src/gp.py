import random
import pandas

from sklearn.metrics import mean_absolute_percentage_error

from config import *
from tree_util import *
from eval import *
from binary_tree_generator import BinaryTreeGenerator
from node import Node


def generate_initial_population(tree_generator: BinaryTreeGenerator) -> [Node]:
    initial_tree_depth = INITIAL_TREE_DEPTH
    population_size = POPULATION_SIZE
    tree_generation_method = TREE_GENERATION_METHOD

    population = []
    num_of_trees_at_each_level = int(population_size / (initial_tree_depth - 1))

    for depth in range(2, initial_tree_depth + 1):
        for i in range(num_of_trees_at_each_level):
            if tree_generation_method == 'RAMPED':
                population.append(tree_generator.ramped(depth, i))
            elif tree_generation_method == 'FULL':
                population.append(tree_generator.full(depth))
            else:
                population.append(tree_generator.grow(depth))

    return population


def calculate_fitness(tree: Node, x_data: pandas.DataFrame, y_data: pandas.Series):
    tree_postfix_expression = convert_tree_to_postfix_string(tree)

    x_data = x_data.to_numpy()
    y = y_data.to_numpy()

    fitness_evaluation = lambda row, expression: evaluate_postfix_string(row, expression)
    y_pred = np.apply_along_axis(fitness_evaluation, axis=1, arr=x_data, expression=tree_postfix_expression)

    return mean_absolute_percentage_error(y, y_pred)


def tournament_selection(population: [Node]) -> Node:
    tournament_size = TOURNAMENT_SIZE
    tournaments = random.sample(population, tournament_size)
    return min(tournaments, key=lambda tree: tree.fitness)


def crossover_is_valid(first_tree: Node, second_tree: Node, first_subtree, second_subtree: Node) -> bool:
    first_subtree_depth = get_tree_depth(first_subtree)
    second_subtree_depth = get_tree_depth(second_subtree)

    first_tree_depth = get_tree_depth(first_tree)
    second_tree_depth = get_tree_depth(second_tree)

    first_offspring_depth = max(first_tree_depth, second_subtree_depth - first_subtree_depth + second_tree_depth)
    second_offspring_depth = max(second_tree_depth, first_subtree_depth - second_subtree_depth + first_tree_depth)

    if first_offspring_depth > MAX_TREE_DEPTH or second_offspring_depth > MAX_TREE_DEPTH:
        return False
    return True


def crossover(first_tree: Node, second_tree: Node) -> (Node, Node):
    first_node = copy_tree(first_tree)
    second_node = copy_tree(second_tree)

    num_of_nodes_in_first_tree = get_number_of_nodes(first_node)
    num_of_nodes_in_second_tree = get_number_of_nodes(second_node)

    while True:
        first_subtree_index = random.randint(1, num_of_nodes_in_first_tree - 1)
        second_subtree_index = random.randint(1, num_of_nodes_in_second_tree - 1)

        first_subtree = get_node_by_index(first_node, first_subtree_index)
        second_subtree = get_node_by_index(second_node, second_subtree_index)

        if crossover_is_valid(first_node, second_node, first_subtree, second_subtree):
            break

    first_subtree_parent_index = (first_subtree_index - 1) // 2
    second_subtree_parent_index = (second_subtree_index - 1) // 2

    first_parent = get_node_by_index(first_node, first_subtree_parent_index)
    if first_subtree_index % 2 != 0:
        first_parent.left = second_subtree
    else:
        first_parent.right = second_subtree

    second_parent = get_node_by_index(second_node, second_subtree_parent_index)
    if second_subtree_index % 2 != 0:
        second_parent.left = first_subtree
    else:
        second_parent.right = first_subtree

    return first_node, second_node


def mutation(tree_generator: BinaryTreeGenerator, tree: Node) -> Node:
    max_tree_depth = MAX_TREE_DEPTH

    number_of_nodes = get_number_of_nodes(tree)
    index = random.randint(1, number_of_nodes - 1)

    if index == 0:
        if random.random() <= 0.5:
            return tree_generator.full(random.randint(2, max_tree_depth))
        else:
            return tree_generator.grow(random.randint(2, max_tree_depth))
    else:
        copy_of_tree = copy_tree(tree)
        node = get_node_by_index(copy_of_tree, index)
        node_level = get_node_level_by_index(node, index)
        tree_depth = random.randint(1, max_tree_depth - node_level)

        if random.random() <= 0.5:
            subtree = tree_generator.full(tree_depth)
        else:
            subtree = tree_generator.grow(tree_depth)

        parent_index = (index - 1) // 2
        parent_tree = get_node_by_index(copy_of_tree, parent_index)

        if index % 2 != 0:
            parent_tree.left = subtree
        else:
            parent_tree.right = subtree

    return tree
