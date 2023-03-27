import pandas
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, r2_score

from config import *
from binary_tree_generator import *
from eval import *
from tree_util import *


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

    if FITNESS_FUNCTION == 'RMSE':
        tree.fitness = np.sqrt(mean_squared_error(y, y_pred))
    elif FITNESS_FUNCTION == 'R2':
        tree.fitness = r2_score(y, y_pred)
    elif FITNESS_FUNCTION == 'MAE':
        tree.fitness = mean_absolute_error(y, y_pred)
    elif FITNESS_FUNCTION == 'MedAE':
        tree.fitness = median_absolute_error(y, y_pred)


def tournament_selection(population: [Node], tournament_size: int) -> Node:
    tournament = random.sample(population, tournament_size)
    tournament.sort(key=lambda t: t.fitness)
    return copy_tree(tournament[0])


def prune_tree(root: Node, max_depth: int, tree_generator: BinaryTreeGenerator) -> Optional[Node]:
    if root is None:
        return None

    if max_depth == 0:
        return tree_generator.full(1)

    root.left = prune_tree(root.left, max_depth - 1, tree_generator)
    root.right = prune_tree(root.right, max_depth - 1, tree_generator)

    return root


def crossover(first_tree: Node, second_tree: Node, tree_generator: BinaryTreeGenerator) -> None:
    num_of_nodes_in_first_tree = get_number_of_nodes(first_tree)
    num_of_nodes_in_second_tree = get_number_of_nodes(second_tree)

    first_subtree_index = random.randint(1, (num_of_nodes_in_first_tree - 1) // 2)
    second_subtree_index = random.randint(1, (num_of_nodes_in_second_tree - 1) // 2)

    first_subtree = get_node_by_index(first_tree, first_subtree_index)
    second_subtree = get_node_by_index(second_tree, second_subtree_index)

    first_subtree_parent_index = (first_subtree_index - 1) // 2
    second_subtree_parent_index = (second_subtree_index - 1) // 2

    first_parent = get_node_by_index(first_tree, first_subtree_parent_index)
    if first_subtree_index % 2 != 0:
        first_parent.left = second_subtree
    else:
        first_parent.right = second_subtree

    second_parent = get_node_by_index(second_tree, second_subtree_parent_index)
    if second_subtree_index % 2 != 0:
        second_parent.left = first_subtree
    else:
        second_parent.right = first_subtree

    first_offspring_depth = get_tree_depth(first_tree)
    second_offspring_depth = get_tree_depth(second_tree)

    if first_offspring_depth > MAX_TREE_DEPTH:
        prune_tree(first_tree, MAX_TREE_DEPTH, tree_generator)

    if second_offspring_depth > MAX_TREE_DEPTH:
        prune_tree(second_tree, MAX_TREE_DEPTH, tree_generator)


def mutation(tree: Node, tree_generator: BinaryTreeGenerator) -> None:
    num_of_nodes_in_tree = get_number_of_nodes(tree)
    mutation_point = random.randint(1, num_of_nodes_in_tree - 1)

    node = get_node_by_index(tree, mutation_point)
    node_level = get_node_level_by_index(node, mutation_point)
    subtree_depth = random.randint(1, MAX_TREE_DEPTH - node_level)

    if random.randint(0, 1) == 0:
        subtree = tree_generator.full(subtree_depth)
    else:
        subtree = tree_generator.grow(subtree_depth)

    mutation_point_parent_index = (mutation_point - 1) // 2
    mutation_point_parent = get_node_by_index(tree, mutation_point_parent_index)

    if mutation_point % 2 != 0:
        mutation_point_parent.left = subtree
    else:
        mutation_point_parent.right = subtree
