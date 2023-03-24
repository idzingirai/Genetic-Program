import random
import pandas
from collections import deque
from typing import Optional


from .config import *
from .eval import *
from tree.binary_tree_generator import BinaryTreeGenerator
from tree.node import Node


def generate_initial_population(tree_generator: BinaryTreeGenerator) -> [Node]:
    population = []
    num_of_trees_at_each_level = int(POPULATION_SIZE / (INITIAL_TREE_DEPTH - 1))

    for depth in range(2, INITIAL_TREE_DEPTH + 1):
        for i in range(num_of_trees_at_each_level):
            if TREE_GENERATION_METHOD == 'RAMPED':
                population.append(tree_generator.ramped(depth, i))
            elif TREE_GENERATION_METHOD == 'FULL':
                population.append(tree_generator.full(depth))
            else:
                population.append(tree_generator.grow(depth))

    return population


def get_tree_depth(node: Node) -> int:
    if node is None:
        return 0

    left_depth = get_tree_depth(node.left)
    right_depth = get_tree_depth(node.right)
    return 1 + max(left_depth, right_depth)


def get_node_level_by_index(root: Node, index: int) -> int:
    # Initialize the queue with the root node and its level
    queue = deque([(root, 1)])

    count = 0
    level = 1

    while queue:
        node, current_level = queue.popleft()

        if count == index:
            return level

        if node.left:
            queue.append((node.left, current_level + 1))
            count += 1
        if count == index:
            level = current_level + 1
            return level
        if node.right:
            queue.append((node.right, current_level + 1))
            count += 1
        if count == index:
            level = current_level + 1
            return level

    # If we have not visited enough nodes, return -1
    return -1


def get_number_of_nodes(root):
    if root is None:
        return 0
    else:
        return 1 + get_number_of_nodes(root.left) + get_number_of_nodes(root.right)


def calculate_fitness(tree: Node, x_data: pandas.DataFrame, y_data: pandas.Series) -> float:
    #   Convert tree to postfix expression
    tree_postfix_expression = convert_tree_to_postfix_string(tree)

    #   Convert data to numpy arrays
    x_data = x_data.to_numpy()
    y = y_data.to_numpy()

    # Calculate the fitness
    fitness_evaluation = lambda row, expression: evaluate_postfix_string(row, expression)
    y_pred = np.apply_along_axis(fitness_evaluation, axis=1, arr=x_data, expression=tree_postfix_expression)

    mean_absolute_percentage_error = (1 / len(x_data)) * np.sum(np.abs((y - y_pred) / y)) * 100
    return mean_absolute_percentage_error


def tournament_selection(population: [Node]) -> Node:
    tournaments = random.sample(population, TOURNAMENT_SIZE)
    return min(tournaments, key=lambda tree: tree.fitness)


def get_node_by_index(root, index):
    if root is None:
        return None

        # Compute the height of the tree to determine the maximum index
    height = get_tree_depth(root)
    max_index = 2 ** height - 1

    # Perform a level-order traversal using the computed indices
    current_level = [root]
    level_index = 0
    while current_level:
        node = current_level[level_index]
        if level_index == index:
            return node
        if node.left is not None:
            left_index = 2 * level_index + 1
            if left_index <= max_index:
                current_level.append(node.left)
        if node.right is not None:
            right_index = 2 * level_index + 2
            if right_index <= max_index:
                current_level.append(node.right)
        level_index += 1
        if level_index == len(current_level):
            current_level = current_level[level_index:]
            level_index = 0

    return None


def copy_tree(root: Node) -> Optional[Node]:
    if not root:
        return None

        # create a copy of the root node
    new_root = Node(root.value)
    # create a queue to store the nodes of the original and copied trees
    queue = [(root, new_root)]

    while queue:
        # get the next node pair from the queue
        node, new_node = queue.pop(0)

        # copy the left child if it exists
        if node.left:
            new_node.left = Node(node.left.value)
            queue.append((node.left, new_node.left))

        # copy the right child if it exists
        if node.right:
            new_node.right = Node(node.right.value)
            queue.append((node.right, new_node.right))

    # return the copied tree's root node
    del queue
    return new_root


def crossover_is_valid(first_tree: Node, second_tree: Node, first_subtree, second_subtree: Node) -> bool:
    #   Get the depths of the subtrees
    first_subtree_depth = get_tree_depth(first_subtree)
    second_subtree_depth = get_tree_depth(second_subtree)

    #   Calculate the maximum depths of the two offspring trees
    first_tree_depth = get_tree_depth(first_tree)
    second_tree_depth = get_tree_depth(second_tree)
    first_offspring_depth = max(first_tree_depth, second_subtree_depth - first_subtree_depth + second_tree_depth)
    second_offspring_depth = max(second_tree_depth, first_subtree_depth - second_subtree_depth + first_tree_depth)

    # Check if either offspring tree has a depth greater than the maximum depth
    if first_offspring_depth > MAX_TREE_DEPTH or second_offspring_depth > MAX_TREE_DEPTH:
        return False
    return True


def crossover(first_tree: Node, second_tree: Node) -> (Node, Node):
    #   Make copies of the trees
    first_node = copy_tree(first_tree)
    second_node = copy_tree(second_tree)

    #   Get the number of nodes in each tree
    num_of_nodes_in_first_tree = get_number_of_nodes(first_node)
    num_of_nodes_in_second_tree = get_number_of_nodes(second_node)

    while True:
        #   Select random points in each tree to swap
        first_subtree_index = random.randint(1, num_of_nodes_in_first_tree - 1)
        second_subtree_index = random.randint(1, num_of_nodes_in_second_tree - 1)

        #   Select the subtrees
        first_subtree = get_node_by_index(first_node, first_subtree_index)
        second_subtree = get_node_by_index(second_node, second_subtree_index)

        # Check if crossover is possible.
        if crossover_is_valid(first_node, second_node, first_subtree, second_subtree):
            break

    #   Get the parent indices of the parents
    first_subtree_parent_index = (first_subtree_index - 1) // 2
    second_subtree_parent_index = (second_subtree_index - 1) // 2

    # Swap the subtrees
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
    #   Select the point in tree to replace
    number_of_nodes = get_number_of_nodes(tree)
    index = random.randint(1, number_of_nodes - 1)

    if index == 0:
        if random.random() < 0.5:
            return tree_generator.full(random.randint(1, MAX_TREE_DEPTH))
        else:
            return tree_generator.grow(random.randint(1, MAX_TREE_DEPTH))
    else:
        #   Make copy of the tree
        copy_of_tree = copy_tree(tree)

        #   Get the node to replace
        node = get_node_by_index(copy_of_tree, index)

        #   Get that node's current level depth
        node_level = get_node_level_by_index(node, index)

        #   Calculate the depth of the new subtree
        tree_depth = random.randint(1, MAX_TREE_DEPTH - node_level)

        #   Generate new subtree
        if random.random() < 0.5:
            subtree = tree_generator.full(tree_depth)
        else:
            subtree = tree_generator.grow(tree_depth)

        #   Get the parent index
        parent_index = (index - 1) // 2

        #   Get the parent node and assign the subtree
        parent_tree = get_node_by_index(copy_of_tree, parent_index)

        if index % 2 != 0:
            parent_tree.left = subtree
        else:
            parent_tree.right = subtree

    return tree
