from typing import Optional
from collections import deque

from node import Node


def get_tree_depth(node: Node) -> int:
    if node is None:
        return 0

    left_depth: int = get_tree_depth(node.left)
    right_depth: int = get_tree_depth(node.right)
    return 1 + max(left_depth, right_depth)


def get_number_of_nodes(root: Node) -> int:
    if root is None:
        return 0
    else:
        return 1 + get_number_of_nodes(root.left) + get_number_of_nodes(root.right)


def get_node_level_by_index(root: Node, index: int) -> int:
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

    return -1


def get_node_by_index(root: Node, index: int) -> Optional[Node]:
    if root is None:
        return None

    height = get_tree_depth(root)
    max_index = 2 ** height - 1
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

    new_root = Node(root.value)
    queue = [(root, new_root)]

    while queue:
        node, new_node = queue.pop(0)

        if node.left:
            new_node.left = Node(node.left.value)
            queue.append((node.left, new_node.left))

        if node.right:
            new_node.right = Node(node.right.value)
            queue.append((node.right, new_node.right))

    del queue
    return new_root

