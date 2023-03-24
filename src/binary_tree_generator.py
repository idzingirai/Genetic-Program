import random
import string

from config import FUNCTION_SET
from node import Node
from tree_util import get_tree_depth


class BinaryTreeGenerator:
    def __init__(self, length_of_terminal_set: int):
        terminal_set: [str] = list(string.ascii_lowercase)
        del terminal_set[length_of_terminal_set:]

        self._terminal_set: [str] = terminal_set
        self._function_set: [str] = FUNCTION_SET

    def full(self, depth: int) -> Node:
        if depth > 1:
            return Node(random.choice(self._function_set), self.full(depth - 1), self.full(depth - 1))
        else:
            return Node(random.choice(self._terminal_set))

    def grow(self, depth: int) -> Node:
        if depth > 1:
            if random.randint(0, 1) == 0:
                return Node(random.choice(self._function_set), self.grow(depth - 1), self.grow(depth - 1))
            else:
                return Node(random.choice(self._terminal_set))
        else:
            return Node(random.choice(self._terminal_set))

    def ramped(self, depth: int, index_on_level: int) -> Node:
        if index_on_level % 2 == 0:
            return self.full(depth)
        else:
            tree = self.grow(depth)
            while get_tree_depth(tree) != depth:
                tree = self.grow(depth)
            return tree
