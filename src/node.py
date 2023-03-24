class Node:
    def __init__(self, value, left=None, right=None):
        self.value: str = value
        self.left: Node = left
        self.right: Node = right
        self.fitness: float = 0
