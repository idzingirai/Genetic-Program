import string
import numpy as np
from numba import types, njit, typed

from node import Node


def convert_tree_to_postfix_string(root: Node) -> str:
    stack = []
    postfix = []

    while True:
        while root:
            if root.right:
                stack.append(root.right)

            stack.append(root)
            root = root.left

        root = stack.pop()

        if root.right and stack and stack[-1] == root.right:
            stack.pop()
            stack.append(root)
            root = root.right
        else:
            postfix.append(root.value)
            root = None

        if not stack:
            break

    return ' '.join(postfix)


@njit
def evaluate_postfix_string(row, expression):
    tokens = expression.split()
    stack = typed.List.empty_list(types.float64)
    row = row.astype(np.float32)
    alphabet = list(string.ascii_lowercase)

    for token in tokens:
        if token in ['+', '-', '*', '/', "sqrt"]:
            b = stack.pop()
            a = stack.pop()

            if token == '+':
                result = a + b
            elif token == '*':
                result = a * b
            elif token == '-':
                result = a - b
            elif token == '/':
                if b == 0:
                    result = 1
                else:
                    result = a / b
            elif token == 'sqrt':
                result = np.sqrt(abs(a))

            stack.append(result)

        else:
            value: float = float(row[alphabet.index(token)])
            stack.append(abs(value))

    return np.sqrt(abs(stack.pop()))
