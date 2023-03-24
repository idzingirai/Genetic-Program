import string

import numba as nb
import numpy as np
from numba import types


def convert_tree_to_postfix_string(root) -> str:
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


@nb.njit
def evaluate_postfix_string(row, expression):
    tokens = expression.split()
    stack = nb.typed.List.empty_list(types.float64)
    row = row.astype(np.float32)
    alphabet = list(string.ascii_lowercase)

    for token in tokens:
        if token in ['+', '-', '*']:
            b = stack.pop()
            a = stack.pop()

            if token == '+':
                result = a + b
            elif token == '*':
                result = a * b
            elif token == '-':
                result = a - b

            stack.append(result)

        else:
            stack.append(float(row[alphabet.index(token)]))

    return stack.pop()
