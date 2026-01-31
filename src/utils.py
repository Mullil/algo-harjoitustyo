from tensor import Tensor


def topological_sort(root: Tensor):
    visited = set()
    result = []

    def helper(node: Tensor):
        if node in visited:
            return
        visited.add(node)
        for c in node.children:
            helper(c)
        result.append(node)
    helper(root)
    return result
