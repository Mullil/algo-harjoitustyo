from tensor import Tensor


def topological_sort(root: Tensor):
    """
    Sorts the Directed Acyclic Graph of the tensors such that each node has its children before it

    Parameters:
        root: The last value computed by the forward pass i.e. the loss

    Returns:
        a list of the nodes sorted according to the description above
    """
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
