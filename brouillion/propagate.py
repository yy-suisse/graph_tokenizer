"""
Value propagation over a directed (sub)graph.

Convention: an edge (u, v) means "u depends on v" / "v is a source for u"
(matches the example: D -> H means H is D's source).

A node's value = average of the values of the nodes it points to,
once ALL of those are resolved. Flagged nodes have a fixed value and
short-circuit this (their own out-edges are ignored for their own value).

Algorithm: reverse Kahn's algorithm / topological DP.
  - Track, for each unflagged node, how many of its out-neighbors are
    still unresolved ("remaining").
  - Seed a queue with resolved nodes (the flags).
  - When a node resolves, notify everything that depends on it
    (its in-neighbors); decrement their remaining counter and add
    the resolved value to their running sum.
  - When a node's remaining count hits 0, compute its average and
    enqueue it.

Complexity: O(V + E) time and space. No recursion, so no stack-depth
limits on deep graphs. Any node left unresolved at the end indicates
a cycle or a dependency chain that never bottoms out in a flag.
"""

from collections import defaultdict, deque
import polars as pl

def get_sem_cov_score(result, mapped_list):
    df_result = pl.DataFrame({"concept": list(result.keys()), "sem_cov": [float(v) for v in result.values()]})
    return df_result.filter(pl.col("concept").is_in(list(mapped_list)))


def propagate_n_get_value(edges, flags, default=None, verbose=False):
    """
    edges: iterable of (u, v) pairs meaning "u's source is v"
    flags: dict {node: fixed_value}
    default: value to assign to unflagged sink nodes (no out-edges).
             Leave as None to leave them unresolved instead.
    verbose: print each resolution step in order, for debugging/tracing.

    Returns: dict {node: resolved_value}
    """
    out_edges = defaultdict(set)   # node -> its sources
    in_edges = defaultdict(set)    # node -> nodes that depend on it
    nodes = set(flags)

    for u, v in edges:
        out_edges[u].add(v)
        in_edges[v].add(u)
        nodes.add(u)
        nodes.add(v)

    value = dict(flags)
    remaining = {n: len(out_edges[n]) for n in nodes if n not in flags} # remining nodes with outedges to be removed later
    acc = defaultdict(float)

    queue = deque(flags.keys())

    # unflagged sinks (no out-edges) have nothing to average
    for n in nodes:
        if n not in flags and not out_edges[n]:
            if default is not None:
                value[n] = default
                queue.append(n)

    order = []
    while queue:
        u = queue.popleft()
        order.append(u)
        for p in in_edges[u]:
            if p in value:
                continue
            acc[p] += value[u]
            remaining[p] -= 1
            if remaining[p] == 0:
                value[p] = acc[p] / len(out_edges[p])
                queue.append(p)

    if verbose:
        for n in order:
            print(f"  {n} = {value[n]}")

    unresolved = nodes - value.keys()
    if unresolved:
        print(f"warning: {len(unresolved)} node(s) never resolved "
              f"(cycle or missing flag): {sorted(unresolved)}")

    return value


# if __name__ == "__main__":
#     # user's example graph
#     edges = [
#         ("D", "H"), # D -> H
#         ("E", "I"),
#         ("M2", "I"), ("M2", "J"),
#         ("A", "C"), ("A", "D"),
#         ("B", "E"), ("B", "D"),
#         ("M1", "A"), ("M1", "B"),
#     ]
#     flags = {"F": 0, "G": 0, "H": 0, "J": 0, "I": 1, "C": 1}

#     print("resolution order and values:")
#     result = propagate(edges, flags, verbose=True)

#     print("\nfinal values:")
#     for n in sorted(result):
#         print(f"  {n}: {result[n]}")
