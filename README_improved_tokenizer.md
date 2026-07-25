# Improved tokenizer — notes

## Combined subgraph computation

Builds the candidate set `V_D = M ∪ {u ∈ V : reachable from M within max_dist relationships}`
(§5 of the semantic-coverage note), used both as the universe for token candidates and as the
graph the coverage recurrence runs on.

**Inputs**
- `df_relations`: SNOMED relationship triples `(src, relation, dst)`.
- `mapped_ids` (`M`): mapped hospital concepts, from `df_mapped["id"]`.
- `max_distance`: the depth budget `D` used to build the candidate neighborhood.

**Steps** (`src/graph_fct.py`)

1. `build_relations_graph(df_relations)` — load all SNOMED relationships into a
   `nx.MultiDiGraph`: one edge per triple, `src -> dst` labeled `relation` (a `(src, dst)` pair
   can carry several relation types, hence `MultiDiGraph`).
2. `get_combined_subgraphs_from_nodes(whole_graph, mapped_ids, max_distance)`:
   - for each mapped concept `m ∈ M`, take `nx.ego_graph(whole_graph, m, radius=max_distance)`
     — the subgraph reachable from `m` by following **outgoing** edges up to `max_distance`
     hops (directed, not undirected);
   - `nx.compose_all(...)` the per-concept ego graphs into a single graph.
3. Result: `combined_subgraphs`, a `MultiDiGraph` whose node set is `V_D` and whose edges are
   exactly the relationship occurrences reachable from some mapped concept within `D` hops —
   this is the `V`/`E` used everywhere below.

```python
whole_graph = graph_fct.build_relations_graph(df_relations, col_src="src.id", col_dst="dst.id", col_relation="relation")
combined_subgraphs = graph_fct.get_combined_subgraphs_from_nodes(whole_graph, mapped_ids, max_distance=configs_mapped.TokenizerParam().max_dist_candidate)
```

Current size (one run): `75 635` nodes, `290 762` edges.

---

## semantic_coverage matrix computation

**Goal:** for a fixed token set `T ⊆ V`, compute
`S[h, u] = S_h(u, T)` for every horizon `h = 0, …, D` and every concept `u ∈ V`,
where `V` = nodes of `combined_subgraphs`.

Matrix shape: `S ∈ {0,1}^{(D+1) × |V|}` (e.g. `[4, |V|]` for `D = 3`).

### Inputs
- `V`: nodes of `combined_subgraphs`
- `T ⊆ V`: current token set
- `D`: depth budget
- `Out(u)`: outgoing neighbors of `u` in `combined_subgraphs` (relation label not needed for the
  score, only the destination)
- `out_degree(u) = len(Out(u))`

### Steps

1. **Node index** — build `idx: V → {0, …, |V|-1}` (arbitrary order; no topological sort needed).
2. **Allocate** `S`, shape `(D+1, |V|)`, filled with `0`.
3. **Out-edge dict** — `Out[u] = [v for (u, r, v) in combined_subgraphs.out_edges(u)]`.
4. **Base case (`h = 0`)**, Eq. (1) first line:
   ```
   S[0, u] = 1   if u ∈ T
   S[0, u] = 0   otherwise
   ```
5. **Recursive layers (`h = 1 … D`)** — computed strictly from the previous layer `S[h-1, :]`,
   never from `S[h, :]` itself:
   ```
   for h in 1..D:
       for u in V:
           if u ∈ T:
               S[h, u] = 1                              # token, absorbing
           elif out_degree(u) == 0:
               S[h, u] = 0                              # dead end
           else:
               S[h, u] = mean(S[h-1, v] for v in Out(u)) # avg of successors' prior-horizon coverage
   ```

### Why no topological order

`S[h, u]` only ever reads `S[h-1, v]`. Since `h-1 < h`, looping `h` as the outer loop already
guarantees every dependency is resolved before it's needed — regardless of node order, and
regardless of whether the graph has cycles (it likely does, since non-`ISA` relations can loop
back across concepts).

### Output

- `F_D(T) = mean(S[D, c] for c in M)` — Eq. (2), the global score for this `T`.
- This is the **full recomputation** (§6.2, cost `O(D(|V|+|E|))`), useful as ground truth / for
  the exact-greedy baseline. The incremental `CANDIDATE_DELTA`/`COMMIT` scheme (§6.3) reuses this
  `S` array instead of recomputing it from scratch for each greedy candidate.
