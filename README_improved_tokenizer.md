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


### Output

- `F_D(T) = mean(S[D, c] for c in M)` — Eq. (2), the global score for this `T`.
- This is the **full recomputation** (§6.2, cost `O(D(|V|+|E|))`), useful as ground truth / for
  the exact-greedy baseline. The incremental `CANDIDATE_DELTA`/`COMMIT` scheme (§6.3) reuses this
  `S` array instead of recomputing it from scratch for each greedy candidate.

### Keynote

**What the implementation actually does, in words.** Rather than walking the graph node by
node, `compute_semantic_coverage` builds one reusable matrix and repeatedly multiplies it:

1. **Build the connectivity matrix `A`.** `A` is the row-normalized out-adjacency matrix of
   `combined_subgraphs`: `A[i, j] = 1 / out_degree(u_i)` if `u_i → u_j` is an edge, else `0`.
   Each row sums to 1, so `A` encodes "one hop, averaged over out-edges" as a single linear
   operator. It's built once, from graph structure (+ which nodes are tokens — see below), and
   reused unchanged at every horizon; nothing about it depends on `h`.
2. **Iterate over horizons `h = 0 … D`.** At `h = 0` every node's value is just `1[u ∈ T]` — no
   matrix involved, purely a lookup. From `h = 1` on, every node's value is *either* an override
   (`1`, if the node is a token) *or* the `A`-weighted average of its successors' values from the
   previous horizon (`S[h] = A @ S[h-1]`) — computed for every node at once as a single
   matrix-vector product, then the token entries are force-overwritten to `1` right after (needed
   at *every* horizon, not just `h=0`: a token's row in `A` is zeroed out precisely so it doesn't
   get averaged from its real successors, which means the raw matmul would otherwise reset it to
   `0` at each step if the override weren't re-applied). Dead ends (`out_degree(u) = 0`) need no
   special case — their all-zero row in `A` already makes the matmul give `0` there naturally.
3. **No topological ordering is needed.** Processing starts at `h = 0`, where every node's score
   depends only on itself (token membership). From then on, `S[h, u]` never depends on any other
   node's value *at the same horizon `h`* — only on values from horizon `h - 1`, which is already
   fully computed by the time horizon `h` starts. So looping `h = 0, 1, …, D` in order is enough
   to guarantee every dependency is resolved before it's used, regardless of how nodes are
   connected in the graph (cycles included).

---

## Lazy greedy token selection (§5-6)

### Algorithm

- **Goal**: pick `T ⊆ V_D`, `|T| = k`, maximizing `F_D(T)` (Eq. 2). `V_D` = candidates that can
  have nonzero gain = `M ∪ {reachable from M within D hops}` (§5) — this is exactly
  `combined_subgraphs.nodes()`, see above.
- **Greedy**: start from `T = ∅`, repeatedly add whichever candidate has the largest marginal
  gain `Δ_D(t | T) = F_D(T ∪ {t}) − F_D(T)` (Eq. 3-4).
- **Why greedy is good enough**: `F_D` is normalized, monotone, and **submodular**
  (Proposition 1) — a token's marginal contribution can only shrink as more tokens get added.
  The classic Nemhauser-Wolsey-Fisher result then guarantees
  `F_D(T_greedy_k) ≥ (1 − 1/e) · max_{|T|≤k} F_D(T)` (Eq. 5) — greedy is provably within
  `(1 − 1/e) ≈ 63%` of optimal, for every prefix size, not just the final `k`.
- **Exact greedy vs. lazy greedy**: exact greedy recomputes every remaining candidate's gain at
  every iteration. Lazy greedy instead keeps one (possibly stale) gain bound per candidate in a
  max-priority queue. Submodularity guarantees a stale bound only ever *overestimates* the
  candidate's current true gain — so popping the queue's current top, recomputing its gain
  exactly, and checking that it still beats the next-best stale bound is enough to certify it's
  the true maximizer, without touching any other candidate that round.
- **Local incremental computation (§6.3)**: rather than a full `O(D(|V|+|E|))` recompute of `S`
  per candidate, `CANDIDATE_DELTA` propagates only the *change* caused by adding `t`, backward
  through `In(v)` edges — touching only nodes that can reach `t` within `D` hops (Eq. 6-7 define
  this sparse delta; Eq. 8 turns `delta[D]` into the marginal gain). `COMMIT` then applies the
  accepted candidate's delta onto the stored `S` and adds it to `T` (§6.1/6.3), instead of
  rerunning `compute_semantic_coverage` from scratch after every token.

### Implementation — `LazyGreedyTokenSelector` (`src/new_tokenizer.py`)

- **State kept per instance**: `G`, `M`, `D`, `V`, `node_to_idx`; `out_degree` = `d+(u)`, static
  since it never depends on `T`; `in_nodes` = `In(v)` (built once); `T` (current token set); `S`
  (current `(D+1, |V|)` coverage matrix, same layout as `compute_semantic_coverage`'s output);
  `current_score` = `F_D(T)` for the current `T`.
- **`candidate_delta(t)`** — thin wrapper around `_candidate_delta_core`, a plain/picklable
  function implementing Eq. (6)-(7) (pulled out so the exact same logic can run as an instance
  method or inside a worker process).
- **`commit(t, delta)`** — applies `delta` onto `S`, adds `t` to `T` (COMMIT, §6.1/6.3).
- **`select(k, candidates=None, verbose=True, n_jobs=1, progress_every=5000)`**:
  1. **Seeding** — compute the *exact* initial gain for every not-yet-selected candidate ("initial
     gains are exact", §6.1) and push `(-gain, t)` onto a max-heap (`heapq` is a min-heap, so
     gains are negated). This is one `CANDIDATE_DELTA` call per candidate — `O(|candidates|)`, and
     the dominant one-time cost on a large graph (e.g. ~65s for all 75,635 real SNOMED candidates
     with `n_jobs=8`, vs. minutes serially). `progress_every` prints scan progress since this phase
     produces no other output; `n_jobs` (`-1` = all cores) parallelizes it across worker processes
     via `_init_worker`/`_worker_seed_gain`, since each candidate's initial gain only depends on the
     fixed `T`/`S` at call time (embarrassingly parallel).
  2. **Greedy loop** — pop the heap's current top, recompute its gain exactly, compare against the
     next-best stale bound still in the queue; commit if it wins, otherwise re-queue with the fresh
     value and repeat. Runs until `k` tokens are selected or candidates run out.
  3. If `verbose`, prints one line per accepted token: rank `i/k`, id, marginal gain, cumulative
     score; `re-evals` (how many stale candidates were re-checked this round before finding the
     true maximizer — a lazy-greedy efficiency indicator, `1` = immediate accept); and `opt<=`, an
     upper bound on the best score achievable at that budget size from the `(1 − 1/e)` guarantee
     (`GREEDY_RATIO = 1 - 1/e`; `OPT_i ≤ F_D(T_greedy_i) / GREEDY_RATIO`), clipped to `1.0` since
     `F_D ≤ 1` always.

### Keynote

**The whole flow, in words, naming every function.**

1. `LazyGreedyTokenSelector.__init__` sets up the read-only structure once — `node_to_idx`,
   `out_degree` (static `d+(u)`, never depends on `T`), `in_nodes` (`In(v)`, needed to walk
   *backward*) — plus the mutable state `T` (empty at first) and `S` (all zero, consistent with
   `T = ∅`).
2. To find out how good it would be to add one candidate `t` to the current `T`, `select` calls
   `candidate_delta(t)`, a thin wrapper delegating to `_candidate_delta_core` — a pure,
   side-effect-free function that only *reads* the current `S`/`T` snapshot and returns
   `(gain, delta)`: `gain` is the marginal improvement to `F_D` from adding `t`; `delta` is the
   sparse set of `S` entries that *would* change if `t` were actually accepted. Crucially,
   `_candidate_delta_core` never writes to `S` — it can be called on the same or different
   candidates repeatedly, freely, with no side effects.
3. Every call to `_candidate_delta_core` for a given `t` only depends on that same frozen `T`/`S`
   snapshot — never mutated during evaluation — so evaluating many *different* candidates is
   embarrassingly parallel (unlike the `h`-loop *inside* one call, which stays sequential since
   horizon `h` needs horizon `h-1`). `select`'s seeding phase exploits exactly this: when
   `n_jobs != 1`, it spins up a `multiprocessing.Pool`, ships the read-only state once per worker
   via `_init_worker`, and has each worker run `_worker_seed_gain(t)` — a thin wrapper around
   `_candidate_delta_core` that discards `delta` and returns only `(t, gain)`, since `delta` will
   be recomputed anyway if `t` ever reaches the top of the heap.
4. Whether computed serially or in parallel, every candidate's initial gain is pushed onto a
   max-priority queue via `heapq.heappush(heap, (-gain, t))` — negated because `heapq` is only a
   min-heap, so the most negative value (the largest real gain) naturally rises to the top; the
   tuple carries `t` along so we still know which candidate that gain belongs to once popped.
5. The greedy loop repeatedly pops the heap's current top, recomputes its gain for real via
   `candidate_delta` (state may have moved since it was pushed), and peeks at the next-best
   remaining bound via `-heap[0][0]` (read-only, does *not* pop). If the fresh gain still beats
   that bound, submodularity guarantees no un-recomputed candidate could beat it either, so it's
   accepted: `commit(t, delta)` applies the delta onto `self.S` and adds `t` to `self.T` — this
   is the **only** place `self.S` ever changes. If the fresh gain doesn't win, it's pushed back
   with its now-corrected value and the next top gets tried instead.
6. On each accepted token, if `verbose`, a line is printed with the rank, gain, cumulative score,
   how many `re-evals` it took to find that token (a lazy-greedy efficiency indicator), and an
   `opt<=` bound computed from `GREEDY_RATIO = 1 - 1/e` (the Nemhauser-Wolsey-Fisher constant) —
   a purely informational number, used nowhere else in the algorithm, that upper-bounds the best
   score achievable at that budget from the `(1 - 1/e)` guarantee.

### Validated against

- The paper's own numeric example (§4): `S1(t1)=1`, `S1(v)=0.5`, `S2(u)=0.75`, exact match.
- A brute-force **exact greedy** (full `compute_semantic_coverage` recompute for every candidate,
  every iteration) on a random 60-node cyclic multigraph — lazy and exact picked identical
  tokens, identical order, identical gains, and both matched a final full recomputation.
- The real `combined_subgraphs` (75,635 nodes / 290,762 edges): `n_jobs=8` seeding + first 5
  tokens completed in ~67s; `re-evals` stayed at 1-4 per token after seeding.

### Usage

```python
selector = new_tokenizer.LazyGreedyTokenSelector(combined_subgraphs, mapped_ids, D=3)
history = selector.select(k=200, n_jobs=-1)   # n_jobs=-1: parallel seeding across all cores

selector.T                 # selected token concept ids
selector.current_score     # final F_D(T)
history                    # [(token, marginal_gain, cumulative_score), ...] in selection order
```
