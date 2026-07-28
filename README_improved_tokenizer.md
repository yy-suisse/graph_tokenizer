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
- `lam` (optional, `0 < lam ≤ 1`, default `1.0`): distance/horizon penalty — see below.

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
               S[h, u] = lam * mean(S[h-1, v] for v in Out(u)) # avg of successors' prior-horizon coverage, scaled by lam
   ```

### Distance / horizon penalty (`lam`)

Optional extension: `lam ∈ (0, 1]` penalizes tokens found farther away. It modifies only Eq. (1)'s
third case (shown above, `lam=1` recovers the original equation exactly). Each hop taken
multiplies by `lam` once, so a token found `j` hops away contributes `lam^j` instead of `1` — this
falls out of the recursion for free, no separate depth bookkeeping needed. A token's own value
stays `1`, unpenalized (being *at* the token costs zero further hops).

This preserves `F_D`'s normalized/monotone/submodular properties (Proposition 1): it's equivalent
to reweighting the random-walk building block from Proposition 1's justification with a
non-increasing weight `lam^position` instead of a flat `1` — a standard distance-weighted coverage
function, still monotone submodular. So the `(1 - 1/e)` greedy guarantee (Eq. 5) still holds for
any `lam ∈ (0, 1]`, and `LazyGreedyTokenSelector` (below) needs no other change to stay correct.

**Where it lives in code**: `compute_semantic_coverage(G, T, D, lam=1.0)` — folds into the same
row-stochastic matrix, `A = lam * A` right after normalizing (Step 3), no other line changes.

Validated: `lam=1.0` reproduces the paper's own example exactly (backward compatible); on a plain
chain with `lam=0.7`, a token 3 hops away scored exactly `0.7³`.

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
   reused unchanged at every horizon; nothing about it depends on `h`. If a distance penalty
   `lam < 1` is used, `A` is scaled once more (`A = lam * A`) right here — the penalty is baked
   into the operator itself, so nothing below needs to know `lam` exists.
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
- **Distance/horizon penalty (`lam`)**: same extension as in `compute_semantic_coverage` (see
  above) — Eq. (7)'s third case gets one extra `lam *` factor. Since `lam` only ever multiplies
  the same submodular building block, the `(1-1/e)` guarantee still holds, and the incremental
  scheme stays exactly consistent with the full recompute for any `lam ∈ (0, 1]` (cross-checked
  numerically — see "Validated against" below).

### Implementation — `LazyGreedyTokenSelector` (`src/new_tokenizer.py`)

- **State kept per instance**: `G`, `M`, `D`, `lam` (distance penalty, default `1.0`), `V`,
  `node_to_idx`; `out_degree` = `d+(u)`, static since it never depends on `T`; `in_nodes` =
  `In(v)` (built once); `T` (current token set); `S` (current `(D+1, |V|)` coverage matrix, same
  layout as `compute_semantic_coverage`'s output); `current_score` = `F_D(T)` for the current `T`.
- **`candidate_delta(t)`** — thin wrapper around `_candidate_delta_core`, a plain/picklable
  function implementing Eq. (6)-(7) (pulled out so the exact same logic can run as an instance
  method or inside a worker process). Takes `lam` as an explicit argument (rather than reading
  `self.lam` directly) precisely so it stays a plain, picklable function usable by workers too.
- **`commit(t, delta)`** — applies `delta` onto `S`, adds `t` to `T` (COMMIT, §6.1/6.3).
- **`select(k, candidates=None, verbose=True, n_jobs=1, progress_every=5000)`**:
  1. **Seeding** — compute the *exact* initial gain for every not-yet-selected candidate ("initial
     gains are exact", §6.1) and push `(-gain, t)` onto a max-heap (`heapq` is a min-heap, so
     gains are negated). This is one `CANDIDATE_DELTA` call per candidate — `O(|candidates|)`, and
     the dominant one-time cost on a large graph (e.g. ~65s for all 75,635 real SNOMED candidates
     with `n_jobs=8`, vs. minutes serially). `progress_every` prints scan progress since this phase
     produces no other output; `n_jobs` (`-1` = all cores) parallelizes it across worker processes
     via `_init_worker`/`_worker_seed_gain`, since each candidate's initial gain only depends on the
     fixed `T`/`S`/`lam` at call time (embarrassingly parallel).
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
   *backward*), `lam` (distance penalty, default `1.0` = no penalty) — plus the mutable state
   `T` (empty at first) and `S` (all zero, consistent with `T = ∅`).
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
   `n_jobs != 1`, it spins up a `multiprocessing.Pool`, ships the read-only state (including
   `lam`) once per worker via `_init_worker`, and has each worker run `_worker_seed_gain(t)` — a
   thin wrapper around `_candidate_delta_core` that discards `delta` and returns only `(t, gain)`,
   since `delta` will be recomputed anyway if `t` ever reaches the top of the heap.
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
- `lam < 1`: ran `LazyGreedyTokenSelector(..., lam=0.6)` on a random 50-node cyclic graph, then
  cross-checked the resulting `T`'s score against an independent
  `compute_semantic_coverage(..., lam=0.6)` recomputation — identical scores, confirming the
  incremental delta propagation stays consistent with the modified equation, not just the full
  matrix path.

### Usage

```python
selector = new_tokenizer.LazyGreedyTokenSelector(combined_subgraphs, mapped_ids, D=3, lam=0.8)
history = selector.select(k=200, n_jobs=-1)   # n_jobs=-1: parallel seeding across all cores

selector.T                 # selected token concept ids
selector.current_score     # final F_D(T)
history                    # [(token, marginal_gain, cumulative_score), ...] in selection order
```

---

## Evaluation metrics (`src/eval.py`)

Six metrics that evaluate a selected `T`'s *realized* representation — complementary to semantic
coverage `F_D(T)`, which is what's used to *select* `T` in the first place (§4-6, above), not what
judges the outcome. All six are built on top of `tokenize_all_rel`'s context trees (§3), computed
once per concept and shared across metrics via `build_context_trees` / `evaluate`.

### Metrics

- **`conciseness`** — mean number of token leaves per concept's context tree. Lower = fewer
  tokens needed to represent a mapped concept = more compressed vocabulary.
- **`distance_score`** — mean hop-distance at which tokens are actually found, averaged per
  concept first (so concepts with many tokens don't dominate) then across concepts. Lower =
  tokens found closer to the concept being represented.
- **`uniqueness_entropy`** — Shannon entropy (bits, normalized to `[0,1]`) of the distribution of
  context-tree `signature()`s across `M`. `1.0` = every concept gets a distinct representation
  (maximally discriminative vocabulary); near `0` = most concepts collapse onto a handful of
  shared signatures (heavy collisions — the vocabulary can't tell them apart).
- **`unk_rate`** — fraction of concepts with at least one context marked `uncovered` (a semantic
  branch that never reached a token within depth `D`). Analogous to an out-of-vocabulary rate.
- **`tree_complexity`** — mean number of contexts (root + subcontexts) per concept's tree — a
  proxy for how structurally tangled the representation is.
- **`exact_rate`** — fraction of `M` that are themselves selected as a token (`c ∈ T`, a trivial
  0-hop representation).

### Worked example

5 mapped concepts, `T = {tokA, tokB, tokC}`, `D = 2`:

```
c1 --IS_A--> a --IS_A--> tokA
c2 --IS_A--> a  (same intermediate node)
c3 --FINDING_SITE--> tokB
c3 --SEVERITY--> tokC
c4 --FINDING_SITE--> z   (z is a dead end: no outgoing edges, not a token)
tokA  (this mapped concept IS a token itself)
```

Resulting context trees:
```
c1   -> tokA
c2   -> tokA
c3   -> FINDING_SITE(tokB), SEVERITY(tokC)
c4   -> FINDING_SITE(∅)          <- ∅ = uncovered
tokA -> tokA
```

Computed values: `conciseness=1.0`, `distance_score=1.25`, `uniqueness_entropy=0.590`,
`unk_rate=0.2`, `tree_complexity=1.6`, `exact_rate=0.2`.

- **`conciseness` = 1.0** — tokens per concept: `c1`→1, `c2`→1, `c3`→2, `c4`→0, `tokA`→1, mean
  `5/5`. `c3` needed 2 tokens (two separate aspects), pulling this up; watch it alongside
  `tree_complexity` to tell "few tokens, simple structure" apart from "few tokens, but only
  because branches died uncovered" (`c4`).
- **`distance_score` = 1.25** — per-concept mean token-distance: `c1`→2 (through `a`), `c2`→2,
  `c3`→1 (direct edges), `tokA`→0 (it *is* the token). `c4` is skipped — zero tokens found, no
  distance to average. Mean of `{2,2,1,0}` = `5/4`.
- **`uniqueness_entropy` = 0.590** — `c1`, `c2`, and `tokA` all produce the **identical**
  signature (root holding exactly `{tokA}`, no subcontexts), even though `tokA` reaches it at
  distance 0 and `c1`/`c2` at distance 2 — `signature()` is deliberately distance-blind, so it
  can't tell "I am the token" from "I resolve down to the token via a plain `IS_A` chain." That
  gives 3 groups out of 5 concepts — `{c1,c2,tokA}` (size 3), `{c3}`, `{c4}` — entropy
  `H=1.371` bits, normalized by `log2(5)=2.322` → `0.590`. All-distinct would be `1.0`;
  all-identical would be `0.0`.
- **`unk_rate` = 0.2** — only `c4` has an `uncovered` context (`z` is a genuine dead end within
  `D=2`) — `1/5`.
- **`tree_complexity` = 1.6** — contexts per concept: `c1`→1 (`IS_A` never opens a subcontext,
  it's transparent), `c2`→1, `c3`→3 (ROOT + `FINDING_SITE` + `SEVERITY`), `c4`→2 (ROOT +
  `FINDING_SITE`), `tokA`→1. Mean `8/5`. `c3`'s two real semantic aspects (site + severity) make
  it the most structurally rich concept here.
- **`exact_rate` = 0.2** — only `tokA` (as a mapped concept) is directly in `T` — `1/5`.

### Implementation

- **`_iter_contexts(ctx)`** — pre-order generator over a context tree (the node itself, then every
  nested subcontext) — the shared traversal every metric below is built on.
- **`build_context_trees(M, G, T, D, id_to_label=None)`** — calls `tokenize_all_rel` once per
  concept in `M`, returns `{concept: tree}`. Every metric function takes this dict, not `M`/`G`/`T`
  directly, so the (potentially expensive) tree construction only happens once regardless of how
  many metrics get computed.
- **`evaluate(M, G, T, D, id_to_label=None)`** — builds the trees once, then returns all six
  metrics in one dict.

### Wired into the parallel evaluation pipeline

`new_tokenizer._worker_coverage_score` (used by `7.evaluation_new.ipynb` to score every candidate
list from `configs_new_mapped.CandidateLists().all_candidate_to_test` in parallel) calls both
`compute_semantic_coverage`/`semantic_coverage_score` **and** `eval.evaluate` against the same
`T`, and merges the result into one dict — so every `(category, file_type, k)` row in the
resulting comparison table carries `semantic_coverage` alongside all six metrics above, computed
from the same shared `combined_subgraphs`/`mapped_ids` stashed once per worker by
`_init_coverage_worker` (now also carrying `id_to_label`).

### Usage

```python
from src import eval as tokeval
metrics = tokeval.evaluate(mapped_ids, combined_subgraphs, selector.T, D=3, id_to_label=id_to_label)
# {'conciseness': ..., 'distance_score': ..., 'uniqueness_entropy': ...,
#  'unk_rate': ..., 'tree_complexity': ..., 'exact_rate': ...}
```
