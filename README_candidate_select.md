# Candidate Selection — Method Comparison

How to pick the **vocabulary** (`candidate_list`) that [`GraphTokenizer`](src/tokenizer.py)
tokenizes the 46,150 mapped SNOMED concepts against, and how the different selection
strategies compare. See [README_tokenizer.md](README_tokenizer.md) for what the tokenizer
itself does and the exact definition of every score below — this document focuses on
*how candidates get selected* and *what that costs/buys* in terms of those scores.

All candidates are drawn from the same pool: every concept reachable from a mapped
concept within `max_dist_candidate = 3` hops (`configs.TokenizerParam().max_dist_candidate`),
75,669 unique concepts after the generic/"non-sense" candidates are pre-filtered out
(noted in [3.candidate_selection.ipynb](3.candidate_selection.ipynb)).

## Candidate selection methods

### Fixed-size baselines — [3.candidate_selection.ipynb](3.candidate_selection.ipynb)

| Method | Key | How it's ranked | Output |
|---|---|---|---|
| All mapped are candidates | — | every mapped concept is its own candidate | trivial ceiling reference (not a deployable vocabulary — zero compression) |
| Random-k | `b_random_k` | uniform random sample of size k from the full candidate pool, **50 independent draws per k** | `baselines_candidates/k_random_all_samples.parquet` (`k`, `iter`, `candidate_id`) |
| Highest degree ancestor | `b_highest_deg_list` | number of distinct mapped concepts reaching the candidate (in-degree), descending | `baselines_candidates/highest_degree.parquet` |
| Highest degree ancestor, distance == 1 | `b_highest_deg_dist_1_list` | same as above, restricted to direct parents only (`distance == 1`) | `baselines_candidates/highest_degree_dist_1.parquet` |
| Most children | `b_most_children_list` | number of reachable IS_A descendants (general candidates **not** removed) | `baselines_candidates/most_children.parquet` |

For all of these except `b_random_k`, "performance at k" = evaluate the **top-k** of the
ranked list. For `b_random_k`, performance at k = **mean score across the 50 simulations**
of that k (see [src/iterative_selection.py](src/iterative_selection.py) is unrelated here —
sampling is done inline in the notebook with `pl.Series.sample`).

### Greedy iterative methods — [2.iterative_process.ipynb](2.iterative_process.ipynb), [src/iterative_selection.py](src/iterative_selection.py)

Both operate on `mapped_candidate_rel_dist_prop` (mapped → candidate edges, `distance <= 3`),
converting `distance` into a similarity weight two ways:

- `inv`: `weight = 1 / (1 + distance)` — slow decay with distance
- `inv_exp`: `weight = 1 / exp(distance)` — fast decay with distance

**Iterative selection with graph reduction** (`iterative_approach_w_graph_red`): each round
picks the `(candidate, relation)` pair with the highest aggregated weight over
currently-uncovered mapped concepts, then permanently removes every row it (and its
IS_A-reachable descendant candidates) now cover — the table strictly shrinks each round,
so the algorithm self-terminates once nothing is left to cover (no fixed k). Two
aggregation rules:

- `sum`: total weighted coverage
- `tempered_sum`: total weighted coverage / `sqrt(number of distinct mapped concepts covered)`
  — penalizes candidates that only cover many concepts shallowly

| Key | Aggregation | Decay | Candidates selected (natural stopping point) |
|---|---|---|---|
| `gr_sum_inv` | sum | inv | 21,880 |
| `gr_sum_inv_exp` | sum | inv_exp | 25,983 |
| `gr_tempered_sum_inv` | tempered_sum | inv | 24,671 |
| `gr_tempered_sum_inv_exp` | tempered_sum | inv_exp | 36,453 |

Stored at `graph_red_candidates/ {sc_type}_{d_type}.parquet`, row order = selection order.

**Marginal-gain greedy** (`IterativeMarginalGainMax.iterative_marginal_gain_greedy_without_k`):
unlike the graph-reduction version, the table is **never** shrunk. Each round picks the
candidate maximizing total marginal gain, `sum(max(0, score - current_best_score))` over
all `(mapped_id, relation)` pairs, where `current_best_score` is the best score any
already-selected candidate has produced for that pair so far.

| Key | Decay | Candidates selected |
|---|---|---|
| `mg_inv` | inv | 46,150 |
| `mg_inv_exp` | inv_exp | 46,150 |

Both runs stopped at `max_candidates = number of unique mapped concepts` — **the cap was
hit, not the gain threshold** (`min_gain`) — so in principle both could keep selecting
further candidates with diminishing returns past this point. Stored at
`margin_gain_candidates/_{distance_type}.parquet`, with an explicit `rank` column.

## Evaluation — [4.evaluation_performance.ipynb](4.evaluation_performance.ipynb)

Every method is evaluated at `k = 500, 1000, ..., 11500` (23 points) via
`tokenizer.evaluate_components_and_tokenize(candidate_ids[:k])` — i.e. all methods are
compared on equal footing at the *same nominal k values*, even though the two
graph-reduction-style methods naturally produce more candidates than `k=11500` if left to
run to completion.

> **Important:** `k` is the *requested* vocabulary size, not the *actual* number of
> candidates used. Methods that don't account for redundancy (e.g. hierarchy pruning)
> may assign fewer unique candidates than k. All charts and comparisons in the app use
> `num_candidates` (the actual unique assigned candidates) on the x-axis, not k.

Scores (full definitions in [README_tokenizer.md](README_tokenizer.md#scores)):

| Score | Direction | One-line meaning |
|---|---|---|
| `final_score` | higher better | **combined score** = `distance_score × uniqueness_entropy_score × sem_cov_score` — joint reward for closeness, diversity, and semantic coverage |
| `sem_cov_score` | higher better | mean fraction of each concept's relation types covered by the vocabulary |
| `distance_score` | higher better | how close (in hops) assigned tokens are to the concept they represent |
| `uniqueness_entropy_score` | higher = more diverse | how evenly distinct the token assignments are across concepts |
| `conciseness_score` | higher = fewer tokens | inverse of mean tokens-per-concept (verbosity, not quality) |
| `compression_rate` | lower = smaller vocab | vocabulary size / number of mapped concepts |
| `UNK_rate` | lower better | fraction of concepts that got no coverage at all |
| `exact_rate` | higher = less abstraction | fraction of concepts that are exact self-matches |

Results are written to `results/scores.parquet` (`method`, `k`, `num_candidates`, + the 7 scores above).
`final_score` is computed on the fly from the three component scores.

## Results

Leaderboard (top-3 methods) at three representative nominal k values.
Note that actual `num_candidates` may differ per method at the same k (see note above).
For a combined ranking, see `final_score` = `distance_score × uniqueness_entropy_score × sem_cov_score` in the app.

**`sem_cov_score`**

| k | 1st | 2nd | 3rd |
|---|---|---|---|
| 500 | `mg_inv` (0.863) | `gr_sum_inv` (0.816) | `gr_tempered_sum_inv` (0.810) |
| 5500 | `mg_inv` (0.995) | `gr_tempered_sum_inv` (0.989) | `gr_sum_inv_exp` (0.989) |
| 11500 | `gr_sum_inv_exp` (0.999) | `mg_inv` (0.999) | `gr_sum_inv` (0.998) |

**`distance_score`**

| k | 1st | 2nd | 3rd |
|---|---|---|---|
| 500 | `b_highest_deg_dist_1_list` (0.586) | `mg_inv_exp` (0.582) | `mg_inv` (0.556) |
| 5500 | `mg_inv_exp` (0.698) | `mg_inv` (0.691) | `b_highest_deg_dist_1_list` (0.688) |
| 11500 | `mg_inv_exp` (0.758) | `mg_inv` (0.753) | `b_highest_deg_dist_1_list` (0.733) |

**`UNK_rate`** (lower is better — `b_random_k` shown as worst-case reference)

| k | best | 2nd best | `b_random_k` |
|---|---|---|---|
| 500 | `b_highest_deg_dist_1_list` (0.093) | `b_most_children_list` (0.096) | 0.890 |
| 5500 | `gr_tempered_sum_inv_exp` (0.0104) | `b_highest_deg_dist_1_list` (0.0106) | 0.287 |
| 11500 | `b_highest_deg_list` (0.0015) | `b_highest_deg_dist_1_list` (0.0032) | 0.097 |

**`exact_rate`** (higher = less abstraction needed)

| k | 1st | 2nd | 3rd |
|---|---|---|---|
| 500 | `b_random_k` (0.007) | `b_most_children_list` (0.003) | `mg_inv_exp` (0.003) |
| 5500 | `gr_tempered_sum_inv_exp` (0.094) | `b_random_k` (0.073) | `mg_inv_exp` (0.068) |
| 11500 | `mg_inv_exp` (0.178) | `mg_inv` (0.177) | `gr_tempered_sum_inv_exp` (0.175) |

### Takeaways

- **`b_random_k` is, as expected, dominated on every coverage/distance/UNK metric** at
  every k — it exists as a noise floor, confirming the other methods' structure-aware
  ranking is doing real work. It only "wins" on `conciseness_score` and small-k
  `exact_rate`, both somewhat degenerate: a small random sample rarely produces more than
  one candidate per concept (inflating conciseness) and a sliver of random self-matches
  rather than reflecting genuinely good targeted selection.
- **Marginal-gain (`mg_inv`, `mg_inv_exp`) is consistently strong on `sem_cov_score` and
  `distance_score`** across the whole k range, and pulls ahead on `exact_rate` at larger k
  — consistent with its design (always extending the pair with the *currently weakest*
  coverage, rather than re-deriving a static rank).
- **`b_highest_deg_dist_1_list` is the strongest single fixed-rank baseline**, competitive
  with or ahead of the greedy methods on `distance_score` and `UNK_rate` — direct parents
  alone already cover a lot of ground cheaply, before abstraction kicks in.
- **Graph-reduction (`gr_*`) variants are competitive on `sem_cov_score`/`UNK_rate` at
  larger k** and `gr_tempered_sum_inv_exp` in particular leads `exact_rate` at mid-range k
  — the `tempered_sum` aggregation (penalizing shallow-but-wide picks) appears to favor
  more specific, exact-match-prone candidates earlier than plain `sum`.
- No method dominates on every score simultaneously — `conciseness_score` /
  `uniqueness_entropy_score` trade off against `sem_cov_score` / `exact_rate` by
  construction (more candidates → better coverage but worse compression), so the right k
  and method depend on which axis matters more for the downstream use case.

## Where the outputs live

| Artifact | Path (`src/configs.py`) |
|---|---|
| Baseline candidate lists / random-k samples | `configs.Baselines().path` → `D:/tokenizer_graph/baselines_candidates/` |
| Graph-reduction candidate lists | `configs.IterativeGraphRed().path` → `D:/tokenizer_graph/graph_red_candidates/` |
| Marginal-gain candidate lists | `configs.IterativeMarginalGain().path` → `D:/tokenizer_graph/margin_gain_candidates/` |
| Final performance table | `configs.Results().path` → `D:/tokenizer_graph/results/scores.parquet` |
