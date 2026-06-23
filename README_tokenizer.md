# Graph Tokenizer

Tokenizes SNOMED concepts against a chosen vocabulary ("candidate") of concepts, by
walking the IS_A hierarchy and other relation types to find, for every mapped concept,
the closest candidate(s) that best represent it.

## Pipeline

### Stage 1 — graph preparation ([1.graph_prepare.ipynb](1.graph_prepare.ipynb), [src/graph_fct.py](src/graph_fct.py), [src/utils.py](src/utils.py))

1. **Build the IS_A graph.** `graph_fct.get_is_a_graph(df_relation)` builds a directed
   graph with edges `child -> parent` from the `IS_A` rows of the relation table.

2. **Compute concept depth.** `graph_fct.get_max_min_depth_snomed(G, df_all_concepts)`
   walks the IS_A graph to compute, per concept, the `max_depth` (longest path to root)
   and `min_depth` (shortest path to root). Concepts with no path to root (missing from
   `G`, or otherwise disconnected) are dropped from the output rather than kept with a
   placeholder depth.

3. **Propagate relations with distance.** `ConceptPropagate(df_relation, df_concept_w_depth).build_all_distance_and_relations()`
   produces `mapped_candidate_rel_dist_prop`: a table of `(src.id, dst.id, relation, distance)`
   describing how far every concept is from each *mapped* concept, via:
   - direct relations (`distance = 1`),
   - a self-loop for every mapped concept (`distance = 0`),
   - relations propagated outward from each mapped concept's neighbors via BFS on the
     IS_A graph (`distance = BFS distance + 1`).
   - `Relation`: type inferred from the hop-1 direct neighbor (e.g. IS_A, FINDING_SITE) — literal only at distance == 1; candidates beyond that are the ancestor concepts on the IS_A path from that direct neighbor up to the SNOMED root, not literally connected to src.id by relation.

4. **Precompute hierarchy reachability.** `precompute_candidate_reachable_child_map(g_is_a, candidate_ids)`
   builds `dict[candidate_id, set[descendant_candidate_ids]]` — for every candidate
   concept, which other candidates are its IS_A descendants. Used later to prune
   redundant parent candidates when a more specific descendant candidate is already
   selected for the same relation type.

### Stage 2 — tokenization ([2.iterative_process.ipynb](2.iterative_process.ipynb), [src/tokenizer.py](src/tokenizer.py))

Before constructing `GraphTokenizer`, `mapped_candidate_rel_dist_prop` is **pre-filtered
to `distance <= max_dist_candidate`** (`configs.TokenizerParam().max_dist_candidate`):

```python
mapped_candidate_rel_dist_prop = (
    pl.read_parquet(configs.ProcessedGraph().mapped_candidate_rel_dist_prop)
    .filter(pl.col("distance") <= configs.TokenizerParam().max_dist_candidate)
)
```

`GraphTokenizer.__init__` applies the same `distance <= max_dist_candidate` filter again
internally when building `self.edges_within_max`. The two filters are redundant as long
as the same threshold is used in both places — the upstream filter exists purely to
shrink the table before it's loaded into the class. This only matters for the *partial*
matching path (which works off `distance <= max_dist_candidate` edges); it does **not**
affect `sem_cov_base` (the semantic-coverage baseline), since that's built from
`distance == 1` rows only, and any `max_dist_candidate >= 1` leaves those rows untouched.

## How a concept gets tokenized

`GraphTokenizer.evaluate_components_and_tokenize(candidate_list)` runs the full
pipeline for a given vocabulary (`candidate_list`). Every mapped concept ends up in
exactly one of four buckets:

| Bucket | Condition | Resulting token(s) |
|---|---|---|
| **Exact** | the concept itself is in `candidate_list` | itself (distance 0) |
| **UNK** | none of its relation types at `distance == 1` (in the *full*, unfiltered graph) are covered by any selected candidate | a single `"UNK"` placeholder, `distance = null` |
| **Partial, single candidate** | for a given relation type, exactly one selected candidate is reachable | that candidate, at the minimum distance found |
| **Partial, multiple candidates** | for a given relation type, more than one selected candidate is reachable | the reachable candidates, after pruning any candidate that is an IS_A ancestor of another reachable candidate for that same relation (`__filter_candidates_by_hierarchy`) |

Concretely, in `get_tokenizer`:
1. `df_tok_exact` — exact matches.
2. `df_tok_unk` — concepts whose semantic coverage fraction (`frac_sem_cov`, see below) is `0.0`.
3. Everything else is grouped by `(concept, relation)`. Groups with one reachable
   candidate pass through directly; groups with several go through the hierarchy filter
   to drop redundant ancestor candidates.
4. All four pieces are concatenated into `df_tok_all`. With `debug=True` (the default),
   two invariants are asserted: every concept lands in exactly one of the buckets, and
   the full set of mapped concepts is accounted for.

## Scores

All scores are returned from `evaluate_components_and_tokenize` / `tokenize` in the
`scores` dict.

### `sem_cov_score` — semantic coverage

For each mapped concept, `num_relation_type` is the number of distinct relation types
(`IS_A`, `FINDING_SITE`, `CAUSATIVE_AGENT`, ...) it has at `distance == 1` in the
**full, unfiltered** graph (the baseline — independent of which candidates are chosen).
`real_num_relation_type` is the same count, but restricted to relation types actually
reachable through the *currently selected* candidates.

```
frac_sem_cov(concept) = real_num_relation_type / num_relation_type      (non-exact concepts)
frac_sem_cov(concept) = 1.0                                              (exact concepts)

sem_cov_score = mean(frac_sem_cov) over all mapped concepts
```

UNK concepts have `frac_sem_cov = 0.0` and are included in the mean, so this score
already reflects unmapped concepts correctly — no separate UNK penalty is needed here.

### `distance_score` — how close the assigned tokens are

```
mean_distance(concept) = mean(distance) over the concept's assigned token rows
distance_score = 1 - mean(mean_distance over all concepts) / (max_dist_candidate + 1)
```

UNK concepts have no real distance (`null`). Rather than letting the mean silently skip
them, their per-concept distance is filled with the worst-case value
`max_dist_candidate + 1` before averaging, so a high UNK rate actively lowers this score.

### `uniqueness_entropy_score` — token-assignment diversity

Mapped concepts are grouped by their exact set of assigned candidate ids
(`candidate_id_list`). Let `p_g` be the fraction of all mapped concepts that fall into
group `g`:

```
entropy = -sum_g( p_g * log(p_g) )
max_entropy = log(total_mapped_concepts)        (or 1.0 if there's only one group)
uniqueness_entropy_score = entropy / max_entropy
```

A score near 1 means concepts mostly get distinguishable token combinations (little
redundancy); near 0 means many concepts collapse onto identical token sets. Concepts
that are all `UNK` are grouped together like any other shared token set — this score
does not separately distinguish "real" redundancy from "didn't get tokenized."

### `conciseness_score` — sequence expansion

This is a **verbosity** measure, not a quality measure: it reflects how much a sequence
of concepts would expand if each concept were replaced by its assigned token(s) —
independent of whether those tokens are a good semantic match.

```
num_tokens(concept) = number of distinct candidate ids assigned to it
conciseness_score = 1 / mean(num_tokens over all mapped concepts)
```

A `UNK` concept still expands to exactly one token (the `UNK` placeholder itself), so it
is included in this average like any other concept, contributing `num_tokens = 1`.

### `compression_rate`

```
compression_rate = n_unique(assigned_candidate_ids) / len(mapped_concepts)
```

`assigned_candidate_ids` is the set of distinct candidate IDs that appear in the final
tokenization (including `"UNK"` if any concept is unmapped, but excluding candidates from
`candidate_list` that were never assigned to any concept). Smaller means more compression.

### `UNK_rate`

```
UNK_rate = count(concepts assigned only "UNK") / len(mapped_concepts)
```

### `exact_rate`

```
exact_rate = count(concepts that are exact self-matches) / len(mapped_concepts)
```

## Notes / known interactions between scores

- `sem_cov_score` and `distance_score` are corrected for `UNK` (a high UNK rate actively
  lowers both). 
- `uniqueness_entropy_score` and `conciseness_score` are not "quality"
  scores in that sense — `conciseness_score` by design measures expansion regardless of
  quality
- `uniqueness_entropy_score` currently treats all-`UNK` concepts as one
  shared group rather than excluding them.

