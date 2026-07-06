import argparse
import pickle
import itertools

import polars as pl
import numpy as np
from tqdm import tqdm


import src.graph_fct as graph_fct
import src.utils as ut
from src.tokenizer import GraphTokenizer
from src.iterative_selection import iterative_approach_w_graph_red, IterativeMarginalGainMax


# ---------------------------------------------------------------------------
# Step 0 — set the concepts to be decomposed
# ---------------------------------------------------------------------------

_parser = argparse.ArgumentParser()
_parser.add_argument("--config", choices=["mapped", "fd"], default="mapped")
_parser.add_argument("--prim", action="store_true", default=False)
_args, _ = _parser.parse_known_args()

concept_to_be_tokenized = _args.config
prim_candidate_only = _args.prim
# ---------------------------------------------------------------------------

if concept_to_be_tokenized == "mapped":
    import src.configs_mapped as configs
    df_concept_hug = pl.read_parquet(configs.GraphConfig.concept_path)
elif concept_to_be_tokenized == "fd":
    import src.configs_fd as configs
    df_concept_hug = pl.read_parquet(configs.GraphConfig.concept_path)

    fd_list_1 = (
               pl.read_parquet(configs.GraphConfig().official_release_path)
                 .filter(pl.col("status") == "defined")["id"].unique().to_list()
               )
    fd_list_2 = (
               df_concept_hug.filter(pl.col("concept_type") == "SCT_POST")["id"].to_list()
               )
    fd_list = fd_list_1 + fd_list_2
    
    prim_list = (
               pl.read_parquet(configs.GraphConfig().official_release_path)
                 .filter(pl.col("status") == "primitive")["id"].unique().to_list()
               )
    df_concept_hug = (
                df_concept_hug
                .with_columns(is_mapped = pl.col("id").is_in(fd_list))
                      )
    

# ---------------------------------------------------------------------------
# Step 1 — Graph preparation
# ---------------------------------------------------------------------------

df_relation = pl.read_parquet(configs.GraphConfig.relation_path)
g_is_a = graph_fct.get_is_a_graph(df_relation)

df_concept_hug_w_depth = graph_fct.get_max_min_depth_snomed(G=g_is_a, df_all_concepts=df_concept_hug)
df_concept_hug_w_depth.write_parquet(configs.ProcessedGraph.concept_w_depth)

concept_propagate = ut.ConceptPropagate(df_relation, df_concept_hug_w_depth)
mapped_candidate_rel_dist_prop_full = concept_propagate.build_all_distance_and_relations()

if prim_candidate_only:
    mapped_candidate_rel_dist_prop_full = (mapped_candidate_rel_dist_prop_full
                                           .filter(pl.col("dst.id").is_in(prim_list)))
    
mapped_candidate_rel_dist_prop_full.write_parquet(configs.ProcessedGraph().mapped_candidate_rel_dist_prop)

candidate_reachable_child_map = ut.precompute_candidate_reachable_child_map(
    g_is_a,
    mapped_candidate_rel_dist_prop_full["dst.id"].unique(),
)
with open(configs.ProcessedGraph().candidate_is_a_reachable_dict, "wb") as f:
    pickle.dump(candidate_reachable_child_map, f)

# Shared filtered view used by steps 2-4
mapped_candidate_rel_dist_prop = (
    mapped_candidate_rel_dist_prop_full
    .filter(pl.col("distance") <= configs.TokenizerParam().max_dist_candidate)
)

"""
# ---------------------------------------------------------------------------
# Step 2 — Iterative candidate ranking
# ---------------------------------------------------------------------------

distance_score_types = ["sum", "tempered_sum"]
distance_types = ["inv", "inv_exp"]

for sc_type, d_type in itertools.product(distance_score_types, distance_types):
    candidate_rows = iterative_approach_w_graph_red(
        mapped_candidate_rel_dist_prop,
        candidate_reachable_child_map,
        sc_type,
        d_type,
    )
    pl.DataFrame(candidate_rows).write_parquet(
        f"{configs.IterativeGraphRed().path}{sc_type}_{d_type}.parquet"
    )
    print(f"graph_red {sc_type}/{d_type}: {len(candidate_rows)} candidates")

max_candidates = mapped_candidate_rel_dist_prop["src.id"].n_unique()

for d_type in distance_types:
    margin_gain_iterative = IterativeMarginalGainMax(
        mapped_candidate_rel_dist_prop, d_type, max_candidates=max_candidates
    )
    selected_marginal_gain, _ = margin_gain_iterative.iterative_marginal_gain_greedy_without_k()
    selected_marginal_gain.write_parquet(f"{configs.IterativeMarginalGain().path}_{d_type}.parquet")
    print(f"marginal_gain {d_type}: done")


# ---------------------------------------------------------------------------
# Step 3 — Baseline candidate sets
# ---------------------------------------------------------------------------
Ks = np.arange(500, 12000, 500)

tokenizer = GraphTokenizer(
    df_concept_hug_w_depth,
    mapped_candidate_rel_dist_prop,
    candidate_reachable_child_map,
    configs.TokenizerParam().max_dist_candidate,
)

n_samples = 50
candidates_all = pl.Series(tokenizer.candidate_concepts)

samples = [
    pl.DataFrame({"k": k, "iter": s, "candidate_id": candidates_all.sample(n=k)})
    for k in Ks
    for s in range(n_samples)
]
pl.concat(samples).write_parquet(f"{configs.Baselines().path}k_random_all_samples.parquet")

(
    mapped_candidate_rel_dist_prop
    .group_by("dst.id")
    .agg(num_cpts=pl.col("src.id").n_unique())
    .sort("num_cpts", descending=True)
    .with_row_index()
).write_parquet(f"{configs.Baselines().path}highest_degree.parquet")

(
    mapped_candidate_rel_dist_prop
    .filter(pl.col("distance") == 1)
    .group_by("dst.id")
    .agg(num_cpts=pl.col("src.id").n_unique())
    .sort("num_cpts", descending=True)
    .with_row_index()
).write_parquet(f"{configs.Baselines().path}highest_degree_dist_1.parquet")

rows = [
    {"candidate": c, "num_reachable_child": len(v)}
    for c, v in candidate_reachable_child_map.items()
]
(
    pl.DataFrame(rows)
    .sort("num_reachable_child", descending=True)
    .with_row_index()
).write_parquet(f"{configs.Baselines().path}most_children.parquet")
"""

# ---------------------------------------------------------------------------
# Step 4 — Evaluation
# ---------------------------------------------------------------------------
Ks = np.arange(500, 12000, 500)
tokenizer = GraphTokenizer(
    df_concept_hug_w_depth,
    mapped_candidate_rel_dist_prop,
    df_relation,
    candidate_reachable_child_map,
    configs.TokenizerParam().max_dist_candidate,
)

baseline_candidates = {
    "b_highest_deg_list": pl.read_parquet(f"{configs.Baselines().path}highest_degree.parquet")["dst.id"].to_list(),
    "b_highest_deg_dist_1_list": pl.read_parquet(f"{configs.Baselines().path}highest_degree_dist_1.parquet")["dst.id"].to_list(),
    "b_most_children_list": pl.read_parquet(f"{configs.Baselines().path}most_children.parquet")["candidate"].to_list(),
    "b_random_k_all": pl.read_parquet(f"{configs.Baselines().path}k_random_all_samples.parquet"),
}

graph_red_candidates = {
    "gr_sum_inv": pl.read_parquet(f"{configs.IterativeGraphRed().path}sum_inv.parquet")["candidate"].to_list(),
    "gr_sum_inv_exp": pl.read_parquet(f"{configs.IterativeGraphRed().path}sum_inv_exp.parquet")["candidate"].to_list(),
    "gr_tempered_sum_inv": pl.read_parquet(f"{configs.IterativeGraphRed().path}tempered_sum_inv.parquet")["candidate"].to_list(),
    "gr_tempered_sum_inv_exp": pl.read_parquet(f"{configs.IterativeGraphRed().path}tempered_sum_inv_exp.parquet")["candidate"].to_list(),
}

margin_gain_candidates = {
    "mg_inv": pl.read_parquet(f"{configs.IterativeMarginalGain().path}_inv.parquet")["candidate_id"].to_list(),
    "mg_inv_exp": pl.read_parquet(f"{configs.IterativeMarginalGain().path}_inv_exp.parquet")["candidate_id"].to_list(),
}

results = []

ordered_candidate_sets = {
    **{name: ids for name, ids in baseline_candidates.items() if name != "b_random_k_all"},
    **graph_red_candidates,
    **margin_gain_candidates,
}

for name, ordered_ids in ordered_candidate_sets.items():
    print(name)
    for k in tqdm(Ks):
        scores, _, _ = tokenizer.evaluate_components_and_tokenize(ordered_ids[:k])
        results.append({"method": name, "k": k, **scores})

random_k_df = baseline_candidates["b_random_k_all"]

for k in tqdm(Ks, desc="b_random_k"):
    iter_scores = []
    for s in random_k_df.filter(pl.col("k") == k)["iter"].unique().sort():
        candidate_ids = random_k_df.filter(
            (pl.col("k") == k) & (pl.col("iter") == s)
        )["candidate_id"].to_list()
        scores, _, _ = tokenizer.evaluate_components_and_tokenize(candidate_ids)
        iter_scores.append(scores)
    mean_scores = pl.DataFrame(iter_scores).mean().to_dicts()[0]
    results.append({"method": "b_random_k", "k": k, **mean_scores})

df_performance = pl.DataFrame(results)
df_performance.write_parquet(f"{configs.Results().path}scores.parquet")
print("Pipeline complete. Scores saved.")
