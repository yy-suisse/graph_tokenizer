import polars as pl
import numpy as np


# 1. get info of best candidate, relation and covered mapped concepts
def __get_best_candidate_and_covered_mapped_concepts(mapped2candidate_rel_dist_score_copy, distance_score_type = "sum"):
    if distance_score_type == "sum":
        best_candidate_row = mapped2candidate_rel_dist_score_copy.group_by("candidate_id", "relation").agg(pl.col("mapped_id"), pl.col("dist_score").sum()).sort("dist_score", descending=True).head(1)

    if distance_score_type == "mean":
        best_candidate_row = mapped2candidate_rel_dist_score_copy.group_by("candidate_id", "relation").agg(pl.col("mapped_id"), pl.col("dist_score").mean()).sort("dist_score", descending=True).head(1)

    if distance_score_type == "tempered_sum":
        best_candidate_row = (mapped2candidate_rel_dist_score_copy
                              .group_by("candidate_id", "relation")
                              .agg(
                                  pl.col("mapped_id"),
                                  pl.col("dist_score").sum(),
                                  pl.col("mapped_id").n_unique().alias("n_covered")
                                  )
                              .with_columns(
                                    (pl.col("dist_score") / (pl.col("n_covered") ** 0.5)).alias("tempered_score")
                                )
                              .sort("tempered_score", descending=True).head(1)
                              )


    candidate = best_candidate_row["candidate_id"][0]
    relation = best_candidate_row["relation"][0]
    mapped_ids = best_candidate_row["mapped_id"][0]
    return candidate, relation, list(mapped_ids)

def __update_connectivity(mapped2candidate_rel_dist_score_copy, candidate, relation, mapped_ids, candidate_reachable_child_map):

    return (mapped2candidate_rel_dist_score_copy
                .filter(
                    ~((pl.col("candidate_id")==candidate)
                    &
                    (pl.col("mapped_id").is_in(mapped_ids))
                    &
                    (pl.col("relation") == relation))

                )
                .filter(
                    ~((pl.col("candidate_id").is_in(candidate_reachable_child_map[candidate]))
                    &
                    (pl.col("mapped_id").is_in(mapped_ids))
                    &
                    (pl.col("relation") == relation))

                )
            )

def iterative_approach_w_graph_red(mapped2candidate_rel_dist, candidate_reachable_child_map, distance_score_type, distance_type = "inv"):
    candidate_rows = []

    mapped2candidate_rel_dist = (mapped2candidate_rel_dist
                                        .rename({
                                            "src.id": "mapped_id",
                                            "dst.id": "candidate_id",
                                        })
                                      )
    
    if distance_type == "inv":
        mapped2candidate_rel_dist_score = mapped2candidate_rel_dist.with_columns(dist_score = 1/(1 + pl.col("distance")))
    
    else:
        mapped2candidate_rel_dist_score = mapped2candidate_rel_dist.with_columns(dist_score = 1/(np.exp(pl.col("distance"))))


    mapped2candidate_rel_dist_score_copy = mapped2candidate_rel_dist_score.clone()

    while len(mapped2candidate_rel_dist_score_copy) > 0:
        n_before = len(mapped2candidate_rel_dist_score_copy)

        candidate, relation, mapped_ids = __get_best_candidate_and_covered_mapped_concepts(
            mapped2candidate_rel_dist_score_copy,
            distance_score_type
        )

        if len(mapped_ids) == 0:
            break

        candidate_rows.append({
            "candidate": candidate,
            "relation": relation,
            "mapped_ids": mapped_ids,
        })

        mapped2candidate_rel_dist_score_copy = __update_connectivity(
            mapped2candidate_rel_dist_score_copy,
            candidate,
            relation,
            mapped_ids,
            candidate_reachable_child_map,
        )

        n_after = len(mapped2candidate_rel_dist_score_copy)

        if n_before - n_after == 0:
            break

        if len(candidate_rows) % 500 == 0:
            print(f"number of candidates selected: {len(candidate_rows)}, number of rows remaining: {n_after}")

    return candidate_rows
