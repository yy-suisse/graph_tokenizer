import polars as pl
import numpy as np
from tqdm import tqdm

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


class IterativeMarginalGainMax:
    def __init__(
            self,
            mapped2candidate_rel_dist: pl.DataFrame,
            distance_type: str = "inv", 
            score_col: str = "dist_score",
            min_gain: float = 1e-8,
            max_candidates: int | None = 50_000,
            verbose_every: int = 1000):
        
        mapped2candidate_rel_dist = (
            mapped2candidate_rel_dist
            .rename({
                "src.id": "mapped_id",
                "dst.id": "candidate_id",
            })
            .group_by("mapped_id", "relation", "candidate_id")
            .agg(pl.col("distance").min())
        )

        if distance_type == "inv":
            self.mapped2candidate_rel_dist_score = mapped2candidate_rel_dist.with_columns(dist_score = 1/(1 + pl.col("distance")))

        else:
            self.mapped2candidate_rel_dist_score = mapped2candidate_rel_dist.with_columns(dist_score = 1/(np.exp(pl.col("distance"))))

        self.score_col = score_col
        self.min_gain = min_gain
        self.max_candidates = max_candidates
        self.verbose_every = verbose_every

    def __get_best_candidate_by_marginal_gain(self, selected_candidates_df, current_state):
        """
        Select the candidate with the highest marginal gain.

        Marginal gain is computed per (mapped_id, relation):

            marginal_gain = max(0, candidate_score - current_best_score)
        """

        candidate_pool = (
            self.mapped2candidate_rel_dist_score
            .join(selected_candidates_df, on="candidate_id", how="anti")
        )

        if candidate_pool.height == 0:
            return None

        gain_df = (
            candidate_pool
            .join(
                current_state,
                on=["mapped_id", "relation"],
                how="left",
            )
            .with_columns(
                pl.col("current_best_score").fill_null(0.0)
            )
            .with_columns(
                (
                    pl.col(self.score_col) - pl.col("current_best_score")
                )
                .clip(lower_bound=0.0)
                .alias("marginal_gain")
            )
            .filter(pl.col("marginal_gain") > 0)
            .group_by("candidate_id")
            .agg(
                pl.col("marginal_gain").sum().alias("total_gain"),
                pl.col("mapped_id").n_unique().alias("n_improved_mapped"),
                pl.col("relation").n_unique().alias("n_improved_relations"),
            )
            .sort("total_gain", descending=True)
        )

        if gain_df.height == 0:
            return None

        return gain_df.row(0, named=True)


    def __update_current_state(
            self,
            current_state: pl.DataFrame,
            selected_candidate,
        ):
        """
        Update current best score after selecting one candidate.
        """

        selected_scores = (
            self.mapped2candidate_rel_dist_score
            .filter(pl.col("candidate_id") == selected_candidate)
            .select(
                "mapped_id",
                "relation",
                pl.col(self.score_col).alias("new_score"),
            )
        )

        updated_state = (
            current_state
            .join(
                selected_scores,
                on=["mapped_id", "relation"],
                how="left",
            )
            .with_columns(
                pl.max_horizontal(
                    pl.col("current_best_score"),
                    pl.col("new_score").fill_null(0.0),
                ).alias("current_best_score")
            )
            .drop("new_score")
        )

        return updated_state

    def iterative_marginal_gain_greedy_without_k(self):
        """
        Marginal-gain greedy candidate selection without fixed K.

        Stop when:

            best_gain < stop_ratio * initial_gain

        Default:
            stop_ratio = 0.01

        Meaning:
            stop when the current best candidate contributes less than 1%
            of the first selected candidate's gain.
        """

        current_state = (
            self.mapped2candidate_rel_dist_score
            .select("mapped_id", "relation")
            .unique()
            .with_columns(
                pl.lit(0.0).alias("current_best_score")
            )
        )

        selected_candidates_df = pl.DataFrame(
            {"candidate_id": []},
            schema={"candidate_id": self.mapped2candidate_rel_dist_score.schema["candidate_id"]},
        )
        selected_rows = []

        initial_gain = None

        with tqdm(desc="Marginal gain greedy", mininterval=1.0, miniters=50) as pbar:

            while True:

                best_row = self.__get_best_candidate_by_marginal_gain(
                    selected_candidates_df=selected_candidates_df,
                    current_state=current_state,
                )

                if best_row is None:
                    print("No candidate with positive marginal gain.")
                    break

                best_candidate = best_row["candidate_id"]
                best_gain = best_row["total_gain"]

                if best_gain <= self.min_gain:
                    print(f"Stopping because gain is too small: {best_gain:.8f}")
                    break

                if initial_gain is None:
                    initial_gain = best_gain

                gain_ratio = best_gain / initial_gain



                selected_candidates_df = selected_candidates_df.vstack(
                    pl.DataFrame({"candidate_id": [best_candidate]}, schema=selected_candidates_df.schema)
                ).rechunk()

                selected_rows.append({
                    "rank": len(selected_rows) + 1,
                    "candidate_id": best_candidate,
                    "total_gain": best_gain,
                    "gain_ratio": gain_ratio,
                    "n_improved_mapped": best_row["n_improved_mapped"],
                    "n_improved_relations": best_row["n_improved_relations"],
                })

                current_state = self.__update_current_state(
                    current_state=current_state,
                    selected_candidate=best_candidate,
                )

                pbar.update(1)

                if self.verbose_every is not None and len(selected_rows) % self.verbose_every == 0:
                    mean_state_score = current_state["current_best_score"].mean()

                    print(
                        f"selected={len(selected_rows)}, "
                        f"best_gain={best_gain:.4f}, "
                        f"gain_ratio={gain_ratio:.4f}, "
                        f"mean_state_score={mean_state_score:.4f}"
                    )

                if self.max_candidates is not None and len(selected_rows) >= self.max_candidates:
                    print(f"Reached max_candidates={self.max_candidates}.")
                    break

        selected_candidate_df = pl.DataFrame(selected_rows)

        return selected_candidate_df, current_state