import math
import networkx as nx
import polars as pl


class GraphTokenizer:
    def __init__(
        self,
        concept_connected_subgraph: pl.DataFrame,
        concept_candidate_dist_n_rel: pl.DataFrame,
        candidate_reachable_child_map: dict | None = None,
        max_dist_candidate: int = 10000,
    ):
        self.max_dist_candidate = max_dist_candidate

        self.mapped_id_list = concept_connected_subgraph.filter(pl.col("is_mapped"))["id"].to_list()
        self.mapped_id_set = set(self.mapped_id_list)

        self.candidate_concepts = concept_candidate_dist_n_rel["dst.id"].unique().to_list()

        if candidate_reachable_child_map is None:
            print(
                "Warning: candidate_reachable_child_map is None. Hierarchy-based parent removal will be disabled.",
            )
        self.candidate_reachable_child_map = candidate_reachable_child_map or {}

        # Main reusable table: only mapped sources, valid candidates, and allowed distance.
        self.edges_within_max = (
            concept_candidate_dist_n_rel.filter(pl.col("src.id").is_in(self.mapped_id_list))
            .filter(pl.col("dst.id").is_in(self.candidate_concepts))
            .filter(pl.col("distance") <= self.max_dist_candidate)
        )

        # Exact edges, precomputed once.
        self.exact_edges = self.edges_within_max.filter(pl.col("src.id") == pl.col("dst.id")).rename({"src.id": "mapped_id", "dst.id": "candidate_id"})
        self.exact_candidate_set = set(self.exact_edges["mapped_id"].unique().to_list())

        # Baseline semantic type count per mapped concept, independent of candidate list.
        self.sem_cov_base = (
            concept_candidate_dist_n_rel.filter(pl.col("src.id").is_in(self.mapped_id_list))
            .filter(pl.col("distance") == 1)
            .group_by("src.id")
            .agg(pl.col("relation").n_unique().alias("num_relation_type"))
            .rename({"src.id": "mapped_id"})
        )

    def __filter_candidates_by_hierarchy(self, candidate_list):
        """
        Remove parent candidates if any reachable child candidate is also present.

        Uses precomputed:
            candidate -> all reachable child candidates

        This avoids recursive graph search during GA evaluation.
        """
        candidate_set = set(candidate_list)
        to_remove = set()

        for candidate in candidate_set:
            reachable_children = self.candidate_reachable_child_map.get(candidate, set())

            if reachable_children & candidate_set:
                to_remove.add(candidate)

        return [c for c in candidate_list if c not in to_remove]

    def __check_eval_mapping_level(self, exact_mapped, non_exact):
        exact_set = set(exact_mapped)
        non_exact_set = set(non_exact)

        assert exact_set.isdisjoint(non_exact_set)
        assert len(self.mapped_id_list) == len(exact_set) + len(non_exact_set)

    def __eval_mapping_exact(self, candidate_set, debug=True):
        exact_mapped = list(self.mapped_id_set & candidate_set & self.exact_candidate_set)
        non_exact_mapped = list(self.mapped_id_set - set(exact_mapped))

        if debug:
            self.__check_eval_mapping_level(exact_mapped, non_exact_mapped)

        return exact_mapped, non_exact_mapped

    def __eval_coverage_n_score(
        self,
        exact_mapped,
        non_exact_mapped,
        selected_edges,
        debug=True,
    ):
        sem_cov_non_exact = self.sem_cov_base.filter(
            pl.col("mapped_id").is_in(non_exact_mapped),
        )

        sem_cov_non_exact_real = (
            selected_edges.filter(pl.col("src.id").is_in(non_exact_mapped)).group_by("src.id").agg(pl.col("relation").n_unique().alias("real_num_relation_type")).rename({"src.id": "mapped_id"})
        )

        sem_cov_non_exact_score = (
            sem_cov_non_exact.join(sem_cov_non_exact_real, on="mapped_id", how="left")
            .fill_null(0.0)
            .with_columns(
                frac_sem_cov=pl.col("real_num_relation_type") / pl.col("num_relation_type"),
            )
            .select("mapped_id", "frac_sem_cov")
        )

        sem_cov_exact_score = pl.DataFrame(
            {"mapped_id": exact_mapped, "frac_sem_cov": 1.0},
        )

        sem_cov = pl.concat(
            [sem_cov_non_exact_score, sem_cov_exact_score],
            how="diagonal_relaxed",
        )

        sem_cov_score = sem_cov["frac_sem_cov"].mean()

        if debug:
            self.__check_eval_mapping_level(
                sem_cov_exact_score["mapped_id"].unique(),
                sem_cov_non_exact_score["mapped_id"].unique(),
            )

        return sem_cov, sem_cov_score

    def get_tokenizer(self, exact_mapped, sem_cov, selected_edges, debug=True):
        df_tok_exact = self.exact_edges.filter(
            pl.col("mapped_id").is_in(exact_mapped),
        )

        df_tok_unk = (
            sem_cov.filter(pl.col("frac_sem_cov") == 0.0)
            .with_columns(
                candidate_id=pl.lit("UNK"),
                relation=pl.lit("IS_A"),
                distance=pl.lit(None, dtype=pl.Float64),
            )
            .drop("frac_sem_cov")
        )

        partial_mapped_id = list(
            self.mapped_id_set - set(df_tok_exact["mapped_id"]) - set(df_tok_unk["mapped_id"]),
        )

        partial_edges = selected_edges.filter(
            pl.col("src.id").is_in(partial_mapped_id),
        )

        partial_mapped = partial_edges.group_by("src.id", "relation").agg(pl.col("dst.id").unique()).with_columns(candidate_same_sem_tag=pl.col("dst.id").list.n_unique())

        df_tok_partial_one_candidate = (
            partial_mapped.filter(pl.col("candidate_same_sem_tag") == 1)
            .explode("dst.id")
            .join(partial_edges, on=["src.id", "relation", "dst.id"])
            .group_by(["src.id", "relation", "dst.id"])
            .agg(pl.col("distance").min())
            .rename({"src.id": "mapped_id", "dst.id": "candidate_id"})
        )

        multi_candidates = partial_mapped.filter(pl.col("candidate_same_sem_tag") > 1).select("src.id", "dst.id", "relation")

        if multi_candidates.height == 0:
            df_tok_partial_multi_candidate = pl.DataFrame(
                schema={
                    "mapped_id": pl.Utf8,
                    "candidate_id": pl.Utf8,
                    "relation": pl.Utf8,
                    "distance": pl.Float64,
                },
            )
        else:
            multi_candidates_filtered = multi_candidates.with_columns(
                filtered_dst=pl.col("dst.id").map_elements(
                    self.__filter_candidates_by_hierarchy,
                    return_dtype=pl.List(pl.Utf8),
                ),
            )

            df_tok_partial_multi_candidate = (
                multi_candidates_filtered.select("src.id", "filtered_dst", "relation")
                .explode("filtered_dst")
                .join(
                    partial_edges,
                    left_on=["src.id", "filtered_dst", "relation"],
                    right_on=["src.id", "dst.id", "relation"],
                )
                .group_by("src.id", "filtered_dst", "relation")
                .agg(pl.col("distance").min())
                .rename({"src.id": "mapped_id", "filtered_dst": "candidate_id"})
            )

        df_tok_all = pl.concat(
            [
                df_tok_exact,
                df_tok_partial_one_candidate,
                df_tok_partial_multi_candidate,
                df_tok_unk,
            ],
            how="diagonal_relaxed",
        )

        if debug:
            assert len(exact_mapped) == len(set(df_tok_exact["mapped_id"]))
            assert set(df_tok_all["mapped_id"]) == self.mapped_id_set

        return df_tok_all

    def __get_distance_n_score(self, df_tok_all):
        df_dist_mapped_candidate = df_tok_all.group_by("mapped_id").agg(pl.col("distance").mean())

        # UNK concepts have a null distance here; treat them as worst-case (max_dist_candidate + 1)
        # instead of letting the mean silently skip them.
        penalty_distance = self.max_dist_candidate + 1
        mean_distance = df_dist_mapped_candidate["distance"].fill_null(penalty_distance).mean()

        if mean_distance is None or math.isnan(mean_distance):
            distance_score = 0.0
        else:
            distance_score = 1 - (mean_distance / (self.max_dist_candidate + 1))

        return df_dist_mapped_candidate, distance_score

    def __get_uniquness_n_entropy_score(self, df_tok_all_n_dist):
        total_concepts = len(self.mapped_id_list)

        redundancy_tok = (
            df_tok_all_n_dist.group_by("mapped_id")
            .agg(pl.col("candidate_id").sort())
            .group_by("candidate_id")
            .agg("mapped_id")
            .rename(
                {
                    "candidate_id": "candidate_id_list",
                    "mapped_id": "mapped_id_w_same_candidate",
                },
            )
            .with_columns(
                num_mapped_w_same_candidate=pl.col(
                    "mapped_id_w_same_candidate",
                ).list.len(),
            )
        )

        entropy = (
            redundancy_tok.with_columns(prop=pl.col("num_mapped_w_same_candidate") / total_concepts).with_columns(entropy_term=-pl.col("prop") * pl.col("prop").log())["entropy_term"].sum()
        )

        max_entropy = math.log(total_concepts) if len(redundancy_tok) > 1 else 1.0
        uniqueness_score = entropy / max_entropy

        return redundancy_tok, uniqueness_score

    def __get_conciseness_n_score(self, df_tok_all_n_dist):
        # UNK still expands to exactly one token (the UNK placeholder itself), so it's
        # included here: this score reflects sequence expansion, not mapping quality.
        df_tokens_per_concept = df_tok_all_n_dist.group_by("mapped_id").agg(pl.col("candidate_id").n_unique().alias("num_tokens"))

        if df_tokens_per_concept.height == 0:
            return df_tokens_per_concept, 0.0

        mean_tokens = df_tokens_per_concept["num_tokens"].mean()
        conciseness_score = 1.0 / mean_tokens if mean_tokens else 0.0

        return df_tokens_per_concept, conciseness_score

    def evaluate_components_and_tokenize(self, candidate_list, debug=True):
        candidate_set = set(candidate_list)

        selected_edges = self.edges_within_max.filter(
            pl.col("dst.id").is_in(candidate_set),
        )

        exact_mapped, non_exact_mapped = self.__eval_mapping_exact(
            candidate_set,
            debug=debug,
        )

        sem_cov, sem_cov_score = self.__eval_coverage_n_score(
            exact_mapped,
            non_exact_mapped,
            selected_edges,
            debug=debug,
        )

        df_tok_all_n_dist = self.get_tokenizer(
            exact_mapped,
            sem_cov,
            selected_edges,
            debug=debug,
        )

        df_dist_mapped_candidate, distance_score = self.__get_distance_n_score(
            df_tok_all_n_dist,
        )
        redundancy_tok, uniqueness_entropy_score = self.__get_uniquness_n_entropy_score(
            df_tok_all_n_dist,
        )
        df_tokens_per_concept, conciseness_score = self.__get_conciseness_n_score(
            df_tok_all_n_dist,
        )

        compression_rate = len(candidate_set) / len(self.mapped_id_list)
        unk_rate = len(df_tok_all_n_dist.filter(pl.col("candidate_id") == "UNK")) / len(self.mapped_id_list)
        exact_rate = len(exact_mapped) / len(self.mapped_id_list)

        scores = {
            "sem_cov_score": sem_cov_score,
            "distance_score": distance_score,
            "uniqueness_entropy_score": uniqueness_entropy_score,
            "conciseness_score": conciseness_score,
            "compression_rate": compression_rate,
            "UNK_rate": unk_rate,
            "exact_rate": exact_rate,
        }

        results = {
            "sem_cov": sem_cov,
            "df_dist_mapped_candidate": df_dist_mapped_candidate,
            "redundancy_tok": redundancy_tok,
            "df_tokens_per_concept": df_tokens_per_concept,
        }

        return scores, results, df_tok_all_n_dist

    def tokenize(self, candidate_list, df_hug):
        scores, best_results, df_tok_all_n_dist = self.evaluate_components_and_tokenize(candidate_list, debug=True)

        df_tok_all_n_dist = (
            df_tok_all_n_dist.join(df_hug, left_on="mapped_id", right_on="id", how="left")
            .rename({"label": "mapped_label"})
            .join(df_hug, left_on="candidate_id", right_on="id", how="left")
            .rename({"label": "candidate_label"})
            .select(
                "mapped_id",
                "mapped_label",
                "candidate_id",
                "candidate_label",
                "relation",
                "distance",
            )
        )

        return scores, best_results, df_tok_all_n_dist


"""

"""
