import pickle

import networkx as nx
import polars as pl

from src import configs, graph_fct
from src.tokenizer import GraphTokenizer

GR_FILE_BY_METHOD = {
    "gr_sum_inv": " sum_inv.parquet",
    "gr_sum_inv_exp": " sum_inv_exp.parquet",
    "gr_tempered_sum_inv": " tempered_sum_inv.parquet",
    "gr_tempered_sum_inv_exp": " tempered_sum_inv_exp.parquet",
}


def load_concepts() -> pl.DataFrame:
    return pl.read_parquet(configs.ProcessedGraph.concept_w_depth)


def load_relations() -> pl.DataFrame:
    return pl.read_parquet(configs.ProcessedGraph().mapped_candidate_rel_dist_prop).filter(
        pl.col("distance") <= configs.TokenizerParam().max_dist_candidate,
    )


def load_candidate_reachable_child_map() -> dict:
    with open(configs.ProcessedGraph().candidate_is_a_reachable_dict, "rb") as f:
        return pickle.load(f)


def load_is_a_graph() -> nx.DiGraph:
    """Full SNOMED IS_A graph (child -> parent), used to walk the path between a
    mapped concept's direct neighbor and a candidate sitting further up the hierarchy.
    """
    df_relation = pl.read_parquet(configs.GraphConfig().relation_path)
    return graph_fct.get_is_a_graph(df_relation)


def build_tokenizer(concepts: pl.DataFrame, relations: pl.DataFrame, candidate_reachable_child_map: dict) -> GraphTokenizer:
    return GraphTokenizer(
        concepts,
        relations,
        candidate_reachable_child_map,
        configs.TokenizerParam().max_dist_candidate,
    )


def get_candidate_ids(method: str, k: int) -> list[str]:
    baselines_path = configs.Baselines().path
    gr_path = configs.IterativeGraphRed().path
    mg_path = configs.IterativeMarginalGain().path

    if method == "b_random_k":
        df = pl.read_parquet(f"{baselines_path}k_random_all_samples.parquet")
        return df.filter((pl.col("k") == k) & (pl.col("iter") == 0))["candidate_id"].to_list()

    if method == "b_highest_deg_list":
        df = pl.read_parquet(f"{baselines_path}highest_degree.parquet").sort("index")
        return df["dst.id"].to_list()[:k]

    if method == "b_highest_deg_dist_1_list":
        df = pl.read_parquet(f"{baselines_path}highest_degree_dist_1.parquet").sort("index")
        return df["dst.id"].to_list()[:k]

    if method == "b_most_children_list":
        df = pl.read_parquet(f"{baselines_path}most_children.parquet").sort("index")
        return df["candidate"].to_list()[:k]

    if method == "mg_inv":
        df = pl.read_parquet(f"{mg_path}_inv.parquet").sort("rank")
        return df["candidate_id"].to_list()[:k]

    if method == "mg_inv_exp":
        df = pl.read_parquet(f"{mg_path}_inv_exp.parquet").sort("rank")
        return df["candidate_id"].to_list()[:k]

    if method in GR_FILE_BY_METHOD:
        df = pl.read_parquet(f"{gr_path}{GR_FILE_BY_METHOD[method]}")
        return df["candidate"].to_list()[:k]

    raise ValueError(f"Unknown method: {method}")


def tokenize_for(tokenizer: GraphTokenizer, method: str, k: int):
    candidate_ids = get_candidate_ids(method, k)
    return tokenizer.evaluate_components_and_tokenize(candidate_ids, debug=False)


def get_all_concept_scores(results: dict) -> pl.DataFrame:
    """Build a per-concept score table from the `results` dict returned by
    GraphTokenizer.evaluate_components_and_tokenize.

    Columns: mapped_id, frac_sem_cov, mean_distance, num_tokens, redundancy_group_size.
    """
    sem_cov = results["sem_cov"].select("mapped_id", "frac_sem_cov")

    dist = results["df_dist_mapped_candidate"].rename({"distance": "mean_distance"})

    tokens = results["df_tokens_per_concept"]

    # Explode the redundancy groups to get group_size per mapped concept.
    group_sizes = (
        results["redundancy_tok"]
        .select("mapped_id_w_same_candidate", "num_mapped_w_same_candidate")
        .explode("mapped_id_w_same_candidate")
        .rename({"mapped_id_w_same_candidate": "mapped_id", "num_mapped_w_same_candidate": "redundancy_group_size"})
    )

    return (
        sem_cov
        .join(dist, on="mapped_id", how="left")
        .join(tokens, on="mapped_id", how="left")
        .join(group_sizes, on="mapped_id", how="left")
    )


def get_concept_scores(results: dict, mapped_id: str) -> dict:
    """Per-concept breakdown of the components behind the method-level scores, for one
    mapped concept, pulled out of the `results` dict returned alongside `scores` by
    GraphTokenizer.evaluate_components_and_tokenize.
    """
    sem_cov_row = results["sem_cov"].filter(pl.col("mapped_id") == mapped_id)
    frac_sem_cov = sem_cov_row["frac_sem_cov"][0] if sem_cov_row.height else None

    dist_row = results["df_dist_mapped_candidate"].filter(pl.col("mapped_id") == mapped_id)
    mean_distance = dist_row["distance"][0] if dist_row.height else None

    tokens_row = results["df_tokens_per_concept"].filter(pl.col("mapped_id") == mapped_id)
    num_tokens = tokens_row["num_tokens"][0] if tokens_row.height else None

    redundancy_row = results["redundancy_tok"].filter(
        pl.col("mapped_id_w_same_candidate").list.contains(mapped_id),
    )
    group_size = redundancy_row["num_mapped_w_same_candidate"][0] if redundancy_row.height else None

    return {
        "frac_sem_cov": frac_sem_cov,
        "mean_distance": mean_distance,
        "num_tokens": num_tokens,
        "redundancy_group_size": group_size,
    }
