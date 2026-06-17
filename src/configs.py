from dataclasses import dataclass


@dataclass
class GraphConfig:
    graph_version: str = "2026-06-17"

    path: str = "D:/HUG_graph_data/"
    graph_path: str = f"{path}/{graph_version}/"

    relation_path: str = f"{graph_path}connectivity.parquet"
    concept_path: str = f"{graph_path}concept_snomed_hug.parquet"
    mapped_path: str = f"{graph_path}mapped_concepts.parquet"
    official_release_path: str = f"{graph_path}released_version.parquet"

class ProcessedGraph:
    path: str = "D:/tokenizer_graph/"
    candidate_is_a_reachable_dict: str = f"{path}candidate_is_a_reachable_dict.pkl"
    mapped_candidate_rel_dist_prop: str = f"{path}mapped_candidate_rel_dist_prop.parquet"
    concept_w_depth: str = f"{path}concept_w_depth.parquet"

class TokenizerParam:
    max_dist_candidate:int = 3

class IterativeGraphRed:
    path: str = "D:/tokenizer_graph/graph_red_candidates/"
    