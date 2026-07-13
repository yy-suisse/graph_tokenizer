from dataclasses import dataclass, field


@dataclass
class GraphConfig:
    graph_version: str = "2026-07-03"

    path: str = "D:/HUG_graph_data/"
    graph_path: str = f"{path}/{graph_version}/"

    relation_path: str = f"{graph_path}connectivity.parquet"
    concept_path: str = f"{graph_path}concept_snomed_hug.parquet"
    mapped_path: str = f"{graph_path}mapped_concepts.parquet"
    official_release_path: str = f"{graph_path}released_version.parquet"

@dataclass
class Context:
    relation: str                                 # the r that opened this context
    tokens:   list = field(default_factory=list)  # tokens attached directly here
    children: list = field(default_factory=list)  # (r, child Context)
    uncovered:list = field(default_factory=list)  # branches that died

class TokenizerParam:
    max_dist_candidate:int = 3
    
class ProcessedGraph:
    path: str = "D:/tokenizer_graph/new_mapped_tokenizer/"
    mapped_cpts: str = f"{path}mapped_cpts.parquet"
    combined_subgraphs : str = f"{path}combined_subgraphs.gpickle"



    