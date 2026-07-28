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

class CandidateLists:
    new_sem_cov_path: str = "D:/tokenizer_graph/new_mapped_tokenizer/new_sem_cov_candidates/"
    lazy_greedy : str = f"{new_sem_cov_path}lazy_greedy_no_degradation_dist.parquet"
    lazy_greedy_08 : str = f"{new_sem_cov_path}lazy_greedy_08.parquet"
    lazy_greedy_06 : str = f"{new_sem_cov_path}lazy_greedy_06.parquet"
    lazy_greedy_04 : str = f"{new_sem_cov_path}lazy_greedy_04.parquet"

    gr_path: str = "D:/tokenizer_graph/new_mapped_tokenizer/graph_red_candidates/"
    gr_sum_inv : str = f"{gr_path}sum_inv.parquet"
    gr_sum_inv_exp : str = f"{gr_path}sum_inv_exp.parquet"

    mg_path: str = "D:/tokenizer_graph/new_mapped_tokenizer/margin_gain_candidates/"
    mg_inv : str = f"{mg_path}inv.parquet"
    mg_inv_exp : str = f"{mg_path}inv_exp.parquet"

    baseline_path: str = "D:/tokenizer_graph/new_mapped_tokenizer/baselines_candidates/"
    highest_degree: str = f"{baseline_path}highest_degree.parquet"
    highest_degree_dist_1: str = f"{baseline_path}highest_degree_dist_1.parquet"
    k_random_all_samples: str = f"{baseline_path}k_random_all_samples.parquet"
    most_children: str = f"{baseline_path}most_children.parquet"

    all_candidate_to_test = {
        "lazy_new":
        {"lazy_greedy": lazy_greedy,
        "lazy_greedy_08": lazy_greedy_08,
        "lazy_greedy_06": lazy_greedy_06,
        "lazy_greedy_04": lazy_greedy_04},

        "gr": 
        {"gr_sum_inv": gr_sum_inv,
        "gr_sum_inv_exp": gr_sum_inv_exp},

        "mg":
        {"mg_inv": mg_inv,
         "mg_inv_exp": mg_inv_exp},

        "baseline":
        {"highest_degree": highest_degree,
         "highest_degree_dist_1": highest_degree_dist_1,
         "k_random_all_samples": k_random_all_samples,
         "most_children": most_children}
        }

class Result:
    result_path: str = "D:/tokenizer_graph/new_mapped_tokenizer/results/"
    metrics_result: str = f"{result_path}metrics_result.parquet"



    

            


    










    
 



    