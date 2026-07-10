import networkx as nx
import polars as pl
from tqdm import tqdm


# ----------------------------
# Graph building
# ----------------------------
def get_is_a_graph(df_relations, col_src="src.id", col_dst="dst.id") -> nx.DiGraph:
    """
    Build an IS_A graph with edges: child -> parent
    """
    df_relations_isa = df_relations.filter(pl.col("relation") == "IS_A").select(col_src, col_dst).group_by(col_src).agg(pl.col(col_dst))

    dict_child_parent = dict(zip(df_relations_isa[col_src], df_relations_isa[col_dst]))

    G = nx.DiGraph()
    for child, parents in dict_child_parent.items():
        for parent in parents:
            G.add_edge(child, parent)

    return G


def build_relations_graph(df_relations, col_src="src.id", col_dst="dst.id", col_relation="relation") -> nx.MultiDiGraph:
    """
    Build a directed graph with all relations as edges: src -> dst, labeled with relation type.
    Uses a MultiDiGraph since a (src, dst) pair can carry more than one relation type.
    """
    G = nx.MultiDiGraph()
    for src, dst, relation in df_relations.select(col_src, col_dst, col_relation).iter_rows():
        G.add_edge(src, dst, relation=relation)
    return G


def get_subgraphs_from_nodes(G: nx.DiGraph, nodes, max_distance: int = 3) -> dict:
    """
    For each node, return the subgraph reachable by following outgoing edges up to
    max_distance hops. Nodes absent from G map to None.
    """
    return {
        node: nx.ego_graph(G, node, radius=max_distance) if node in G else None
        for node in nodes
    }


def get_all_descendants(G: nx.DiGraph, concept: str):
    """
    useful for finding all more specific concepts and concept itself in a is-a graph
    """
    nodes = nx.ancestors(G, concept)
    return {concept, *nodes}


def get_max_min_depth_snomed(G: nx.DiGraph, df_all_concepts: pl.DataFrame):
    """
    Returns (df_with_depths, unreachable_ids). unreachable_ids holds concepts
    that are either missing from G or have no path to ROOT_ID.
    """
    max_depths = {}
    min_depths = {}
    unreachable_ids = []
    ROOT_ID = "138875005"

    # Compute depth using DP (longest path to ROOT)
    def longest_depth(node):
        if node in max_depths:
            return max_depths[node]
        if node == ROOT_ID:
            max_depths[node] = 0
            return 0
        if node not in G:
            max_depths[node] = -1  # concept missing from graph
            return -1
        parent_depths = [longest_depth(p) for p in G.successors(node)]  # parents since edges are child→parent
        valid_depths = [d for d in parent_depths if d != -1]
        max_depths[node] = 1 + max(valid_depths) if valid_depths else -1  # -1: no path to root
        return max_depths[node]

    for concept in tqdm(df_all_concepts["id"], desc="Calculating depths"):
        longest_depth(concept)
        if max_depths[concept] == -1:
            unreachable_ids.append(concept)
        else:
            min_depths[concept] = nx.shortest_path_length(G, source=concept, target=ROOT_ID)

    df_result = (df_all_concepts
                 .with_columns(
                    pl.col("id").map_elements(lambda x: max_depths.get(x)).alias("max_depth"),
                )
                .with_columns(
                    pl.col("id").map_elements(lambda x: min_depths.get(x)).alias("min_depth"),
                )
                .filter(~pl.col("id").is_in(unreachable_ids))
)
    return df_result