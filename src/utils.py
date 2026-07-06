from tqdm import tqdm
import networkx as nx
import polars as pl
from src import graph_fct



class ConceptPropagate:
    """
    Prepare concept indexing + relation edges for propagation.
    output table of (src.id, dst.id, relation, distance) describing, 
    for every mapped concept, how far every concept reachable from its immediate neighbors.
    """
    def __init__(
        self,
        df_relation: pl.DataFrame,
        df_concept_w_depth_init: pl.DataFrame,
        keep_only_concepts_in_dst: bool = True,
    ):
        self.df_relation: pl.DataFrame | None = None
        self.df_concept_w_depth: pl.DataFrame | None = None
        self.df_concept2idx: pl.DataFrame | None = None
        self.df_concept_ids: pl.DataFrame | None = None
        self.mapped_idx = None

        self.__prepare_concept_and_relation_for_propagation__(
            df_relation=df_relation,
            df_concept_w_depth=df_concept_w_depth_init,
            keep_only_concepts_in_dst=keep_only_concepts_in_dst,
        )

    @staticmethod
    def __conceptid2idx__(df_concept_ids: pl.DataFrame) -> pl.DataFrame:
        # Stable-ish indexing: if you want deterministic index, sort first
        # df_concept_ids = df_concept_ids.sort("id")
        return df_concept_ids.with_row_index(name="concept_idx")

    def __prepare_concept_and_relation_for_propagation__(
        self,
        df_relation: pl.DataFrame,
        df_concept_w_depth: pl.DataFrame,
        keep_only_concepts_in_dst: bool = True,
    ):
        # ---- 1) clean concept table ----
        df_concept_w_depth = df_concept_w_depth.select(["id", "is_mapped", "max_depth"])
        df_concept_ids = df_concept_w_depth.select("id").unique()

        # concept id -> idx mapping
        df_concept2idx = self.__conceptid2idx__(df_concept_ids)

        # attach idx to concept_w_depth (keep only what you need for propagation)
        df_concept_w_depth_idx = df_concept_w_depth.join(df_concept2idx, on="id", how="left").select(["concept_idx", "is_mapped", "max_depth"])

        # mapped idx (Series) for quick use later
        mapped_idx = df_concept_w_depth_idx.filter(pl.col("is_mapped") == True).get_column("concept_idx").to_list()

        # ---- 2) build relation table with indices (avoid to_series/is_in) ----
        # Keep only relations whose src is in concept set by inner join on src.id
        rel = df_relation.select(["src.id", "dst.id", "relation"]).join(df_concept_ids, left_on="src.id", right_on="id", how="inner")

        # map src.id -> src_idx
        rel = rel.join(df_concept2idx, left_on="src.id", right_on="id", how="left").rename({"concept_idx": "src_idx"})

        # map dst.id -> dst_idx (left join keeps dst outside set as null)
        rel = rel.join(df_concept2idx, left_on="dst.id", right_on="id", how="left").rename({"concept_idx": "dst_idx"})

        # optionally drop edges where dst not in concept set
        if keep_only_concepts_in_dst:
            rel = rel.filter(pl.col("dst_idx").is_not_null())

        rel = rel.select(["src_idx", "dst_idx", "relation"])

        # ---- 3) save attributes ----
        self.df_concept_ids = df_concept_ids
        self.df_concept2idx = df_concept2idx
        self.df_concept_w_depth = df_concept_w_depth_idx
        self.mapped_idx = mapped_idx
        self.df_relation = rel

    def __build_original_relations_with_dist__(self):
        #  direct neighbors with distance = 1
        return self.df_relation.with_columns(distance=1)

    def __build_self_relations__(self):
        # self loop with distance = 0 for mapped concepts
        return pl.DataFrame({"src_idx": self.mapped_idx, "dst_idx": self.mapped_idx, "relation": "IS_A", "distance": 0})

    def __build_mapped_to_all_candidate_relations__(self):
        G_is_a_index = graph_fct.get_is_a_graph(self.df_relation, col_src="src_idx", col_dst="dst_idx")
        # get all neighors list and relations
        neighbors_by_src = self.df_relation.group_by("src_idx").agg(
            [
                pl.col("dst_idx").alias("nbr_dst"),
                pl.col("relation").alias("nbr_rel"),
            ],
        )

        # Keep only mapped concepts, sorted
        df_mapped = self.df_concept_w_depth.filter(pl.col("concept_idx").is_in(self.mapped_idx)).sort("max_depth", descending=True).select(pl.col("concept_idx").alias("src_idx"))

        # Join to get neighbors of mapped concepts
        df_work = df_mapped.join(neighbors_by_src, on="src_idx", how="left")

        # propagate mapped concepts to all candidate concepts via their neighbors with corresponding relation and distance
        out_src = []
        out_dst = []
        out_rel = []
        out_dist = []

        dist_cache = {}  # neighbor -> {reachable: dist}

        for row in tqdm(df_work.iter_rows(named=True), total=df_work.height):
            src = row["src_idx"]
            nbr_dst = row["nbr_dst"] or []
            nbr_rel = row["nbr_rel"] or []

            for n, r in zip(nbr_dst, nbr_rel):
                # cache BFS per neighbor node
                dist_map = dist_cache.get(n)
                if dist_map is None:
                    dist_map = dict(nx.single_source_shortest_path_length(G_is_a_index, n))
                    dist_cache[n] = dist_map

                # append to column lists (much lighter than dict per row)
                for reachable, d in dist_map.items():
                    out_src.append(src)
                    out_dst.append(reachable)
                    out_rel.append(r)
                    out_dist.append(d + 1)
        return pl.DataFrame(
            {
                "src_idx": out_src,
                "dst_idx": out_dst,
                "relation": out_rel,
                "distance": out_dist,
            },
        )

    def __idx2id_concept__(self, df_rel_dist):
        return (
            df_rel_dist.join(self.df_concept2idx, left_on="src_idx", right_on="concept_idx", how="left")
            .rename({"id": "src.id"})
            .join(self.df_concept2idx, left_on="dst_idx", right_on="concept_idx", how="left")
            .rename({"id": "dst.id"})
            .select(["src.id", "dst.id", "relation", "distance"])
        )

    def build_all_distance_and_relations(self):
        rel_original = self.__build_original_relations_with_dist__()
        rel_self = self.__build_self_relations__().cast(rel_original.schema)
        rel_propagated = self.__build_mapped_to_all_candidate_relations__().cast(rel_original.schema)
        rel_all = (pl.concat([rel_original, rel_self, rel_propagated], how="vertical")
                   .unique()
                   .filter(pl.col("src_idx").is_in(self.mapped_idx)) # get the subgraph formed by mapped concepts reachable to ROOT.
                   )

        return self.__idx2id_concept__(rel_all)


def precompute_candidate_reachable_child_map(
    G_is_a: nx.DiGraph,
    candidate_concepts: list | set,
    graph_direction: str = "child_to_parent",
):
    """
    Precompute candidate -> all reachable child candidates.

    Parameters
    ----------
    G_is_a:
        SNOMED IS_A graph.

    candidate_concepts:
        Full candidate universe, independent of K.

    graph_direction:
        - "child_to_parent": edge = child -> parent
        - "parent_to_child": edge = parent -> child

    Returns
    -------
    dict[str, set[str]]
        candidate_reachable_child_map[parent_candidate]
        = all descendant candidate concepts.

    """
    candidate_set = set(candidate_concepts)
    candidate_reachable_child_map = {}

    for candidate in tqdm(candidate_set, desc="Precomputing reachable child candidates"):
        if candidate not in G_is_a:
            candidate_reachable_child_map[candidate] = set()
            continue

        if graph_direction == "child_to_parent":
            # In a child -> parent graph, descendants/children are NetworkX ancestors.
            reachable_children = nx.ancestors(G_is_a, candidate)

        elif graph_direction == "parent_to_child":
            # In a parent -> child graph, descendants/children are NetworkX descendants.
            reachable_children = nx.descendants(G_is_a, candidate)

        else:
            raise ValueError(
                "graph_direction must be either 'child_to_parent' or 'parent_to_child'",
            )

        candidate_reachable_child_map[candidate] = reachable_children & candidate_set

    return candidate_reachable_child_map