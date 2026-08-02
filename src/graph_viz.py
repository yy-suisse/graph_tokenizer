import networkx as nx
import polars as pl
from pyvis.network import Network

COLOR_CENTER = "#1f3a93"
COLOR_CANDIDATE = "#2ecc71"
COLOR_NOT_CANDIDATE = "#b0b0b0"
COLOR_UNK = "#e74c3c"

NODE_FONT = {"size": 26, "face": "arial", "strokeWidth": 3, "strokeColor": "#ffffff"}
EDGE_FONT = {"size": 20, "face": "arial", "strokeWidth": 3, "strokeColor": "#ffffff", "align": "middle"}


def _label(concept_id: str, id_to_label: dict) -> str:
    label = id_to_label.get(concept_id)
    return f"{label}\n({concept_id})" if label else concept_id


def build_new_concept_graph_html(
    concept_id: str,
    G: nx.MultiDiGraph,
    T,
    D: int,
    id_to_label: dict,
    max_nodes: int = 200,
    height: str = "780px",
) -> str:
    """
    New-tokenizer counterpart of build_concept_graph_html: walks combined_subgraphs
    directly from concept_id (the same graph new_tokenizer.expand() draws candidates
    from), following every out-edge up to D hops -- IS_A included, since the new
    tokenizer has no separate is_a_graph to reconstruct chains from.

    Colors: blue = concept_id itself; green = a node in T actually reached (a token);
    red = a branch that died (depth budget D exhausted, or a dead end) without hitting
    a token; gray = an intermediate node still being explored (not itself a candidate).

    This traversal is a simplified visualization aid, not a re-run of tokenize_all_rel's
    exact context-tree semantics (no relation-based subcontext dedup) -- the token/relation
    table shown alongside it is the ground truth for what the tokenizer actually assigned.
    """
    net = Network(height=height, width="100%", directed=True, notebook=False, cdn_resources="remote")
    net.barnes_hut(spring_length=320, spring_strength=0.02)

    T_set = set(T)

    added_nodes: set[str] = set()
    added_edges: set[tuple[str, str, str]] = set()
    visited: set[str] = set()

    def add_node(node_id: str, **kwargs) -> None:
        if node_id in added_nodes:
            return
        added_nodes.add(node_id)
        net.add_node(node_id, **kwargs)

    def add_edge(src: str, dst: str, relation: str, **kwargs) -> None:
        key = (src, dst, relation)
        if key in added_edges:
            return
        added_edges.add(key)
        net.add_edge(src, dst, **kwargs)

    add_node(
        concept_id,
        label=_label(concept_id, id_to_label),
        color=COLOR_CENTER,
        size=55,
        font=NODE_FONT,
        title="Tokenized concept",
    )

    def walk(u: str, d: int) -> None:
        if u in visited or len(added_nodes) >= max_nodes:
            return
        visited.add(u)
        if u in T_set or d == D or G.out_degree(u) == 0:
            # Mirrors expand()'s own top-of-function check: once u is itself a token
            # (or a dead end), expand() stops there and never looks at u's neighbors --
            # most importantly for u == concept_id, an exact match (concept_id in T)
            # means the tokenizer never explored anything beyond concept_id itself.
            return

        for _, v, data in sorted(G.out_edges(u, data=True), key=lambda e: e[2]["relation"] == "IS_A"):
            if len(added_nodes) >= max_nodes:
                break
            if v == concept_id:
                continue
            relation = data["relation"]

            is_token = v in T_set
            is_dead_end = (not is_token) and (d + 1 == D or G.out_degree(v) == 0)

            if is_token:
                color, title, size = COLOR_CANDIDATE, "Tokenizing candidate", 46
            elif is_dead_end:
                color, title, size = COLOR_UNK, "Depth limit / dead end without a candidate", 36
            else:
                color, title, size = COLOR_NOT_CANDIDATE, "Not used as a tokenizing candidate", 40

            add_node(v, label=_label(v, id_to_label), size=size, font=NODE_FONT, color=color, title=title)
            add_edge(
                u, v, relation,
                label=relation, font=EDGE_FONT, color=color,
                width=3 if is_token else 2, dashes=not is_token,
            )

            if not is_token and not is_dead_end:
                walk(v, d + 1)

    walk(concept_id, 0)

    return net.generate_html(notebook=False)


def build_concept_graph_html(
    concept_id: str,
    is_a_graph: nx.DiGraph,
    neighbors: pl.DataFrame,
    used_candidates: pl.DataFrame,
    id_to_label: dict,
    max_nodes: int = 200,
    height: str = "780px",
) -> str:
    """
    Shows every concept reachable from concept_id within max_dist_candidate hops, not
    just the ones selected as candidates — and draws the real hop-by-hop connectivity
    (mapped concept -> distance-1 neighbor -> IS_A ancestor -> ... -> destination)
    instead of a direct shortcut edge, so every intermediate concept on the path is
    visible.

    neighbors: columns dst.id, relation, distance — every neighbor of concept_id up to
        max_dist_candidate hops (the same table the tokenizer draws candidates from).
        distance == 1 rows are direct neighbors (any relation type); distance > 1 rows
        are reached from a distance-1 neighbor purely via IS_A hops.
    used_candidates: columns candidate_id, relation, distance (candidates the tokenizer
        actually assigned to concept_id, excluding "UNK").
    """
    net = Network(height=height, width="100%", directed=True, notebook=False, cdn_resources="remote")
    net.barnes_hut(spring_length=320, spring_strength=0.02)

    used_candidate_ids = set(used_candidates["candidate_id"].to_list()) if used_candidates.height else set()
    unk_present = used_candidates.height and (used_candidates["candidate_id"] == "UNK").any()

    added_nodes: set[str] = set()
    added_edges: set[tuple[str, str]] = set()

    def add_node(node_id: str, **kwargs) -> None:
        if node_id in added_nodes:
            return
        added_nodes.add(node_id)
        net.add_node(node_id, **kwargs)

    def add_edge(src: str, dst: str, **kwargs) -> None:
        if (src, dst) in added_edges:
            return
        added_edges.add((src, dst))
        net.add_edge(src, dst, **kwargs)

    add_node(
        concept_id,
        label=_label(concept_id, id_to_label),
        color=COLOR_CENTER,
        size=55,
        font=NODE_FONT,
        title="Tokenized concept",
    )

    if unk_present:
        add_node(
            "UNK",
            label="UNK",
            color=COLOR_UNK,
            size=42,
            font=NODE_FONT,
            title="No candidate covers this concept",
        )
        add_edge(concept_id, "UNK", label="UNK", color=COLOR_UNK, width=3, font=EDGE_FONT)

    def node_style(node_id: str) -> dict:
        if node_id in used_candidate_ids:
            return {"color": COLOR_CANDIDATE, "title": "Tokenizing candidate"}
        return {"color": COLOR_NOT_CANDIDATE, "title": "Not used as a tokenizing candidate"}

    def edge_style(target_id: str) -> dict:
        if target_id in used_candidate_ids:
            return {"color": COLOR_CANDIDATE, "dashes": False, "width": 3}
        return {"color": COLOR_NOT_CANDIDATE, "dashes": True, "width": 2}

    direct_rows = neighbors.filter(pl.col("distance") == 1)
    further_rows = neighbors.filter(pl.col("distance") > 1)

    direct_neighbor_by_relation: dict[str, list[str]] = {}
    for row in direct_rows.iter_rows(named=True):
        neighbor = row["dst.id"]
        relation = row["relation"]

        if neighbor == concept_id or len(added_nodes) >= max_nodes:
            continue

        style = node_style(neighbor)
        add_node(neighbor, label=_label(neighbor, id_to_label), size=46, font=NODE_FONT, **style)
        add_edge(concept_id, neighbor, label=relation, font=EDGE_FONT, **edge_style(neighbor))
        direct_neighbor_by_relation.setdefault(relation, []).append(neighbor)

    # For distance > 1, reconstruct the actual IS_A chain from the distance-1 neighbor
    # (reached via the same relation) to the destination, so intermediate ancestors show
    # up as real nodes instead of being hidden behind a shortcut edge.
    max_extra_hops = int(further_rows["distance"].max()) - 1 if further_rows.height else 0
    is_a_path_cache: dict[str, dict[str, list[str]]] = {}

    def is_a_paths_from(neighbor: str) -> dict[str, list[str]]:
        if neighbor not in is_a_path_cache:
            if neighbor in is_a_graph:
                is_a_path_cache[neighbor] = nx.single_source_shortest_path(is_a_graph, neighbor, cutoff=max_extra_hops)
            else:
                is_a_path_cache[neighbor] = {}
        return is_a_path_cache[neighbor]

    for row in further_rows.iter_rows(named=True):
        if len(added_nodes) >= max_nodes:
            break

        dst = row["dst.id"]
        relation = row["relation"]
        distance = row["distance"]

        if dst == concept_id or dst in added_nodes:
            continue

        # Prefer a chain starting from a same-relation distance-1 neighbor, but fall back
        # to any distance-1 neighbor with a real IS_A path to dst — the exact hop count
        # can differ slightly from the recorded distance because is_a_graph (built from
        # the full, unfiltered connectivity table) can contain shortcut IS_A edges that
        # weren't present in the smaller graph used to precompute distances.
        chain = None
        for neighbor in direct_neighbor_by_relation.get(relation, []):
            chain = is_a_paths_from(neighbor).get(dst)
            if chain:
                break

        if chain is None:
            for neighbor_list in direct_neighbor_by_relation.values():
                for neighbor in neighbor_list:
                    chain = is_a_paths_from(neighbor).get(dst)
                    if chain:
                        break
                if chain:
                    break

        if chain is None:
            # No reconstructable IS_A path at all (fully disconnected in is_a_graph) —
            # still show the concept rather than silently dropping it.
            style = node_style(dst)
            add_node(dst, label=_label(dst, id_to_label), size=36, font=NODE_FONT, **style)
            add_edge(concept_id, dst, label=f"{relation} (d={int(distance)})", font=EDGE_FONT, **edge_style(dst))
            continue

        prev = chain[0]
        for node in chain[1:]:
            style = node_style(node)
            add_node(node, label=_label(node, id_to_label), size=36, font=NODE_FONT, **style)
            add_edge(prev, node, label="IS_A", font=EDGE_FONT, **edge_style(node))
            prev = node

    return net.generate_html(notebook=False)
