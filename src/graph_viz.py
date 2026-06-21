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


def build_concept_graph_html(
    concept_id: str,
    is_a_graph: nx.DiGraph,
    direct_neighbors: pl.DataFrame,
    used_candidates: pl.DataFrame,
    id_to_label: dict,
    max_dist: int = 3,
    max_nodes: int = 200,
    height: str = "780px",
) -> str:
    """
    Shows every concept reachable from concept_id within max_dist hops, not just the
    ones selected as candidates: the direct (distance-1) neighbors, plus every
    intermediate IS_A ancestor on the path up to a further-away candidate, so the chain
    mapped -> intermediate -> candidate is visible instead of a direct shortcut edge.

    direct_neighbors: columns dst.id, relation (distance == 1 neighbors of concept_id).
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

    for row in direct_neighbors.iter_rows(named=True):
        if len(added_nodes) >= max_nodes:
            break

        neighbor = row["dst.id"]
        relation = row["relation"]

        if neighbor == concept_id:
            continue

        style = node_style(neighbor)
        add_node(neighbor, label=_label(neighbor, id_to_label), size=46, font=NODE_FONT, **style)
        add_edge(concept_id, neighbor, label=relation, font=EDGE_FONT, **edge_style(neighbor))

        # Walk up the IS_A graph (child -> parent edges) from this direct neighbor to
        # surface every intermediate ancestor up to max_dist, regardless of whether it
        # was selected as a candidate.
        frontier = [neighbor]
        visited_chain = {neighbor}
        for _hop in range(1, max_dist):
            next_frontier = []
            for node in frontier:
                if node not in is_a_graph:
                    continue
                for parent in is_a_graph.successors(node):
                    if parent in visited_chain:
                        continue
                    visited_chain.add(parent)
                    next_frontier.append(parent)

                    if len(added_nodes) >= max_nodes:
                        continue

                    style = node_style(parent)
                    add_node(parent, label=_label(parent, id_to_label), size=36, font=NODE_FONT, **style)
                    add_edge(node, parent, label="IS_A", font=EDGE_FONT, **edge_style(parent))

            frontier = next_frontier
            if not frontier or len(added_nodes) >= max_nodes:
                break

    return net.generate_html(notebook=False)
