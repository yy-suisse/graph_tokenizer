import pickle
from collections import Counter

import networkx as nx
import polars as pl

from src import configs_new_mapped as configs
from src.eval import _iter_contexts, build_context_trees
from src.new_tokenizer import compute_semantic_coverage, signature

CANDIDATE_COL_CANDIDATES = ["token", "candidate_id", "candidate", "dst.id"]
RANK_COL_CANDIDATES = ["index", "rank"]


def load_graph() -> nx.MultiDiGraph:
    with open(configs.ProcessedGraph().combined_subgraphs, "rb") as f:
        return pickle.load(f)


def load_mapped_concepts() -> pl.DataFrame:
    return pl.read_parquet(configs.ProcessedGraph().mapped_cpts)


def load_id_to_label() -> dict:
    df = pl.read_parquet(configs.GraphConfig().concept_path).select("id", "label")
    return dict(zip(df["id"].to_list(), df["label"].to_list()))


def _pick_col(df: pl.DataFrame, options: list[str]) -> str | None:
    for c in options:
        if c in df.columns:
            return c
    return None


def get_candidate_ids(category: str, file_type: str, k: int, iter: int = 0) -> list[str]:
    """Top-k candidate ids for one (category, file_type) candidate list, in the
    file's own rank order -- mirrors the head(k) selection used in 7.evaluation_new.ipynb.
    """
    path = configs.CandidateLists().all_candidate_to_test[category][file_type]
    df = pl.read_parquet(path)

    if file_type == "k_random_all_samples":
        return df.filter((pl.col("k") == k) & (pl.col("iter") == iter))["candidate_id"].to_list()

    candidate_col = _pick_col(df, CANDIDATE_COL_CANDIDATES)
    rank_col = _pick_col(df, RANK_COL_CANDIDATES)
    if rank_col is not None:
        df = df.sort(rank_col)
    return df.head(k)[candidate_col].to_list()


def get_all_concept_scores(trees: dict, S, node_to_idx: dict, mapped_ids: list[str]) -> pl.DataFrame:
    """Per-concept score table, analogous to drilldown.get_all_concept_scores but built
    directly from the context trees / semantic-coverage matrix of the new tokenizer.

    Columns: mapped_id, frac_sem_cov, mean_distance, num_tokens, redundancy_group_size,
    signature_id (concepts with the same signature_id share the exact same representation
    -- use it to look up the other members of a concept's redundancy group).
    """
    sigs = {c: signature(trees[c]) for c in mapped_ids if c in trees}
    sig_counts = Counter(sigs.values())
    sig_to_gid: dict = {}

    rows = []
    for c in mapped_ids:
        tree = trees.get(c)
        if tree is None:
            continue
        contexts = list(_iter_contexts(tree))
        distances = [d for ctx in contexts for _, d in ctx.tokens]
        sig = sigs[c]
        rows.append({
            "mapped_id": c,
            "frac_sem_cov": float(S[-1, node_to_idx[c]]) if c in node_to_idx else None,
            "mean_distance": sum(distances) / len(distances) if distances else None,
            "num_tokens": sum(len(ctx.tokens) for ctx in contexts),
            "redundancy_group_size": sig_counts[sig],
            "signature_id": sig_to_gid.setdefault(sig, len(sig_to_gid)),
        })
    return pl.DataFrame(rows)


def flatten_concept_tree(root) -> tuple[list[dict], list[dict]]:
    """Flatten a concept's context tree into display rows.

    Returns (token_rows, uncovered_rows):
      - token_rows: {candidate_id, relation, distance} for every token found, `relation`
        being the full chain of relations opened to reach that context (" > "-joined;
        "ROOT" for tokens found without leaving the IS_A chain).
      - uncovered_rows: {relation} for every branch that died without finding a token.
    """
    token_rows, uncovered_rows = [], []

    def walk(ctx, path):
        rel_path = " > ".join(path) if path else "ROOT"
        for tok, d in ctx.tokens:
            token_rows.append({"candidate_id": tok, "relation": rel_path, "distance": d})
        if ctx.uncovered:
            uncovered_rows.append({"relation": rel_path})
        for sub in ctx.subcontexts:
            walk(sub, path + [sub.relation])

    walk(root, [])
    return token_rows, uncovered_rows


def render_context_tree(root, id_to_label: dict) -> str:
    """ASCII tree view of a concept's context tree, for display alongside
    flatten_concept_tree's flat table -- each relation the tree opens becomes a nested
    branch, with token leaves and uncovered (`∅`) leaves shown at whatever depth
    they were actually found.
    """
    lines: list[str] = []

    def label(cid: str) -> str:
        name = id_to_label.get(cid)
        return f"{name} ({cid})" if name else cid

    def render(node, prefix: str, is_last: bool, is_root: bool) -> None:
        head = node.relation
        if node.destination is not None:
            head += f" -> {label(node.destination)}"
        connector = "" if is_root else ("└── " if is_last else "├── ")
        lines.append(f"{prefix}{connector}{head}")

        child_prefix = prefix if is_root else prefix + ("    " if is_last else "│   ")

        leaves = [("token", tok, d) for tok, d in node.tokens]
        if node.uncovered:
            leaves.append(("uncovered", None, None))
        entries = len(leaves) + len(node.subcontexts)

        i = 0
        for kind, tok, d in leaves:
            i += 1
            conn = "└── " if i == entries else "├── "
            if kind == "token":
                lines.append(f"{child_prefix}{conn}• {label(tok)}  d={d}")
            else:
                lines.append(f"{child_prefix}{conn}∅ uncovered")

        for sub in node.subcontexts:
            i += 1
            render(sub, child_prefix, i == entries, is_root=False)

    render(root, "", True, is_root=True)
    return "\n".join(lines)


def is_exact_match(tree, concept_id: str) -> bool:
    return (
        len(tree.tokens) == 1
        and tree.tokens[0][0] == concept_id
        and tree.tokens[0][1] == 0.0
        and not tree.subcontexts
        and not tree.uncovered
    )


def is_fully_unk(tree) -> bool:
    return sum(len(ctx.tokens) for ctx in _iter_contexts(tree)) == 0


def tokenize_for(G, mapped_ids: list[str], id_to_label: dict, D: int, category: str, file_type: str, k: int):
    """End-to-end: pick the candidate set for (category, file_type, k), build every mapped
    concept's context tree against it, and compute the per-concept score table alongside.
    """
    T = get_candidate_ids(category, file_type, k)
    trees = build_context_trees(mapped_ids, G, T, D, id_to_label)
    S, node_to_idx = compute_semantic_coverage(G, T, D)
    concept_scores = get_all_concept_scores(trees, S, node_to_idx, mapped_ids)
    return T, trees, concept_scores
