"""
Evaluation metrics for a selected token set T, complementary to semantic coverage
(F_D(T), see new_tokenizer.compute_semantic_coverage / LazyGreedyTokenSelector).

All metrics here are computed from the context trees (new_tokenizer.tokenize_all_rel)
that T actually produces for the mapped concepts M -- i.e. they evaluate the *realized*
representation, not the coverage score used to select T in the first place.
"""

import math
from collections import Counter

from src.new_tokenizer import tokenize_all_rel, signature


def _iter_contexts(ctx):
    """Yield ctx and every nested subcontext (pre-order) -- the whole tree, flattened."""
    yield ctx
    for sub in ctx.subcontexts:
        yield from _iter_contexts(sub)


def build_context_trees(M, G, T, D, id_to_label=None):
    """tokenize_all_rel(c, G, T, D, id_to_label) for every mapped concept c in M, once."""
    labels = id_to_label or {}
    return {c: tokenize_all_rel(c, G, T, D, labels) for c in M}


def conciseness(trees):
    """Mean number of token leaves per concept's context tree (lower = more concise)."""
    counts = [sum(len(ctx.tokens) for ctx in _iter_contexts(tree)) for tree in trees.values()]
    return sum(counts) / len(counts) if counts else 0.0


def distance_score(trees):
    """
    Mean hop-distance at which tokens are found, averaged per concept first (so concepts
    with many tokens don't dominate) then across concepts. Lower = tokens found closer.
    Concepts with zero tokens found (fully uncovered) are skipped -- there's no distance to average.
    """
    per_concept_means = []
    for tree in trees.values():
        distances = [d for ctx in _iter_contexts(tree) for _, d in ctx.tokens]
        if distances:
            per_concept_means.append(sum(distances) / len(distances))
    return sum(per_concept_means) / len(per_concept_means) if per_concept_means else 0.0


def uniqueness_entropy(trees):
    """
    Shannon entropy (bits, normalized to [0,1]) of the distribution of context-tree
    signatures across concepts. 1.0 = every concept gets a distinct signature (maximally
    discriminative vocabulary); near 0 = most concepts collapse onto a handful of shared
    signatures (heavy collisions, low uniqueness).
    """
    sigs = [signature(tree) for tree in trees.values()]
    n = len(sigs)
    if n <= 1:
        return 1.0
    counts = Counter(sigs)
    probs = [c / n for c in counts.values()]
    h = -sum(p * math.log2(p) for p in probs)
    return h / math.log2(n)


def unk_rate(trees):
    """
    Fraction of mapped concepts whose context tree has at least one context marked
    `uncovered` -- i.e. at least one semantic branch never reached a token within the
    depth budget D. Analogous to an out-of-vocabulary rate.
    """
    n = len(trees)
    if n == 0:
        return 0.0
    n_with_unk = sum(1 for tree in trees.values() if any(ctx.uncovered for ctx in _iter_contexts(tree)))
    return n_with_unk / n


def tree_complexity(trees):
    """Mean number of contexts (root + subcontexts) per concept's context tree."""
    counts = [sum(1 for _ in _iter_contexts(tree)) for tree in trees.values()]
    return sum(counts) / len(counts) if counts else 0.0


def exact_rate(M, T):
    """Fraction of mapped concepts that are themselves selected as a token (0-hop representation)."""
    M = list(M)
    T = set(T)
    return sum(1 for c in M if c in T) / len(M) if M else 0.0


def evaluate(M, G, T, D, id_to_label=None):
    """
    Compute all six metrics at once for a given token set T, building the context trees
    only once and reusing them across metrics. Returns a dict of metric_name -> value.
    """
    trees = build_context_trees(M, G, T, D, id_to_label)
    return {
        "conciseness": conciseness(trees),
        "distance_score": distance_score(trees),
        "uniqueness_entropy": uniqueness_entropy(trees),
        "unk_rate": unk_rate(trees),
        "tree_complexity": tree_complexity(trees),
        "exact_rate": exact_rate(M, T),
    }
