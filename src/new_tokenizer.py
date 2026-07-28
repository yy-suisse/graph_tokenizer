import heapq
import multiprocessing as mp
import os

import numpy as np
import scipy.sparse as sp


class Context:
    """One node of a context tree.
    ISA edges stay in the current context; only non-ISA edges open a sub-context."""

    def __init__(self, labels, relation="ROOT", distance=0):
        self.labels      = labels      # shared id -> label dict
        self.relation    = relation
        self.distance    = distance
        self.destination = None        # concept this subcontext points at (None for ROOT)
        self.tokens      = []          # [(token, distance)]  <-- single source of truth
        self.uncovered   = False
        self.subcontexts = []

    def _label(self, cid):
        return self.labels.get(cid, cid)   # fall back to the id if not in the map

    def add_token(self, token, distance):
        self.tokens.append((token, distance))

    def mark_uncovered(self):
        self.uncovered = True

    def open_subcontext(self, relation, destination, distance):
        for child in self.subcontexts:
            if child.relation == relation and child.destination == destination:
                child.distance = min(child.distance, distance)   # keep most direct
                return child, False
        child = Context(self.labels, relation, distance)          # <-- pass labels through
        child.destination = destination
        self.subcontexts.append(child)
        return child, True
    
    def to_sexpr(self, show_uncovered=True):
        items  = [str(tok) for tok, _ in self.tokens]
        if self.uncovered and show_uncovered:
            items.append("∅")
        items += [f"{sub.relation}({sub.to_sexpr(show_uncovered)})"
                  for sub in self.subcontexts]
        return ", ".join(items)

    def __repr__(self):
        return f"({self.to_sexpr()})"
    # def __repr__(self):
    #     counter = itertools.count()
    #     def render(ctx, indent):
    #         qid, pad = f"q{next(counter)}", "    " * indent
    #         head = ctx.relation
    #         if ctx.destination is not None:                       # show what the edge points at
    #             head += f" → {ctx.destination} ({ctx._label(ctx.destination)})"
    #         lines = [f"{pad}[{qid}] {head}"]
    #         for tok, d in ctx.tokens:
    #             if tok == ctx.destination:            # already shown in the header
    #                 lines.append(f"{pad}    token: {tok}  (d={d})")
    #             else:
    #                 lines.append(f"{pad}    token: {tok} ({ctx._label(tok)})  (d={d})")
    #         if ctx.uncovered:
    #             lines.append(f"{pad}    uncovered")
    #         for sub in ctx.subcontexts:
    #             lines.append(render(sub, indent + 1))
    #         return "\n".join(lines)
    #     return render(self, 0)

def signature(ctx):
    """Content identity of a context, ignoring distance."""
    return (
        ctx.relation,
        ctx.uncovered,
        frozenset(tok for tok, _ in ctx.tokens),
        frozenset(signature(sub) for sub in ctx.subcontexts),
    )

def collapse_redundant(ctx):
    for sub in ctx.subcontexts:          # recurse first: resolve nested dups bottom-up
        collapse_redundant(sub)

    best = {}                            # signature -> shortest-distance subcontext
    for sub in ctx.subcontexts:
        key = signature(sub)
        if key not in best or sub.distance < best[key].distance:
            best[key] = sub
    ctx.subcontexts = list(best.values())   # dict preserves first-seen order

    seen = {}                            # same idea for tokens in THIS context
    for tok, dd in ctx.tokens:
        if tok not in seen or dd < seen[tok]:
            seen[tok] = dd
    ctx.tokens = list(seen.items())
    return ctx


def expand(u, q, d, G, T, D):
    if u in T:
        q.add_token(u, d)
    elif d == D or G.out_degree(u) == 0:
        q.mark_uncovered()
    else:
        # non-IS_A first: the direct (shallower) statement wins; parents' restatements dedup away
        edges = sorted(G.out_edges(u, data=True),
                       key=lambda e: e[2]["relation"] == "IS_A")
        for _, v, data in edges:
            r = data["relation"]
            if r == "IS_A":
                expand(v, q, d + 1, G, T, D)          # transparent, stay in q
            else:
                child, is_new = q.open_subcontext(r, v, d + 1)
                if is_new:
                    expand(v, child, d + 1, G, T, D)
    
def tokenize_all_rel(c, G, T, D, id_to_label):
    root = Context(id_to_label, "ROOT", distance=0)
    expand(c, root, 0, G, T, D)
    # return collapse_redundant(root)
    return root


# --- semantic coverage (Eq. 1) ---------------------------------------------

def build_out_edges(G):
    """Out(u) = successors of u, one entry per relationship occurrence (multi-edges kept)."""
    return {u: [v for _, v in G.out_edges(u)] for u in G.nodes()}


def compute_semantic_coverage(G, T, D, lam=1.0):
    """
    Full recomputation of S[h, u] = S_h(u, T) for h = 0..D, u in V = G.nodes() (Eq. 1, §6.2).
    Layer h only reads layer h-1, so nodes are processed in an arbitrary (non-topological) order;
    only the horizon loop 0..D needs to be in order.

    lam (0 < lam <= 1): distance/horizon penalty. Modified Eq. (1), third case:
        S_h(u, T) = lam * mean(S_{h-1}(v) for v in Out(u))
    Each hop taken multiplies by lam once, so a token found j hops away contributes lam**j
    instead of 1 (lam=1 recovers the original Eq. 1 exactly). Preserves F_D's normalized /
    monotone / submodular properties (Proposition 1), so the (1-1/e) greedy guarantee (Eq. 5)
    still holds for any lam in (0, 1].

    Returns S (shape (D+1, |V|)) and node_to_idx (node id -> column in S).
    """
    V = list(G.nodes())
    n = len(V)

    # 1.  index the nodes
    node_to_idx = {u: i for i, u in enumerate(V)}                 # column each node occupies in S
    # T_idx: columns of the tokens; index array so we can do S[h, T_idx] = 1 in one vectorized write
    T_idx = np.fromiter((node_to_idx[t] for t in T if t in node_to_idx), dtype=int)

    # 2. collect every edge occurrence as (row, col) pairs

    # Collect every edge occurrence (u, v) as a (row, col) pair, so that A[i, j] will hold
    # "how much of u's outgoing weight goes to v" once normalized below. A multi-edge u->v
    # (two different relation types) contributes two (i, j) pairs, which sp.csr_matrix sums
    # into a single entry of value 2 -- this is what makes d+(u) come out right after normalizing.
    # Token nodes are skipped here rather than zeroed afterward: Eq. 1 never averages a token's
    # successors (S_h(token) = 1 outright, applied via the S[h, T_idx] = 1 overrides below), and
    # a post-hoc `A[T_idx, :] = 0` on a lil_matrix broadcasts a dense (len(T_idx), n) array
    # internally, which OOMs once |T| and |V| are both large (e.g. k=11500, |V|=75635 -> 6.5 GiB).
    T_set = set(T)
    out_edges = build_out_edges(G)
    rows, cols = [], []
    for u in V:
        if u in T_set:
            continue
        i = node_to_idx[u]
        for v in out_edges[u]:
            rows.append(i)
            cols.append(node_to_idx[v])

    # 3. build the raw adjacency matrix and normalize by out-degree
    # A[i, j] = number of relationship occurrences from u_i to u_j (raw out-adjacency, unnormalized)
    A = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    out_degree = np.asarray(A.sum(axis=1)).flatten()               # d+(u) for every u, row sums of A
    # 1 / d+(u), but 0 (not inf) for dead ends (d+(u) = 0) -- token rows are dead ends here since
    # their edges were skipped above, so their row in A is already all-zero
    inv_out_degree = np.divide(1.0, out_degree, out=np.zeros(n), where=out_degree > 0)
    A = sp.diags(inv_out_degree) @ A          # each row i now sums to 1: A[i, :] @ S[h-1] = mean(S[h-1, v] for v in Out(u_i))
    A = lam * A                                # distance penalty: one factor of lam per hop, folded into the same matrix

    # 4. base case, h = 0
    S = np.zeros((D + 1, n))
    S[0, T_idx] = 1.0                           # base case, Eq. (1): S_0(u) = 1[u in T], 0 elsewhere (already 0)

    # 5. one matrix multiply per horizon
    for h in range(1, D + 1):
        # one horizon step for every node at once: S[h, u] = mean(S[h-1, v] for v in Out(u)),
        # 0 for dead ends (their row in A is all zero) -- this is exactly Eq. (1)'s "otherwise" case
        S[h] = A @ S[h - 1]
        S[h, T_idx] = 1.0                       # re-apply the token override (A zeroed those rows, so A @ S[h-1] gave 0 there)
    return S, node_to_idx


def semantic_coverage_score(S, node_to_idx, M):
    """F_D(T) = mean over mapped concepts c in M of S_D(c, T) (Eq. 2)."""
    idx = [node_to_idx[c] for c in M if c in node_to_idx]  # columns of the mapped concepts in S
    return S[-1, idx].mean()                               # S[-1] = S[D, :], the last (deepest) horizon row


_COVERAGE_WORKER_STATE = {}   # per-process globals, populated once by _init_coverage_worker (avoids re-pickling G/M on every task)


def _init_coverage_worker(G, M, D, id_to_label=None, lam=1.0):
    """Pool(initializer=...) target: stash the read-only graph/mapped-ids once per worker process."""
    _COVERAGE_WORKER_STATE.update(G=G, M=M, D=D, id_to_label=id_to_label, lam=lam)


def _worker_coverage_score(task):
    """
    Pool task target for scoring one (label, T) pair against the shared G/M/D/lam stashed by
    _init_coverage_worker. `label` is opaque to this function -- it's just carried through so the
    caller can match results back to their (k, iter) / (category, file_type, k) origin after
    imap_unordered reorders them. Returns (label, metrics) where metrics is eval.evaluate's dict
    (conciseness, distance_score, uniqueness_entropy, unk_rate, tree_complexity, exact_rate) plus
    semantic_coverage -- both computed from the same T against the same shared G/M, so bundling
    them here avoids building the context trees or shipping T over IPC twice.
    """
    from src.eval import evaluate as eval_all_metrics  # local import: eval.py imports from this module at load time, so importing it back at module level here would cycle

    label, T = task
    st = _COVERAGE_WORKER_STATE
    S, node_to_idx = compute_semantic_coverage(st["G"], T, st["D"], st["lam"])
    metrics = eval_all_metrics(st["M"], st["G"], T, st["D"], st["id_to_label"])
    metrics["semantic_coverage"] = semantic_coverage_score(S, node_to_idx, st["M"])
    return label, metrics


# --- lazy greedy token selection (§6) ---------------------------------------

GREEDY_RATIO = 1 - 1 / np.e   # Nemhauser-Wolsey-Fisher constant: F_D(T_greedy) >= GREEDY_RATIO * OPT (Eq. 5)


def _candidate_delta_core(t, node_to_idx, D, S, in_nodes, out_degree, T, M, lam=1.0):
    """
    Pure-function core of CANDIDATE_DELTA (§6.3, Eq. 6-7): sparse delta_h(u) =
    S_h(u, T u {t}) - S_h(u, T), propagated backward from t through in_nodes (only nodes
    that can reach t within D hops are touched). Takes plain picklable arguments (no
    `self`) so it can run identically inside a worker process or as an instance method.
    Returns (gain, delta): gain = Delta_D(t | T) (Eq. 8);
    delta[h] is a {node: value}: dict of horizon-h changes, any node not in it is unaffected (delta 0) by adding t.

    lam (0 < lam <= 1): same distance/horizon penalty as compute_semantic_coverage's lam,
    applied to Eq. (7)'s third case so the incremental computation matches the modified Eq. (1).
    """
    idx = node_to_idx
    delta = [None] * (D + 1)
    delta[0] = {t: 1.0 - S[0, idx[t]]}                           # Eq. (6): only t changes at horizon 0
    for h in range(1, D + 1):
        layer = {}
        for v, dv in delta[h - 1].items():
            for u in in_nodes.get(v, ()):
                if u in T or u == t:
                    continue                                     # existing tokens & t itself: handled by the override below, not by propagation
                layer[u] = layer.get(u, 0.0) + lam * dv / out_degree[u]   # Eq. (7), third case: push v's change onto predecessor u, weighted like Eq. (1)'s average, with the lam distance penalty
        override = 1.0 - S[h, idx[t]]
        if override > 0:
            layer[t] = override                                  # Eq. (7), first case: t is absorbing at every horizon
        delta[h] = layer
    gain = sum(delta[D].get(c, 0.0) for c in M) / len(M)          # Eq. (8)
    return gain, delta


_WORKER_STATE = {}   # per-process globals, populated once by _init_worker (avoids re-pickling on every task)


def _init_worker(node_to_idx, D, S, in_nodes, out_degree, T, M, lam):
    """Pool(initializer=...) target: stash the read-only selector state once per worker process."""
    _WORKER_STATE.update(node_to_idx=node_to_idx, D=D, S=S, in_nodes=in_nodes,
                          out_degree=out_degree, T=T, M=M, lam=lam)


def _worker_seed_gain(t):
    """
    Pool task target for the initial candidate scan. Only returns (t, gain), not the delta:
    the paper's own SELECT_TOKENS pseudocode discards the delta from the initial seeding call
    ("gain, unused_delta = CANDIDATE_DELTA(t)") since it will be recomputed anyway once (if)
    the candidate reaches the top of the heap -- so there's no reason to ship the (potentially
    large) delta dict back over IPC.
    """
    st = _WORKER_STATE
    gain, _ = _candidate_delta_core(t, st["node_to_idx"], st["D"], st["S"], st["in_nodes"], st["out_degree"], st["T"], st["M"], st["lam"])
    return t, gain


class LazyGreedyTokenSelector:
    """
    Lazy greedy selection of tokens maximizing F_D(T) (§5-6). Maintains S_h(u, T) for the
    currently committed T and updates it incrementally through CANDIDATE_DELTA/COMMIT (§6.3),
    instead of rerunning compute_semantic_coverage from scratch after every selected token.
    """

    def __init__(self, G, M, D, lam=1.0):
        self.G = G
        self.M = list(M)
        self.D = D
        self.lam = lam                                      # distance/horizon penalty, Eq. (1)/(7) third case (0 < lam <= 1; 1 = no penalty)
        self.V = list(G.nodes())
        self.node_to_idx = {u: i for i, u in enumerate(self.V)}
        self.out_degree = {u: G.out_degree(u) for u in self.V}                 # d+(u), static: doesn't depend on T
        # In(v): predecessors of v, one entry per relationship occurrence (mirrors Out(u)'s multiplicity)
        self.in_nodes = {v: [u for u, _ in G.in_edges(v)] for v in self.V}
        self.T = set()
        self.S = np.zeros((D + 1, len(self.V)))            # S[h, idx(u)] = S_h(u, T) for the current T
        self.current_score = 0.0                            # F_D(T) for the current T

    def candidate_delta(self, t):
        """CANDIDATE_DELTA (§6.3, Eq. 6-7) against the currently committed T/S. See _candidate_delta_core."""
        return _candidate_delta_core(t, self.node_to_idx, self.D, self.S, self.in_nodes, self.out_degree, self.T, self.M, self.lam)

    def commit(self, t, delta):
        """COMMIT (§6.1/§6.3): apply the accepted candidate's sparse delta onto S, and add it to T."""
        idx = self.node_to_idx
        for h in range(self.D + 1):
            for u, du in delta[h].items():
                self.S[h, idx[u]] += du
        self.T.add(t)

    def select(self, k, candidates=None, verbose=True, n_jobs=1, progress_every=5000):
        """
        SELECT_TOKENS(k) (§6.1), lazy-greedy variant: keeps one (possibly stale) gain bound
        per candidate in a max-heap. Submodularity (Proposition 1) guarantees a stale bound is
        always >= the candidate's true current gain, so recomputing only the heap's current top
        and comparing it against the next-best stale bound is enough to certify the true maximizer,
        without recomputing every remaining candidate at every iteration (unlike exact greedy).

        Before the loop starts, every candidate needs one *exact* initial gain to seed the heap
        (§6.1: "Initial gains are exact") -- this one-time pass is O(|candidates|) CANDIDATE_DELTA
        calls and is the dominant cost on a large graph (e.g. ~75k candidates), yet it produces no
        output on its own since nothing has been selected yet. progress_every prints a line every
        that many candidates scored (in both this phase and, further below, the greedy loop) so it
        doesn't look hung. n_jobs parallelizes exactly this initial scan across worker processes --
        each candidate's initial gain only depends on the fixed T/S at call time, so the scan is
        embarrassingly parallel. n_jobs=1 (default) runs it serially (no subprocess overhead, fine
        for small graphs); n_jobs=-1 uses all CPU cores. The lazy loop itself is left serial: with
        the observed ~1-7 re-evals per accepted token, IPC overhead would dominate any speedup there.

        If verbose, prints one line per selected token with:
          - its rank i/k, id, marginal gain, and cumulative score F_D(T_i);
          - re-evals: how many stale candidates had to be recomputed this round before the
            true maximizer was found (a lazy-greedy efficiency indicator -- 1 means the top
            of the heap was accepted immediately);
          - opt<=: an upper bound on the best score any T with |T| = i could reach, from the
            (1 - 1/e) guarantee (Eq. 5): since F_D(T_greedy_i) >= GREEDY_RATIO * OPT_i, we get
            OPT_i <= F_D(T_greedy_i) / GREEDY_RATIO. The guarantee holds at every prefix size,
            not just the final k, so this bound is meaningful after each token, not only at the end.

        Returns the selection history as a list of (token, marginal_gain, cumulative_score),
        in the order tokens were picked.
        """
        candidates = self.V if candidates is None else candidates
        candidates = [t for t in candidates if t not in self.T]
        n_candidates = len(candidates)
        heap = []                                                    # (-gain, token): heapq is a min-heap, negate for max-heap

        if n_jobs == 1:
            for i, t in enumerate(candidates, 1):
                gain, _ = self.candidate_delta(t)                    # initial gains are exact (§6.1 comment)
                heapq.heappush(heap, (-gain, t))
                if verbose and (i % progress_every == 0 or i == n_candidates):
                    print(f"  seeding candidates: {i}/{n_candidates}")
        else:
            n_workers = os.cpu_count() if n_jobs == -1 else n_jobs
            init_args = (self.node_to_idx, self.D, self.S, self.in_nodes, self.out_degree, self.T, self.M, self.lam)
            with mp.Pool(n_workers, initializer=_init_worker, initargs=init_args) as pool:
                if verbose:
                    print(f"  seeding candidates: 0/{n_candidates}  ({n_workers} workers)")
                for i, (t, gain) in enumerate(pool.imap_unordered(_worker_seed_gain, candidates, chunksize=64), 1):
                    heapq.heappush(heap, (-gain, t))
                    if verbose and (i % progress_every == 0 or i == n_candidates):
                        print(f"  seeding candidates: {i}/{n_candidates}  ({n_workers} workers)")

        history = []
        while len(self.T) < k and heap:
            n_evals = 0
            while True:
                _, t = heapq.heappop(heap)
                gain, delta = self.candidate_delta(t)                 # recompute exactly for the current top
                n_evals += 1
                best_other_bound = -heap[0][0] if heap else float("-inf")
                if gain >= best_other_bound:                          # true maximizer: nothing else in the queue can beat it
                    self.commit(t, delta)
                    self.current_score += gain
                    history.append((t, gain, self.current_score))
                    if verbose:
                        i = len(self.T)
                        opt_upper_bound = min(1.0, self.current_score / GREEDY_RATIO)   # F_D <= 1 always (§4), so clip the loose (1-1/e) bound
                        print(f"[{i:>4}/{k}] token={t!r:<15} gain={gain:.5f}  "
                              f"score={self.current_score:.5f}  opt<={opt_upper_bound:.5f}  "
                              f"(re-evals={n_evals}, queue={len(heap)})")
                    break
                heapq.heappush(heap, (-gain, t))                       # stale bound: re-queue with the freshly computed gain
        return history
