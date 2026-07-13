I read through the whole thing. It's a clean, self-contained method, and the pieces fit together around one central idea. Let me walk you through it in the order that makes the logic click, not just the order of the sections.

## The big reframe

Your earlier direction was to *learn* a codebook (VQ-style) that maps concepts into a small latent vocabulary. This document does something quite different and more classical: it doesn't learn anything. It **selects a subset of existing SNOMED concepts to promote into "tokens."** You pick k concepts (your under-10K target is exactly this k) out of the ~400K in the graph, and everything else gets represented *in terms of* those chosen tokens by walking the graph.

So the whole problem becomes: *which k concepts should we anoint as tokens so that the 40K mapped hospital concepts are all well-covered?* That's a combinatorial selection problem, and the rest of the paper is (1) how to score a candidate token set, and (2) how to search for a good one efficiently. The payoff of doing it this way: it's fully interpretable (tokens are real concepts) and it comes with a mathematical approximation guarantee, which a learned codebook can't offer.

## Step 1 — The two kinds of edges, and why the representation is a tree

SNOMED edges come in two flavors, and the method treats them very differently:

- **ISA edges** (the hierarchy) are *transparent*. Walking `kidney disorder ISA→ disorder` just moves you to something more general. It doesn't change *what aspect* of the concept you're talking about. So when you climb ISA links looking for a token, you stay in the same "context."
- **Non-ISA edges** (attributes like `finding site`, `laterality`, `causative agent`) each open a *new semantic facet*. Following one creates a **subcontext**.

This is why a concept is represented as a **context tree**, not a flat bag. The kidney/adrenal example in §3.2 is the whole justification: a concept with two `finding site` relationships (left kidney, right adrenal) would, in a flat bag `{(site, kidney), (site, adrenal), (laterality, left), (laterality, right)}`, lose *which laterality goes with which site*. The tree keeps `left` nested under the kidney branch and `right` under the adrenal branch. Structure is preserved by nesting.

## Step 2 — How a single concept gets tokenized (the EXPAND procedure)

To tokenize a mapped concept `c`, you do a bounded recursive descent (Listing 1). At each node `u`, current context `q`, depth `d`:

1. **`u` is already a token** → attach it to `q`, stop this branch. Done.
2. **You've hit the depth limit `d == D`, or `u` has no outgoing edges** → this branch is *uncovered* (a semantic loss).
3. **Otherwise, expand** every outgoing edge: ISA edges recurse in the *same* context; non-ISA edges recurse in a *fresh subcontext* labelled by the relationship type.

The one subtlety they flag: the order of checks 1 and 2 matters — a token sitting *exactly* at depth `D` still counts. `D` is the "how far are we willing to walk before giving up" horizon.

## Step 3 — Scoring: semantic coverage (the heart of it)

Now the key question: given a token set T, *how good is it?* This is the coverage function, and the cleanest way to understand it is as a **random walk**.

> Start at concept `u`. At each step, pick one outgoing edge uniformly at random. Walk up to `h` steps. **S_h(u, T) is the probability that this walk lands on a token.**

Read the recurrence (eq. 1) through that lens:

- **S₀(u, T) = 1 if u ∈ T, else 0.** With zero steps allowed, you're covered only if you're already standing on a token.
- **If u ∈ T: coverage = 1**, forever, at every horizon. Tokens are *absorbing* — once you hit one you've succeeded.
- **If u ∉ T and has no outgoing edges: coverage = 0.** Dead end, can't reach anything.
- **Otherwise: coverage = the average of S_{h−1}(v) over all outgoing edges.** You take one random step and inherit the (shorter-horizon) coverage of wherever you land.

Two design consequences worth noticing:

The recurrence **averages** rather than **maxes**. It's not asking "does *some* path reach a token?" — it's asking "what *fraction* of my semantic branches reach a token?" That's deliberate and it matches the main hypothesis: every facet of a concept should be represented, so a concept with one covered branch and one uncovered branch scores 0.5, not 1.

Every outgoing edge occurrence gets equal weight `1/d⁺(u)` — including all the ISA parents *and* all the attribute edges, lumped together. (They explicitly note reweighting is future work.)

### The toy example, traced

The graph is `u → t₁`, `u → v`, then `v → t₂`, `v → z`, with t₁, t₂ tokens and z an uncovered leaf, D = 2:

- `S₁(t₁) = 1` (it's a token)
- `S₁(v) = ½·S₀(t₂) + ½·S₀(z) = ½·1 + ½·0 = ½` (one branch reaches a token, one hits the dead leaf)
- `S₂(u) = ½·S₁(t₁) + ½·S₁(v) = ½·1 + ½·½ = ¾`

So from `u`, one of its two branches is fully covered and the other is half-covered → ¾. You can literally read it as: of all length-≤2 random walks from `u`, three-quarters of the probability mass ends on a token.

The global objective just averages this over all mapped concepts:
**F_D(T) = (1/|M|) · Σ_{c∈M} S_D(c, T)** — mean coverage across the 40K concepts, always between 0 and 1.

**One conceptual flag worth your attention:** the coverage score (§4) treats every outgoing edge uniformly and is *blind to the ISA/non-ISA context distinction* that §3 works so hard to preserve. The context-tree representation cares about facets; the objective being optimized does not directly reward preserving them. That may be fine — but it's the kind of gap ("does my objective actually reward the thing my representation is built around?") that's worth deciding on explicitly rather than by default.

## Step 4 — Why greedy selection is justified (submodularity)

You want the best T of size k: `T_k = argmax F_D(T)`. That's NP-hard in general, so you build T greedily — repeatedly add whichever concept gives the biggest **marginal gain** Δ_D(t | T) = F_D(T ∪ {t}) − F_D(T).

The reason greedy is *provably good* here is that F_D is **submodular** — diminishing returns. Adding a token to a small set helps at least as much as adding it to a bigger set:

$$F(A ∪ \{t\}) − F(A) \ge F(B ∪ \{t\}) − F(B), \quad A ⊆ B$$

Intuition: a token's value is "the walks it newly catches." Once other tokens already catch many walks, a new token catches fewer *additional* ones. The proof in the paper is exactly the random-walk view: fix any single walk `p`, and the function "does `p` hit T?" is a basic coverage function (monotone + submodular); F_D is an average of these, and averages of submodular functions stay submodular.

Submodularity buys you the **Nemhauser–Wolsey–Fisher guarantee**: greedy achieves at least **(1 − 1/e) ≈ 63%** of the optimal achievable coverage. That's your headline theoretical result, and it's why this framing is attractive.

## Step 5 — Making it fast (this is the real engineering)

Naively, each greedy step would recompute F_D for every candidate — O(D(|V|+|E|)) *per candidate*, per step. Two ideas kill that cost.

**Local backward propagation (§6.2).** Adding one token `t` only changes the coverage of nodes that can *reach* `t`. So instead of recomputing everything, you compute the *change* δ^t_h(u) = S_h(u, T∪{t}) − S_h(u, T). At `t` itself the change is `1 − S_h(t)` (jumping it up to full coverage); everywhere else the change propagates **backward** along incoming edges (Listing 2 walks `In(v)`, dividing by out-degree). Because most nodes are unaffected, the delta dictionaries are sparse — you only touch the nodes within D reverse-hops of `t`. When you commit the winning token (Listing 3), you apply just those sparse deltas to your stored `S[h,u]` table instead of rescanning it.

**Lazy greedy (§6.3).** Submodularity guarantees a candidate's gain can only *shrink* as T grows — so an old computed gain is a valid **upper bound** on its current gain. Keep candidates in a priority queue by their last-known gain. Pop the top, recompute only *that* one; if its refreshed gain still beats every other candidate's (stale) upper bound, it's provably the true best — select it. Otherwise reinsert it with its new value and pop the next. In practice this recomputes a tiny fraction of candidates per step, and it returns *exactly* the same tokens as plain greedy.

## Step 6 — What the experiments and extensions are

**Experiments (§7):** sweep D ∈ {1..5}, characterize the candidate set V_D (size, reachable edges, degree distributions), run greedy to k_max, and plot the coverage curve **k ↦ F_D(T_k)** plus the distribution of per-concept coverage S_D(c, T_k). Then compare against distance-based selection, random selection, exact-match, etc. That coverage curve is essentially your data-efficiency-analog for the selection method.

**Perspectives (§8), three genuinely open threads:**
- *Swaps/removal* — greedy never deletes, so an early token can become redundant once later ones cover its semantics; a marginal-loss pass could prune/replace.
- *Primitive vs. fully-defined concepts* — expanding a *primitive* concept loses meaning not captured by its relationships. They propose a reconstructibility factor α(u) = 1 (fully defined) or η < 1 (primitive) that discounts coverage gained by expanding primitives, unless the primitive is itself a token.
- *Ontology similarity* — bolt on a similarity term (facility-location style) to penalize collisions, with the caveat that submodularity/monotonicity must be re-verified if you do.

---

If it's useful, the two things I'd probe next before implementing are (1) whether the uniform edge weighting mis-serves concepts with many ISA parents (they dilute every real attribute branch), and (2) whether you want the objective to actually reward context preservation, given the gap I flagged in Step 3. Happy to dig into any single section — the submodularity proof, the delta propagation code, or the primitive-concept extension — in more depth.