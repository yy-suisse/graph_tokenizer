import polars as pl
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

from src import configs_new_mapped as configs
from src import drilldown_new, graph_viz

st.set_page_config(page_title="Graph Tokenizer — New Tokenizer Comparison", layout="wide")

D = configs.TokenizerParam().max_dist_candidate


@st.cache_data
def load_scores() -> pl.DataFrame:
    df = pl.read_parquet(configs.Result().metrics_result)
    return df.with_columns((pl.col("category") + "/" + pl.col("file_type")).alias("method"))


@st.cache_resource(show_spinner="Loading combined subgraph + mapped concepts...")
def load_graph_data():
    G = drilldown_new.load_graph()
    df_mapped = drilldown_new.load_mapped_concepts()
    mapped_ids = df_mapped["id"].to_list()
    id_to_label = drilldown_new.load_id_to_label()
    return G, mapped_ids, id_to_label, df_mapped


@st.cache_data
def load_mapped_concept_options():
    _G, _mapped_ids, _id_to_label, df_mapped = load_graph_data()
    return df_mapped.select("id", "label").to_pandas()


@st.cache_data(show_spinner="Tokenizing with this method/k...")
def cached_tokenize_for(method: str, k: int):
    category, file_type = method.split("/", 1)
    G, mapped_ids, id_to_label, _df_mapped = load_graph_data()
    T, trees, concept_scores = drilldown_new.tokenize_for(G, mapped_ids, id_to_label, D, category, file_type, k)
    return T, trees, concept_scores.to_pandas()


SCORE_COLS = [
    "semantic_coverage",
    "distance_score",
    "uniqueness_entropy",
    "conciseness",
    "tree_complexity",
    "unk_rate",
    "exact_rate",
]

LOWER_IS_BETTER = {"distance_score", "conciseness", "tree_complexity", "unk_rate"}

df = load_scores()
k_values = sorted(df["k"].unique().to_list())
methods = sorted(df["method"].unique().to_list())

st.title("New Tokenizer — Candidate Selection Method Comparison")
st.caption(
    "Browse precomputed evaluation metrics (`7.evaluation_new.ipynb`) for every candidate-selection "
    "method produced by the new context-tree tokenizer (`6.new_tokenizer_test_ground.ipynb`), on the "
    f"**{len(load_graph_data()[1]):,} mapped concepts**. Charts use **k** directly on the x-axis — "
    "every method's candidate list is evaluated at the same nominal k values."
)

with st.sidebar:
    st.header("Filters")
    k_range = st.select_slider(
        "k range",
        options=k_values,
        value=(min(k_values), max(k_values)),
    )

# --- Method toggle panel ---
with st.container(border=True):
    toggle_cols = st.columns([1, 1] + [2] * len(methods))
    with toggle_cols[0]:
        if st.button("All", use_container_width=True):
            for m in methods:
                st.session_state[f"method_toggle_{m}"] = True
    with toggle_cols[1]:
        if st.button("None", use_container_width=True):
            for m in methods:
                st.session_state[f"method_toggle_{m}"] = False
    selected_methods = []
    for col, method in zip(toggle_cols[2:], methods):
        checked = col.checkbox(
            method,
            value=st.session_state.get(f"method_toggle_{method}", True),
            key=f"method_toggle_{method}",
        )
        if checked:
            selected_methods.append(method)

if not selected_methods:
    st.warning("No methods selected — select at least one above.")
    st.stop()

filtered = df.filter(
    pl.col("method").is_in(selected_methods)
    & (pl.col("k") >= k_range[0])
    & (pl.col("k") <= k_range[1])
)

tab_compare, tab_table, tab_dist, tab_drilldown = st.tabs(
    ["Compare scores", "Raw table", "Score distributions", "Concept drill-down"],
)

with tab_compare:
    st.subheader("All scores vs k")
    compare_cols = st.columns(2)
    for i, score in enumerate(SCORE_COLS):
        with compare_cols[i % 2]:
            direction = "lower is better" if score in LOWER_IS_BETTER else "higher is better"
            fig_small = px.line(
                filtered.sort("k").to_pandas(),
                x="k",
                y=score,
                color="method",
                markers=True,
                title=f"{score} ({direction})",
            )
            fig_small.update_layout(
                height=300,
                showlegend=(i == 0),
                legend_itemclick=False,
                legend_itemdoubleclick=False,
                margin={"t": 40, "b": 10},
            )
            st.plotly_chart(fig_small, use_container_width=True, key=f"compare_small_{score}")

with tab_table:
    st.subheader("Filtered raw scores")
    st.dataframe(filtered.sort(["method", "k"]), use_container_width=True, hide_index=True)
    st.download_button(
        "Download filtered scores as CSV",
        filtered.write_csv(),
        file_name="filtered_scores_new_tokenizer.csv",
        mime="text/csv",
    )

with tab_dist:
    st.subheader("Per-concept score distributions")
    st.caption(
        "Choose a method and k to see how individual scores are spread across all mapped concepts "
        "— revealing whether good aggregate scores hide a long tail of poorly-covered concepts.",
    )

    col_dm, col_dk = st.columns(2)
    with col_dm:
        dist_method = st.selectbox("Method", methods, key="dist_method")
    with col_dk:
        dist_method_rows = df.filter(pl.col("method") == dist_method).sort("k")
        dist_k_options = dist_method_rows["k"].to_list()
        dist_k = st.select_slider("k", options=dist_k_options, key="dist_k")

    if dist_method.endswith("k_random_all_samples"):
        st.info(
            "`k_random_all_samples` uses a single representative draw (iter 0) here, not the "
            "multi-draw average shown in the leaderboard — scores will differ slightly.",
        )

    method_row = dist_method_rows.filter(pl.col("k") == dist_k)
    _T, _trees, concept_df = cached_tokenize_for(dist_method, dist_k)
    st.markdown("**Method-level aggregate scores at this k**")
    agg_cols = st.columns(len(SCORE_COLS))
    for col, name in zip(agg_cols, SCORE_COLS):
        val = method_row[name][0] if method_row.height else None
        col.metric(name.replace("_", " "), f"{val:.3f}" if val is not None else "—")

    st.divider()

    # --- distributions ---
    DIST_SPECS = [
        ("frac_sem_cov",          "Semantic coverage per concept",   "S_D(c, T) — 1.0 = full coverage, 0.0 = fully uncovered",    True),
        ("mean_distance",         "Mean token distance per concept",  "Average hop distance to found tokens; fully uncovered concepts excluded",   False),
        ("num_tokens",            "Number of tokens per concept",     "How many token leaves a concept's context tree resolves to",                     False),
        ("redundancy_group_size", "Concepts sharing the same context-tree signature — how many mapped concepts have an identical representation", "Number of concepts in the same signature group (1 = fully unique representation)", False),
    ]

    for col_a, col_b in zip(DIST_SPECS[::2], DIST_SPECS[1::2]):
        left, right = st.columns(2)
        for pane, (col_name, title, x_label, show_unk_note) in zip([left, right], [col_a, col_b]):
            with pane:
                plot_df = concept_df[["mapped_id", col_name]].dropna()

                # Binary summary bar for the token-set uniqueness metric
                if col_name == "redundancy_group_size":
                    n_unique = int((plot_df[col_name] == 1).sum())
                    n_shared = int((plot_df[col_name] > 1).sum())
                    total = n_unique + n_shared
                    import plotly.graph_objects as go
                    fig_bar = go.Figure(data=[
                        go.Bar(name="Unique signature", x=["Token set uniqueness"], y=[n_unique],
                               marker_color="#2ecc71",
                               text=[f"{n_unique} ({100*n_unique/total:.1f}%)"], textposition="outside"),
                        go.Bar(name="Shared signature (≥2 concepts)", x=["Token set uniqueness"], y=[n_shared],
                               marker_color="#e74c3c",
                               text=[f"{n_shared} ({100*n_shared/total:.1f}%)"], textposition="outside"),
                    ])
                    fig_bar.update_layout(
                        barmode="stack",
                        title="Unique vs. shared context-tree signatures",
                        height=260,
                        showlegend=True,
                        margin={"t": 40, "b": 10},
                        yaxis_title="Number of concepts",
                    )
                    st.plotly_chart(fig_bar, use_container_width=True, key="dist_uniqueness_bar")

                if col_name == "redundancy_group_size":
                    max_val = int(plot_df[col_name].max())
                    x_cap = 50
                    tail_n = int((plot_df[col_name] > x_cap).sum())
                    fig = px.histogram(
                        plot_df,
                        x=col_name,
                        title=title,
                        labels={col_name: x_label},
                        color_discrete_sequence=["#4a90d9"],
                    )
                    fig.update_traces(xbins=dict(start=0.5, end=max_val + 0.5, size=1))
                    fig.update_layout(height=340, bargap=0.05, showlegend=False,
                                      margin={"t": 40, "b": 10}, xaxis_range=[0, x_cap])
                else:
                    fig = px.histogram(
                        plot_df,
                        x=col_name,
                        nbins=40,
                        title=title,
                        labels={col_name: x_label},
                        color_discrete_sequence=["#4a90d9"],
                    )
                    fig.update_layout(height=340, bargap=0.05, showlegend=False,
                                      margin={"t": 40, "b": 10})
                st.plotly_chart(fig, use_container_width=True, key=f"dist_{col_name}")
                if col_name == "redundancy_group_size" and tail_n > 0:
                    st.caption(f"{tail_n} concept(s) with group size > {x_cap} not shown (max = {max_val}).")
                unk_n = (concept_df["frac_sem_cov"] == 0.0).sum()
                if show_unk_note and unk_n > 0:
                    st.caption(f"{unk_n} fully-uncovered concept(s) (frac_sem_cov = 0) included at the left edge.")
                null_n = concept_df[col_name].isna().sum()
                if null_n > 0:
                    st.caption(f"{null_n} concept(s) with no value for this metric are excluded.")

    st.divider()
    st.markdown("**Joint distribution: semantic coverage vs mean distance**")
    joint_df = concept_df[["frac_sem_cov", "mean_distance", "num_tokens"]].dropna()
    fig_joint = px.scatter(
        joint_df,
        x="mean_distance",
        y="frac_sem_cov",
        color="num_tokens",
        color_continuous_scale="Viridis",
        labels={
            "mean_distance": "Mean token distance",
            "frac_sem_cov": "Semantic coverage",
            "num_tokens": "# tokens",
        },
        opacity=0.4,
    )
    fig_joint.update_traces(marker_size=4)
    fig_joint.update_layout(height=420)
    st.plotly_chart(fig_joint, use_container_width=True)

with tab_drilldown:
    st.subheader("How is a single concept tokenized?")
    st.caption(
        "This tab re-runs the tokenizer for one method/k on demand (not precomputed), "
        "so picking a new method/k takes several seconds the first time.",
    )

    col_method, col_k = st.columns(2)
    with col_method:
        dd_method = st.selectbox("Method", methods, key="dd_method")
    with col_k:
        dd_method_rows = df.filter(pl.col("method") == dd_method).sort("k")
        dd_k_options = dd_method_rows["k"].to_list()
        dd_k = st.select_slider("k", options=dd_k_options, key="dd_k")

    if dd_method.endswith("k_random_all_samples"):
        st.info(
            "`k_random_all_samples` uses a single representative draw (iter 0) here, not the "
            "multi-draw average shown in the leaderboard — scores will differ slightly.",
        )

    mapped_options = load_mapped_concept_options()

    search = st.text_input("Search mapped concept by id or label", "")
    if search:
        mask = mapped_options["id"].str.contains(search, case=False, regex=False) | mapped_options["label"].str.contains(
            search, case=False, regex=False,
        )
        matches = mapped_options[mask].head(200)
    else:
        matches = mapped_options.head(200)

    if matches.empty:
        st.warning("No mapped concept matches that search.")
    else:
        option_labels = [f"{row.label} ({row.id})" for row in matches.itertuples()]
        picked = st.selectbox("Mapped concept", option_labels, key="dd_concept")
        picked_id = matches.iloc[option_labels.index(picked)]["id"]

        T, trees, concept_df = cached_tokenize_for(dd_method, dd_k)
        G, _mapped_ids, id_to_label, _df_mapped = load_graph_data()

        tree = trees.get(picked_id)

        st.markdown(f"### {id_to_label.get(picked_id, picked_id)} (`{picked_id}`)")

        if tree is None:
            st.warning("This concept was not found in the tokenization output.")
        else:
            token_rows, uncovered_rows = drilldown_new.flatten_concept_tree(tree)

            if drilldown_new.is_fully_unk(tree):
                st.error("This concept is **fully uncovered** — no selected candidate covers any of its relation types.")
            elif drilldown_new.is_exact_match(tree, picked_id):
                st.success("This concept is an **exact match** — it is itself in the candidate vocabulary.")
            else:
                st.info(f"This concept is tokenized via **{len(token_rows)} candidate assignment(s)**, shown below.")
                if uncovered_rows:
                    st.caption(f"{len(uncovered_rows)} branch(es) of this concept's context tree remain uncovered.")

            if token_rows:
                detail = (
                    pl.DataFrame(token_rows)
                    .with_columns(
                        pl.col("candidate_id").replace_strict(id_to_label, default=None).alias("candidate_label"),
                    )
                    .select("candidate_id", "candidate_label", "relation", "distance")
                    .sort("relation", "distance")
                )
            else:
                detail = pl.DataFrame(
                    schema={"candidate_id": pl.Utf8, "candidate_label": pl.Utf8, "relation": pl.Utf8, "distance": pl.Float64},
                )
            st.dataframe(detail, use_container_width=True, hide_index=True)

            st.markdown("#### Tokenized context tree")
            st.caption(
                "IS_A hops stay inside the same branch; every other relation opens a nested one. "
                "`•` = a token found; `∅` = a branch that died without one.",
            )
            st.code(drilldown_new.render_context_tree(tree, id_to_label), language=None)

            st.markdown("#### This concept's scores")
            concept_row = concept_df[concept_df["mapped_id"] == picked_id]
            method_row = df.filter((pl.col("method") == dd_method) & (pl.col("k") == dd_k))

            def _method_avg(col: str):
                return method_row[col][0] if method_row.height else None

            frac_sem_cov = concept_row["frac_sem_cov"].iloc[0] if len(concept_row) else None
            mean_distance = concept_row["mean_distance"].iloc[0] if len(concept_row) else None
            num_tokens = concept_row["num_tokens"].iloc[0] if len(concept_row) else None
            redundancy_group_size = concept_row["redundancy_group_size"].iloc[0] if len(concept_row) else None

            m1, m2, m3, m4 = st.columns(4)
            m1.metric(
                "Semantic coverage",
                f"{frac_sem_cov:.2f}" if frac_sem_cov is not None else "—",
                help=f"Method average at k={dd_k}: {_method_avg('semantic_coverage'):.3f}" if _method_avg("semantic_coverage") is not None else None,
            )
            m2.metric(
                "Mean token distance",
                f"{mean_distance:.2f}" if mean_distance is not None and mean_distance == mean_distance else "—",
                help="Average hop distance to this concept's found token(s); fully-uncovered isn't included here.",
            )
            m3.metric(
                "Number of tokens",
                int(num_tokens) if num_tokens is not None and num_tokens == num_tokens else "—",
                help=f"Method average tokens/concept at k={dd_k}: {_method_avg('conciseness'):.2f}" if _method_avg("conciseness") is not None else None,
            )
            m4.metric(
                "Concepts sharing this exact context-tree signature",
                int(redundancy_group_size) if redundancy_group_size is not None and redundancy_group_size == redundancy_group_size else "—",
                help="Higher = more concepts collapse onto the same representation as this one (lower uniqueness).",
            )

            if len(concept_row) and redundancy_group_size and redundancy_group_size > 1:
                sig_id = concept_row["signature_id"].iloc[0]
                other_ids = concept_df.loc[
                    (concept_df["signature_id"] == sig_id) & (concept_df["mapped_id"] != picked_id),
                    "mapped_id",
                ].tolist()
                sharing_df = (
                    pl.DataFrame({"id": other_ids})
                    .with_columns(pl.col("id").replace_strict(id_to_label, default=None).alias("label"))
                    .select("label", "id")
                    .sort("label")
                )
                with st.expander(f"Other {len(other_ids)} concept(s) sharing this exact representation"):
                    st.dataframe(sharing_df, use_container_width=True, hide_index=True)

            st.markdown("#### Graph view")
            st.caption(
                "Blue = the tokenized concept · Green = candidates actually used to tokenize it · "
                "Red = a branch that hit the depth limit or a dead end without finding a candidate · "
                f"Gray = other concepts reachable within {D} hops that were **not** selected as "
                "tokenizing candidates.",
            )

            html = graph_viz.build_new_concept_graph_html(picked_id, G, T, D, id_to_label)
            components.html(html, height=780, scrolling=True)
