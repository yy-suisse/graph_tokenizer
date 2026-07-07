import polars as pl
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

from src import configs_fd, configs_mapped, drilldown, graph_viz

st.set_page_config(page_title="Graph Tokenizer — Candidate Selection Comparison", layout="wide")

CONFIGS = {"mapped": configs_mapped, "fd": configs_fd}
CONCEPT_SET_LABELS = {"mapped": "Mapped concepts", "fd": "Fully defined concepts"}


@st.cache_data
def load_scores(concept_set: str) -> pl.DataFrame:
    config = CONFIGS[concept_set]
    return pl.read_parquet(f"{config.Results().path}scores.parquet")


@st.cache_resource
def load_tokenizer(concept_set: str):
    concepts = drilldown.load_concepts(concept_set)
    relations = drilldown.load_relations(concept_set)
    df_relation = drilldown.load_df_relation(concept_set)
    candidate_reachable_child_map = drilldown.load_candidate_reachable_child_map(concept_set)
    tokenizer = drilldown.build_tokenizer(concepts, relations, df_relation, candidate_reachable_child_map, concept_set)
    id_to_label = dict(zip(concepts["id"].to_list(), concepts["label"].to_list()))
    return tokenizer, relations, id_to_label


@st.cache_resource(show_spinner="Loading SNOMED IS_A graph...")
def load_is_a_graph(concept_set: str):
    return drilldown.load_is_a_graph(concept_set)


@st.cache_data
def load_mapped_concept_options(concept_set: str):
    concepts = drilldown.load_concepts(concept_set)
    mapped = concepts.filter(pl.col("is_mapped"))
    return mapped.select("id", "label").to_pandas()


@st.cache_data(show_spinner="Tokenizing with this method/k...")
def cached_tokenize_for(concept_set: str, method: str, k: int):
    tokenizer, _relations, _id_to_label = load_tokenizer(concept_set)
    scores, results, df_tok_all_n_dist = drilldown.tokenize_for(tokenizer, concept_set, method, k)
    return scores, results, df_tok_all_n_dist


SCORE_COLS = [
    "final_score",
    "sem_cov_score",
    "distance_score",
    "uniqueness_entropy_score",
    "conciseness_score",
    "compression_rate",
    "UNK_rate",
    "exact_rate",
]

LOWER_IS_BETTER = {"compression_rate", "UNK_rate"}

concept_set = st.sidebar.radio(
    "Concept set",
    options=list(CONCEPT_SET_LABELS),
    format_func=lambda c: CONCEPT_SET_LABELS[c],
    key="concept_set",
)
config = CONFIGS[concept_set]

df = load_scores(concept_set).with_columns(
    final_score=(
        pl.col("distance_score") * pl.col("uniqueness_entropy_score") * pl.col("sem_cov_score")
    )
)
k_values = sorted(df["k"].unique().to_list())
num_candidates_values = sorted(df["num_candidates"].unique().to_list())
methods = sorted(df["method"].unique().to_list())

st.title("Candidate Selection — Method Comparison")
st.caption(
    f"Browse precomputed tokenizer scores for every candidate-selection method, on **{CONCEPT_SET_LABELS[concept_set]}**. "
    "Charts use **num_candidates** (actual unique assigned candidates) on the x-axis — "
    "not the nominal k — so methods are compared on equal footing regardless of redundancy."
)

with st.sidebar:
    st.header("Filters")
    nc_range = st.select_slider(
        "num_candidates range",
        options=num_candidates_values,
        value=(min(num_candidates_values), max(num_candidates_values)),
    )
    score_choice = st.selectbox("Score for the comparison chart", SCORE_COLS, index=0)

# --- Method toggle panel ---
with st.container(border=True):
    toggle_cols = st.columns([1, 1] + [3] * len(methods))
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
    & (pl.col("num_candidates") >= nc_range[0])
    & (pl.col("num_candidates") <= nc_range[1])
)

tab_compare, tab_table, tab_dist, tab_drilldown = st.tabs(
    ["Compare scores", "Raw table", "Score distributions", "Concept drill-down"],
)

with tab_compare:
    st.subheader(f"{score_choice} vs num_candidates")
    direction = "lower is better" if score_choice in LOWER_IS_BETTER else "higher is better"
    st.caption(f"({direction})")

    fig = px.line(
        filtered.sort("num_candidates").to_pandas(),
        x="num_candidates",
        y=score_choice,
        color="method",
        markers=True,
    )
    fig.update_layout(height=550, legend_itemclick=False, legend_itemdoubleclick=False)
    st.plotly_chart(fig, use_container_width=True, key="compare_main")

    st.subheader("All scores, small multiples")
    compare_cols = st.columns(2)
    for i, score in enumerate(SCORE_COLS):
        with compare_cols[i % 2]:
            fig_small = px.line(
                filtered.sort("num_candidates").to_pandas(),
                x="num_candidates",
                y=score,
                color="method",
                markers=True,
            )
            fig_small.update_layout(
                height=300,
                showlegend=(i == 0),
                legend_itemclick=False,
                legend_itemdoubleclick=False,
                margin={"t": 30, "b": 10},
            )
            st.plotly_chart(fig_small, use_container_width=True, key=f"compare_small_{score}")

with tab_table:
    st.subheader("Filtered raw scores")
    st.dataframe(filtered.sort(["method", "k"]), use_container_width=True, hide_index=True)
    st.download_button(
        "Download filtered scores as CSV",
        filtered.write_csv(),
        file_name="filtered_scores.csv",
        mime="text/csv",
    )

with tab_dist:
    st.subheader("Per-concept score distributions")
    st.caption(
        "Choose a method and k to see how individual scores are spread across all 46 k mapped concepts "
        "— revealing whether good aggregate scores hide a long tail of poorly-covered concepts.",
    )

    col_dm, col_dk = st.columns(2)
    with col_dm:
        dist_method = st.selectbox("Method", methods, key="dist_method")
    with col_dk:
        dist_method_rows = df.filter(pl.col("method") == dist_method).sort("num_candidates")
        dist_nc_options = dist_method_rows["num_candidates"].to_list()
        dist_nc = st.select_slider("num_candidates", options=dist_nc_options, key="dist_nc")

    method_row = dist_method_rows.filter(pl.col("num_candidates") == dist_nc)
    dist_k = int(method_row["k"][0])
    _dist_scores, dist_results, _dist_tok = cached_tokenize_for(concept_set, dist_method, dist_k)
    concept_df = drilldown.get_all_concept_scores(dist_results).to_pandas()
    st.markdown("**Method-level aggregate scores at this k**")
    agg_cols = st.columns(len(SCORE_COLS))
    for col, name in zip(agg_cols, SCORE_COLS):
        val = method_row[name][0] if method_row.height else None
        col.metric(name.replace("_", " "), f"{val:.3f}" if val is not None else "—")

    st.divider()

    # --- distributions ---
    DIST_SPECS = [
        ("frac_sem_cov",          "Semantic coverage per concept",   "Fraction of relation types covered (1.0 = full coverage, 0.0 = UNK)",    True),
        ("mean_distance",         "Mean token distance per concept",  "Average hop distance to assigned candidate(s); UNK concepts excluded",   False),
        ("num_tokens",            "Number of tokens per concept",     "How many distinct candidates a concept expands into",                     False),
        ("redundancy_group_size", "Concepts sharing the same token set — how many mapped concepts have an identical token combination", "Number of concepts in the same token group (1 = fully unique representation)", False),
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
                        go.Bar(name="Unique token set", x=["Token set uniqueness"], y=[n_unique],
                               marker_color="#2ecc71",
                               text=[f"{n_unique} ({100*n_unique/total:.1f}%)"], textposition="outside"),
                        go.Bar(name="Shared token set (≥2 concepts)", x=["Token set uniqueness"], y=[n_shared],
                               marker_color="#e74c3c",
                               text=[f"{n_shared} ({100*n_shared/total:.1f}%)"], textposition="outside"),
                    ])
                    fig_bar.update_layout(
                        barmode="stack",
                        title="Unique vs. shared token sets",
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
                    st.caption(f"{unk_n} UNK concept(s) (frac_sem_cov = 0) included at the left edge.")
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
        "so picking a new method/k takes a few seconds the first time.",
    )

    col_method, col_k = st.columns(2)
    with col_method:
        dd_method = st.selectbox("Method", methods, key="dd_method")
    with col_k:
        dd_method_rows = df.filter(pl.col("method") == dd_method).sort("num_candidates")
        dd_nc_options = dd_method_rows["num_candidates"].to_list()
        dd_nc = st.select_slider("num_candidates", options=dd_nc_options, key="dd_nc")

    dd_k = int(dd_method_rows.filter(pl.col("num_candidates") == dd_nc)["k"][0])

    if dd_method == "b_random_k":
        st.info(
            "`b_random_k` uses a single representative draw (iter 0) here, not the "
            "50-draw average shown in the leaderboard — scores will differ slightly.",
        )

    mapped_options = load_mapped_concept_options(concept_set)

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

        scores, results, df_tok_all_n_dist = cached_tokenize_for(concept_set, dd_method, dd_k)
        _tokenizer, relations, id_to_label = load_tokenizer(concept_set)

        concept_rows = df_tok_all_n_dist.filter(pl.col("mapped_id") == picked_id)

        st.markdown(f"### {id_to_label.get(picked_id, picked_id)} (`{picked_id}`)")

        if concept_rows.height == 0:
            st.warning("This concept was not found in the tokenization output.")
        elif (concept_rows["candidate_id"] == "UNK").all():
            st.error("This concept is **UNK** — no selected candidate covers any of its relation types.")
        elif concept_rows.height == 1 and concept_rows["candidate_id"][0] == picked_id and concept_rows["distance"][0] == 0.0:
            st.success("This concept is an **exact match** — it is itself in the candidate vocabulary.")
        else:
            st.info(f"This concept is tokenized via **{concept_rows.height} candidate(s)**, shown below.")

        detail = (
            concept_rows.with_columns(
                pl.col("candidate_id").replace_strict(id_to_label, default=None).alias("candidate_label"),
            )
            .select("candidate_id", "candidate_label", "relation", "distance")
            .sort("relation", "distance")
        )
        st.dataframe(detail, use_container_width=True, hide_index=True)

        st.markdown("#### This concept's scores")
        concept_scores = drilldown.get_concept_scores(results, picked_id)
        method_row = df.filter((pl.col("method") == dd_method) & (pl.col("num_candidates") == dd_nc))

        def _method_avg(col: str):
            return method_row[col][0] if method_row.height else None

        m1, m2, m3, m4 = st.columns(4)
        m1.metric(
            "Semantic coverage",
            f"{concept_scores['frac_sem_cov']:.2f}" if concept_scores["frac_sem_cov"] is not None else "—",
            help=f"Method average at num_candidates={dd_nc}: {_method_avg('sem_cov_score'):.3f}" if _method_avg("sem_cov_score") is not None else None,
        )
        m2.metric(
            "Mean token distance",
            f"{concept_scores['mean_distance']:.2f}" if concept_scores["mean_distance"] is not None else "—",
            help="Average hop distance to this concept's assigned token(s); null distance (UNK) isn't included here.",
        )
        m3.metric(
            "Number of tokens",
            concept_scores["num_tokens"] if concept_scores["num_tokens"] is not None else "—",
            help=f"Method average tokens/concept at k={dd_k}: {1 / _method_avg('conciseness_score'):.2f}" if _method_avg("conciseness_score") else None,
        )
        m4.metric(
            "Concepts sharing this exact token set",
            concept_scores["redundancy_group_size"] if concept_scores["redundancy_group_size"] is not None else "—",
            help="Higher = more concepts collapse onto the same candidate(s) as this one (lower uniqueness).",
        )

        st.markdown("#### Graph view")
        st.caption(
            "Blue = the tokenized concept · Green = candidates actually used to tokenize it · "
            "Red = UNK (no coverage) · Gray dashed = every other concept reachable within "
            f"{config.TokenizerParam().max_dist_candidate} hops (via the real intermediate "
            "IS_A chain) that were **not** selected as tokenizing candidates.",
        )

        is_a_graph = load_is_a_graph(concept_set)

        neighbors = relations.filter(
            pl.col("src.id") == picked_id,
        ).select("dst.id", "relation", "distance")

        used_candidates = concept_rows.select("candidate_id", "relation", "distance")

        html = graph_viz.build_concept_graph_html(
            picked_id,
            is_a_graph,
            neighbors,
            used_candidates,
            id_to_label,
        )
        components.html(html, height=780, scrolling=True)
