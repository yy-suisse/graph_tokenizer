import polars as pl
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

from src import configs, drilldown, graph_viz

st.set_page_config(page_title="Graph Tokenizer — Candidate Selection Comparison", layout="wide")


@st.cache_data
def load_scores() -> pl.DataFrame:
    return pl.read_parquet(f"{configs.Results().path}scores.parquet")


@st.cache_resource
def load_tokenizer():
    concepts = drilldown.load_concepts()
    relations = drilldown.load_relations()
    candidate_reachable_child_map = drilldown.load_candidate_reachable_child_map()
    tokenizer = drilldown.build_tokenizer(concepts, relations, candidate_reachable_child_map)
    id_to_label = dict(zip(concepts["id"].to_list(), concepts["label"].to_list()))
    return tokenizer, relations, id_to_label


@st.cache_resource(show_spinner="Loading SNOMED IS_A graph...")
def load_is_a_graph():
    return drilldown.load_is_a_graph()


@st.cache_data
def load_mapped_concept_options(_concepts_dummy: int):
    concepts = drilldown.load_concepts()
    mapped = concepts.filter(pl.col("is_mapped"))
    return mapped.select("id", "label").to_pandas()


@st.cache_data(show_spinner="Tokenizing with this method/k...")
def cached_tokenize_for(method: str, k: int):
    tokenizer, _relations, _id_to_label = load_tokenizer()
    scores, results, df_tok_all_n_dist = drilldown.tokenize_for(tokenizer, method, k)
    return scores, results, df_tok_all_n_dist


SCORE_COLS = [
    "sem_cov_score",
    "distance_score",
    "uniqueness_entropy_score",
    "conciseness_score",
    "compression_rate",
    "UNK_rate",
    "exact_rate",
]

LOWER_IS_BETTER = {"compression_rate", "UNK_rate"}

df = load_scores()
methods = sorted(df["method"].unique().to_list())
k_values = sorted(df["k"].unique().to_list())

st.title("Candidate Selection — Method Comparison")
st.caption(
    "Browse precomputed tokenizer scores for every candidate-selection method, "
    "at every evaluated vocabulary size (k)."
)

with st.sidebar:
    st.header("Filters")
    selected_methods = st.multiselect("Methods", methods, default=methods)
    k_range = st.select_slider(
        "k range",
        options=k_values,
        value=(min(k_values), max(k_values)),
    )
    score_choice = st.selectbox("Score for the comparison chart", SCORE_COLS, index=0)

filtered = df.filter(
    pl.col("method").is_in(selected_methods)
    & (pl.col("k") >= k_range[0])
    & (pl.col("k") <= k_range[1])
)

tab_compare, tab_leaderboard, tab_table, tab_dist, tab_drilldown = st.tabs(
    ["Compare scores", "Leaderboard at k", "Raw table", "Score distributions", "Concept drill-down"],
)

with tab_compare:
    st.subheader(f"{score_choice} vs k")
    direction = "lower is better" if score_choice in LOWER_IS_BETTER else "higher is better"
    st.caption(f"({direction})")

    fig = px.line(
        filtered.sort("k").to_pandas(),
        x="k",
        y=score_choice,
        color="method",
        markers=True,
    )
    fig.update_layout(height=550)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("All scores, small multiples")
    cols = st.columns(2)
    for i, score in enumerate(SCORE_COLS):
        with cols[i % 2]:
            fig_small = px.line(
                filtered.sort("k").to_pandas(),
                x="k",
                y=score,
                color="method",
                markers=True,
            )
            fig_small.update_layout(height=300, showlegend=(i == 0), margin={"t": 30, "b": 10})
            st.plotly_chart(fig_small, use_container_width=True)

with tab_leaderboard:
    st.subheader("Leaderboard at a chosen k")
    k_pick = st.select_slider("k", options=k_values, value=k_values[len(k_values) // 2], key="leaderboard_k")
    score_pick = st.selectbox("Rank by", SCORE_COLS, index=0, key="leaderboard_score")
    ascending = score_pick in LOWER_IS_BETTER

    board = (
        filtered.filter(pl.col("k") == k_pick)
        .sort(score_pick, descending=not ascending)
        .select(["method", *SCORE_COLS])
    )
    st.dataframe(board, use_container_width=True, hide_index=True)

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
        dist_k = st.select_slider("k", options=k_values, key="dist_k")

    _dist_scores, dist_results, _dist_tok = cached_tokenize_for(dist_method, dist_k)
    concept_df = drilldown.get_all_concept_scores(dist_results).to_pandas()

    # --- aggregate KPI summary ---
    method_row = df.filter((pl.col("method") == dist_method) & (pl.col("k") == dist_k))
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
        ("redundancy_group_size", "Redundancy group size",            "How many concepts share the exact same set of assigned candidates",       False),
    ]

    for col_a, col_b in zip(DIST_SPECS[::2], DIST_SPECS[1::2]):
        left, right = st.columns(2)
        for pane, (col_name, title, x_label, show_unk_note) in zip([left, right], [col_a, col_b]):
            with pane:
                plot_df = concept_df[["mapped_id", col_name]].dropna()
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
                st.plotly_chart(fig, use_container_width=True)
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
        dd_k = st.select_slider("k", options=k_values, key="dd_k")

    if dd_method == "b_random_k":
        st.info(
            "`b_random_k` uses a single representative draw (iter 0) here, not the "
            "50-draw average shown in the leaderboard — scores will differ slightly.",
        )

    mapped_options = load_mapped_concept_options(0)

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

        scores, results, df_tok_all_n_dist = cached_tokenize_for(dd_method, dd_k)
        _tokenizer, relations, id_to_label = load_tokenizer()

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
        method_row = df.filter((pl.col("method") == dd_method) & (pl.col("k") == dd_k))

        def _method_avg(col: str):
            return method_row[col][0] if method_row.height else None

        m1, m2, m3, m4 = st.columns(4)
        m1.metric(
            "Semantic coverage",
            f"{concept_scores['frac_sem_cov']:.2f}" if concept_scores["frac_sem_cov"] is not None else "—",
            help=f"Method average at k={dd_k}: {_method_avg('sem_cov_score'):.3f}" if _method_avg("sem_cov_score") is not None else None,
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
            "Red = UNK (no coverage) · Gray dashed = every other concept reachable on the way "
            "there (direct neighbors and intermediate IS_A ancestors) that were "
            "**not** selected as tokenizing candidates.",
        )

        is_a_graph = load_is_a_graph()

        direct_neighbors = relations.filter(
            (pl.col("src.id") == picked_id) & (pl.col("distance") == 1),
        ).select("dst.id", "relation")

        used_candidates = concept_rows.select("candidate_id", "relation", "distance")

        html = graph_viz.build_concept_graph_html(
            picked_id,
            is_a_graph,
            direct_neighbors,
            used_candidates,
            id_to_label,
            max_dist=configs.TokenizerParam().max_dist_candidate,
        )
        components.html(html, height=780, scrolling=True)
