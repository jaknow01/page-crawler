"""
SEO Competitor Analysis — Streamlit UI.

Visualises the vector database, cluster map, gap analysis and page explorer.
Data sources: data/reports/reduction.parquet + data/reports/gap_report.json
"""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="SEO Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=IBM+Plex+Mono:wght@300;400;500&display=swap" rel="stylesheet">

<style>
/* ── base ── */
html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #080c18;
    color: #c8d6e8;
}
.stApp { background-color: #080c18; }

/* ── sidebar ── */
[data-testid="stSidebar"] {
    background-color: #0b1020;
    border-right: 1px solid #1a2540;
}
[data-testid="stSidebar"] .stRadio label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #4a6080;
    transition: color 0.2s;
}
[data-testid="stSidebar"] .stRadio label:hover { color: #f59e0b; }

/* ── metric cards ── */
[data-testid="stMetric"] {
    background: #0f1628;
    border: 1px solid #1a2540;
    border-radius: 6px;
    padding: 1rem 1.2rem;
}
[data-testid="stMetricLabel"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #4a6080 !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2rem !important;
    color: #e2e8f0 !important;
}

/* ── headings ── */
h1 { font-family: 'Syne', sans-serif; font-weight: 800; color: #f1f5f9; letter-spacing: -0.02em; }
h2 { font-family: 'Syne', sans-serif; font-weight: 700; color: #cbd5e1; letter-spacing: -0.01em; }
h3 { font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; font-weight: 500;
     letter-spacing: 0.1em; text-transform: uppercase; color: #3b82f6; }

/* ── cluster badge ── */
.cluster-badge {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    padding: 2px 8px;
    border-radius: 3px;
    background: #1e2d4a;
    color: #60a5fa;
    border: 1px solid #2d4070;
    letter-spacing: 0.05em;
}

/* ── gap card ── */
.gap-card {
    background: #0f1628;
    border: 1px solid #1a2540;
    border-left: 3px solid #ef4444;
    border-radius: 6px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.75rem;
}
.gap-card.covered {
    border-left-color: #10b981;
}
.gap-card h4 {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 0.95rem;
    color: #e2e8f0;
    margin: 0 0 0.4rem 0;
}
.gap-card .meta {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: #4a6080;
    letter-spacing: 0.05em;
}
.gap-card .score-bar-wrap {
    height: 4px;
    background: #1a2540;
    border-radius: 2px;
    margin: 0.6rem 0;
}
.gap-card .score-bar {
    height: 4px;
    border-radius: 2px;
    background: linear-gradient(90deg, #ef4444, #f59e0b, #10b981);
}
.page-link {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: #3b82f6;
    text-decoration: none;
    word-break: break-all;
}

/* ── explorer table ── */
.stDataFrame { border: 1px solid #1a2540; border-radius: 6px; }
[data-testid="stDataFrameResizable"] { background: #0f1628; }

/* ── divider ── */
hr { border-color: #1a2540; }

/* ── scrollbar ── */
::-webkit-scrollbar { width: 6px; background: #080c18; }
::-webkit-scrollbar-thumb { background: #1a2540; border-radius: 3px; }

/* ── plotly container ── */
.js-plotly-plot { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ── constants ─────────────────────────────────────────────────────────────────

REPORTS = Path("data/reports")
REDUCTION_FILE = REPORTS / "reduction.parquet"
GAP_REPORT_FILE = REPORTS / "gap_report.json"

OWN_COLOR = "#f59e0b"      # amber — instantly visible on dark bg
NOISE_COLOR = "#1e2d4a"    # dark slate for unclassified points

CLUSTER_PALETTE = [
    "#3b82f6", "#8b5cf6", "#06b6d4", "#10b981",
    "#f472b6", "#a78bfa", "#34d399", "#60a5fa",
    "#c084fc", "#2dd4bf", "#fb923c", "#e879f9",
]

# ── data loading ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def load_reduction() -> pd.DataFrame | None:
    if not REDUCTION_FILE.exists():
        return None
    return pd.read_parquet(REDUCTION_FILE)


@st.cache_data(ttl=60)
def load_gap_report() -> dict | None:
    if not GAP_REPORT_FILE.exists():
        return None
    return json.loads(GAP_REPORT_FILE.read_text())


def cluster_color(cluster_id: int) -> str:
    if cluster_id == -1:
        return NOISE_COLOR
    return CLUSTER_PALETTE[cluster_id % len(CLUSTER_PALETTE)]

# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ◈ SEO Intelligence")
    st.markdown('<p style="font-family:\'IBM Plex Mono\',monospace;font-size:0.65rem;color:#2d4070;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:1.5rem;">Competitor Analysis</p>', unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["Dashboard", "Cluster Map", "Gap Analysis", "Page Explorer"],
        label_visibility="collapsed",
    )
    st.markdown("---")

    df = load_reduction()
    gap = load_gap_report()

    if df is not None:
        own_n = (df["source"] == "own").sum()
        comp_n = (df["source"] == "competitor").sum()
        domains_n = df[df["source"] == "competitor"]["domain"].nunique()
        clusters_n = df[df["cluster_id"] >= 0]["cluster_id"].nunique()
        st.markdown(f'<p class="meta" style="font-family:\'IBM Plex Mono\',monospace;font-size:0.68rem;color:#2d4070;line-height:1.8;">'
                    f'Own pages: <span style="color:#f59e0b">{own_n}</span><br>'
                    f'Competitor pages: <span style="color:#60a5fa">{comp_n}</span><br>'
                    f'Domains: <span style="color:#60a5fa">{domains_n}</span><br>'
                    f'Clusters: <span style="color:#60a5fa">{clusters_n}</span>'
                    f'</p>', unsafe_allow_html=True)

# ── helper: no data state ─────────────────────────────────────────────────────

def no_data_warning(file: str):
    st.markdown(f"""
    <div style="background:#0f1628;border:1px solid #1a2540;border-radius:8px;padding:2rem;text-align:center;margin-top:3rem;">
        <p style="font-family:'IBM Plex Mono',monospace;font-size:0.75rem;color:#4a6080;letter-spacing:0.1em;text-transform:uppercase;">No data found</p>
        <p style="color:#64748b;font-size:0.9rem;">Missing: <code style="color:#f59e0b">{file}</code></p>
        <p style="color:#4a6080;font-size:0.85rem;">Run the full pipeline first:<br>
        <code style="color:#60a5fa">uv run python main.py run-all</code></p>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

if page == "Dashboard":
    st.markdown("# SEO Intelligence")
    st.markdown("---")

    if df is None:
        no_data_warning("data/reports/reduction.parquet")
        st.stop()

    own = df[df["source"] == "own"]
    comp = df[df["source"] == "competitor"]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Own pages", f"{len(own):,}")
    with col2:
        st.metric("Competitor pages", f"{len(comp):,}")
    with col3:
        st.metric("Domains", f"{comp['domain'].nunique():,}")
    with col4:
        valid_clusters = df[df["cluster_id"] >= 0]["cluster_id"].nunique()
        st.metric("Clusters", f"{valid_clusters:,}")

    st.markdown("---")

    col_a, col_b = st.columns([3, 2])

    with col_a:
        st.markdown("### Top domains by page count")
        top_domains = (
            comp.groupby("domain").size()
            .reset_index(name="pages")
            .sort_values("pages", ascending=False)
            .head(15)
        )
        fig = px.bar(
            top_domains, x="pages", y="domain", orientation="h",
            color="pages",
            color_continuous_scale=[[0, "#1e2d4a"], [1, "#3b82f6"]],
        )
        fig.update_layout(
            plot_bgcolor="#0f1628", paper_bgcolor="#0f1628",
            font=dict(family="IBM Plex Mono", color="#c8d6e8", size=11),
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(gridcolor="#1a2540", zerolinecolor="#1a2540"),
            yaxis=dict(gridcolor="rgba(0,0,0,0)", categoryorder="total ascending"),
            coloraxis_showscale=False,
            height=380,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("### Cluster sizes")
        if gap:
            cluster_sizes = pd.DataFrame([
                {"label": c["cluster_label"][:30], "pages": c["page_count"], "covered": c["covered"]}
                for c in sorted(gap["clusters"], key=lambda x: x["page_count"], reverse=True)[:12]
            ])
            colors = [("#10b981" if r["covered"] else "#ef4444") for _, r in cluster_sizes.iterrows()]
            fig2 = px.bar(
                cluster_sizes, x="pages", y="label", orientation="h",
                color_discrete_sequence=colors,
            )
            fig2.update_traces(marker_color=colors)
            fig2.update_layout(
                plot_bgcolor="#0f1628", paper_bgcolor="#0f1628",
                font=dict(family="IBM Plex Mono", color="#c8d6e8", size=10),
                margin=dict(l=0, r=0, t=10, b=0),
                xaxis=dict(gridcolor="#1a2540", zerolinecolor="#1a2540"),
                yaxis=dict(gridcolor="rgba(0,0,0,0)", categoryorder="total ascending"),
                showlegend=False, height=380,
            )
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown(
                '<span style="color:#10b981;font-family:\'IBM Plex Mono\',monospace;font-size:0.7rem">■ covered</span>&nbsp;&nbsp;'
                '<span style="color:#ef4444;font-family:\'IBM Plex Mono\',monospace;font-size:0.7rem">■ missing</span>',
                unsafe_allow_html=True,
            )

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: CLUSTER MAP
# ─────────────────────────────────────────────────────────────────────────────

elif page == "Cluster Map":
    st.markdown("# Cluster Map")
    st.markdown('<p style="color:#4a6080;font-size:0.85rem;margin-top:-0.5rem;">2D UMAP projection of all page embeddings — own site in amber, competitors by cluster</p>', unsafe_allow_html=True)

    if df is None:
        no_data_warning("data/reports/reduction.parquet")
        st.stop()

    # Filter controls
    col_f1, col_f2, col_f3 = st.columns([2, 2, 1])
    with col_f1:
        all_labels = sorted(df[df["cluster_id"] >= 0]["cluster_label"].unique())
        selected_clusters = st.multiselect("Filter clusters", all_labels, placeholder="All clusters")
    with col_f2:
        all_domains = sorted(df[df["source"] == "competitor"]["domain"].unique())
        selected_domains = st.multiselect("Filter domains", all_domains, placeholder="All domains")
    with col_f3:
        show_noise = st.checkbox("Show unclassified", value=False)

    # Apply filters
    mask = pd.Series([True] * len(df), index=df.index)
    if selected_clusters:
        mask &= df["cluster_label"].isin(selected_clusters) | (df["source"] == "own")
    if selected_domains:
        mask &= df["domain"].isin(selected_domains) | (df["source"] == "own")
    if not show_noise:
        mask &= (df["cluster_id"] >= 0) | (df["source"] == "own")

    plot_df = df[mask].copy()

    # Build Plotly figure
    fig = go.Figure()

    # Competitor points — one trace per cluster for legend
    comp_df = plot_df[plot_df["source"] == "competitor"]
    if show_noise:
        noise_df = comp_df[comp_df["cluster_id"] == -1]
        if not noise_df.empty:
            fig.add_trace(go.Scattergl(
                x=noise_df["x"], y=noise_df["y"],
                mode="markers",
                marker=dict(size=4, color=NOISE_COLOR, opacity=0.4),
                name="unclassified",
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "<span style='color:#6b7280'>%{customdata[1]}</span><br>"
                    "<i>%{customdata[2]}</i><extra></extra>"
                ),
                customdata=noise_df[["title", "domain", "summary"]].values,
            ))

    for cluster_id in sorted(comp_df[comp_df["cluster_id"] >= 0]["cluster_id"].unique()):
        c_df = comp_df[comp_df["cluster_id"] == cluster_id]
        label = c_df["cluster_label"].iloc[0] if not c_df.empty else str(cluster_id)
        color = cluster_color(cluster_id)
        fig.add_trace(go.Scattergl(
            x=c_df["x"], y=c_df["y"],
            mode="markers",
            marker=dict(size=5, color=color, opacity=0.65,
                        line=dict(width=0.5, color="rgba(255,255,255,0.1)")),
            name=label[:35],
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "<span style='color:#6b7280'>%{customdata[1]}</span><br>"
                "<i>%{customdata[2]}</i><br>"
                "<span style='color:#4a6080'>%{customdata[3]}</span><extra></extra>"
            ),
            customdata=c_df[["title", "domain", "summary", "keywords"]].values,
        ))

    # Own site points — on top, amber, larger, diamond shape
    own_df = plot_df[plot_df["source"] == "own"]
    if not own_df.empty:
        fig.add_trace(go.Scattergl(
            x=own_df["x"], y=own_df["y"],
            mode="markers",
            marker=dict(
                size=10, color=OWN_COLOR, symbol="diamond",
                line=dict(width=1.5, color="#ffffff"),
                opacity=1.0,
            ),
            name="◈ Own site",
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "<i>%{customdata[1]}</i><br>"
                "<span style='color:#4a6080'>%{customdata[2]}</span><extra></extra>"
            ),
            customdata=own_df[["title", "summary", "keywords"]].values,
        ))

    fig.update_layout(
        plot_bgcolor="#080c18",
        paper_bgcolor="#080c18",
        font=dict(family="IBM Plex Mono", color="#c8d6e8", size=10),
        legend=dict(
            bgcolor="#0b1020", bordercolor="#1a2540", borderwidth=1,
            font=dict(size=10), itemsizing="constant",
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        height=620,
        hovermode="closest",
        dragmode="pan",
    )

    st.plotly_chart(fig, use_container_width=True)
    st.markdown(f'<p style="font-family:\'IBM Plex Mono\',monospace;font-size:0.68rem;color:#2d4070;text-align:right">'
                f'{len(plot_df):,} points · {len(comp_df["cluster_id"].unique())} clusters</p>',
                unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: GAP ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

elif page == "Gap Analysis":
    st.markdown("# Gap Analysis")
    st.markdown('<p style="color:#4a6080;font-size:0.85rem;margin-top:-0.5rem;">Clusters ranked from most missing to best covered — red border = content gap, green = covered</p>', unsafe_allow_html=True)

    if gap is None:
        no_data_warning("data/reports/gap_report.json")
        st.stop()

    summary = gap["summary"]
    clusters = gap["clusters"]

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total clusters", summary["total_clusters"])
    with col2:
        st.metric("Missing", summary["missing_clusters"], delta=None)
    with col3:
        st.metric("Covered", summary["covered_clusters"])
    with col4:
        pct = round(summary["covered_clusters"] / max(summary["total_clusters"], 1) * 100)
        st.metric("Coverage", f"{pct}%")

    st.markdown("---")

    # Filter
    col_f1, col_f2 = st.columns([2, 1])
    with col_f1:
        show_filter = st.radio("Show", ["All", "Missing only", "Covered only"], horizontal=True)
    with col_f2:
        min_pages = st.slider("Min pages in cluster", 1, 50, 1)

    filtered = clusters
    if show_filter == "Missing only":
        filtered = [c for c in clusters if not c["covered"]]
    elif show_filter == "Covered only":
        filtered = [c for c in clusters if c["covered"]]
    filtered = [c for c in filtered if c["page_count"] >= min_pages]

    st.markdown(f'<p style="font-family:\'IBM Plex Mono\',monospace;font-size:0.7rem;color:#2d4070;">{len(filtered)} clusters shown</p>', unsafe_allow_html=True)

    for c in filtered:
        covered = c["covered"]
        score_pct = int(c["coverage_score"] * 100)
        card_class = "gap-card covered" if covered else "gap-card"
        status_color = "#10b981" if covered else "#ef4444"
        status_label = "covered" if covered else "gap"

        examples_html = ""
        for ex in c["example_pages"][:3]:
            examples_html += (
                f'<div style="margin:0.3rem 0 0 0;">'
                f'<a class="page-link" href="{ex["url"]}" target="_blank">{ex["title"][:70] or ex["url"][:70]}</a>'
                f'<span style="color:#2d4070;font-family:\'IBM Plex Mono\',monospace;font-size:0.65rem"> — {ex["domain"]}</span>'
                f'</div>'
            )

        best_own = c.get("best_own_page", {})
        best_own_html = ""
        if best_own.get("url"):
            best_own_html = (
                f'<div style="margin-top:0.6rem;padding-top:0.5rem;border-top:1px solid #1a2540;">'
                f'<span style="font-family:\'IBM Plex Mono\',monospace;font-size:0.65rem;color:#2d4070;text-transform:uppercase;letter-spacing:0.08em;">Closest own page (dist {best_own["distance"]:.3f})</span><br>'
                f'<a class="page-link" style="color:#f59e0b" href="{best_own["url"]}" target="_blank">{best_own.get("title","")[:70] or best_own["url"][:60]}</a>'
                f'</div>'
            )

        st.markdown(f"""
        <div class="{card_class}">
            <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                <h4>{c['cluster_label']}</h4>
                <span style="font-family:'IBM Plex Mono',monospace;font-size:0.7rem;color:{status_color};letter-spacing:0.08em;text-transform:uppercase;">{status_label}</span>
            </div>
            <div class="meta">{c['page_count']} competitor pages · distance {c['min_distance_to_own']:.3f}</div>
            <div class="score-bar-wrap">
                <div class="score-bar" style="width:{score_pct}%"></div>
            </div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.65rem;color:#2d4070;margin-bottom:0.5rem;">coverage {score_pct}%</div>
            {examples_html}
            {best_own_html}
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: PAGE EXPLORER
# ─────────────────────────────────────────────────────────────────────────────

elif page == "Page Explorer":
    st.markdown("# Page Explorer")
    st.markdown('<p style="color:#4a6080;font-size:0.85rem;margin-top:-0.5rem;">Browse all pages in the vector database</p>', unsafe_allow_html=True)

    if df is None:
        no_data_warning("data/reports/reduction.parquet")
        st.stop()

    # Filters
    col1, col2, col3 = st.columns([1, 2, 2])
    with col1:
        source_filter = st.selectbox("Source", ["All", "own", "competitor"])
    with col2:
        domain_opts = ["All"] + sorted(df["domain"].dropna().unique())
        domain_filter = st.selectbox("Domain", domain_opts)
    with col3:
        label_opts = ["All"] + sorted(df[df["cluster_id"] >= 0]["cluster_label"].dropna().unique())
        cluster_filter = st.selectbox("Cluster", label_opts)

    search = st.text_input("Search title / URL", placeholder="logopeda...")

    filtered = df.copy()
    if source_filter != "All":
        filtered = filtered[filtered["source"] == source_filter]
    if domain_filter != "All":
        filtered = filtered[filtered["domain"] == domain_filter]
    if cluster_filter != "All":
        filtered = filtered[filtered["cluster_label"] == cluster_filter]
    if search:
        mask = (
            filtered["title"].str.contains(search, case=False, na=False) |
            filtered["url"].str.contains(search, case=False, na=False)
        )
        filtered = filtered[mask]

    st.markdown(f'<p style="font-family:\'IBM Plex Mono\',monospace;font-size:0.7rem;color:#2d4070;">{len(filtered):,} pages</p>', unsafe_allow_html=True)

    display_cols = ["source", "domain", "cluster_label", "title", "url", "keywords", "summary"]
    available = [c for c in display_cols if c in filtered.columns]

    st.dataframe(
        filtered[available].reset_index(drop=True),
        use_container_width=True,
        height=480,
        column_config={
            "url": st.column_config.LinkColumn("URL"),
            "source": st.column_config.TextColumn("Source", width="small"),
            "domain": st.column_config.TextColumn("Domain", width="medium"),
            "cluster_label": st.column_config.TextColumn("Cluster", width="medium"),
            "title": st.column_config.TextColumn("Title", width="large"),
            "keywords": st.column_config.TextColumn("Keywords"),
            "summary": st.column_config.TextColumn("Summary"),
        },
    )

    if len(filtered) > 0:
        st.markdown("---")
        st.markdown("### Page detail")
        idx = st.selectbox("Select row", range(min(50, len(filtered))),
                           format_func=lambda i: filtered.iloc[i]["title"][:80] or filtered.iloc[i]["url"])
        row = filtered.iloc[idx]

        col_a, col_b = st.columns([3, 2])
        with col_a:
            st.markdown(f"**{row.get('title','')}**")
            st.markdown(f'<a class="page-link" href="{row["url"]}" target="_blank">{row["url"]}</a>', unsafe_allow_html=True)
            st.markdown(f"*{row.get('summary','')}*")
            st.text_area("Content snippet", row.get("content_snippet", ""), height=120)
        with col_b:
            st.markdown(f"**Source:** `{row.get('source','')}`")
            st.markdown(f"**Domain:** `{row.get('domain','')}`")
            st.markdown(f"**Cluster:** `{row.get('cluster_label','unclassified')}`")
            st.markdown(f"**Keywords:** {row.get('keywords','')}")
