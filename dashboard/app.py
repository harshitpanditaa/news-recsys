"""Personalized News Recommendation System - interactive Streamlit dashboard.

Reads ONLY real project outputs exported by pipeline/step09_export_dashboard.py
(dashboard/data/*.parquet + *.json). Run:  streamlit run dashboard/app.py
"""
import json
import os

import pandas as pd
import plotly.express as px
import streamlit as st

DATA = os.environ.get("NEWSREC_DASHBOARD_DATA", os.path.join(os.path.dirname(__file__), "data"))
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SEQ = ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]

st.set_page_config(page_title="News Recommender", page_icon="📰", layout="wide")


@st.cache_data
def pq(name):
    p = os.path.join(DATA, f"{name}.parquet")
    return pd.read_parquet(p) if os.path.exists(p) else pd.DataFrame()


@st.cache_data
def js(name):
    p = os.path.join(DATA, f"{name}.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def style(fig, height=360):
    fig.update_layout(template="simple_white", height=height, margin=dict(l=10, r=10, t=40, b=10),
                      font=dict(size=13), hoverlabel=dict(bgcolor="white"))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="#ececea")
    return fig


if not os.path.exists(os.path.join(DATA, "kpis.json")):
    st.error(f"No dashboard data found in `{DATA}`. Run the pipeline (Step 9 export) first.")
    st.stop()

kpis, train_m, eval_m, valid = js("kpis"), js("training_metrics"), js("evaluation_metrics"), js("validation_report")

st.sidebar.title("📰 News Recommender")
st.sidebar.caption("Personalized News Recommendation System using Big Data")
page = st.sidebar.radio("Page", ["Overview", "Trends", "Personalized feed", "Model results",
                                 "User segments", "Data quality & pipeline"])
st.sidebar.markdown("---")
st.sidebar.caption(f"Data: MIND-small (Microsoft) + The Guardian Open Platform API\n\n"
                   f"Exported: {kpis.get('exported_at_utc', '')[:19]} UTC")

# ---------------------------------------------------------------- Overview
if page == "Overview":
    st.title("Overview")
    st.caption("Key numbers from the Silver/Gold layers and the evaluated model.")
    c = st.columns(4)
    c[0].metric("Users", f"{kpis['users']:,}")
    c[1].metric("Articles (MIND + Guardian)", f"{kpis['articles_mind'] + kpis['articles_guardian']:,}",
                help=f"MIND {kpis['articles_mind']:,} · Guardian {kpis['articles_guardian']:,}")
    c[2].metric("Impressions", f"{kpis['impressions']:,}")
    c[3].metric("Clicks", f"{kpis['clicks']:,}")
    c = st.columns(4)
    c[0].metric("Overall CTR", f"{kpis['ctr']:.2%}")
    c[1].metric("Reading sessions", f"{kpis['sessions']:,}")
    if eval_m:
        m = eval_m["model"]
        c[2].metric("Model AUC (held-out day)", f"{m['AUC']:.3f}",
                    delta=f"{m['AUC'] - eval_m['baseline_random']['AUC']:+.3f} vs random")
        c[3].metric("Model nDCG@10", f"{m['nDCG@10']:.3f}",
                    delta=f"{m['nDCG@10'] - eval_m['baseline_random']['nDCG@10']:+.3f} vs random")

    daily = pq("daily_kpis")
    l, r = st.columns(2)
    if not daily.empty:
        l.plotly_chart(style(px.line(daily, x="event_date", y="clicks", markers=True,
                                     title="Clicks per day", color_discrete_sequence=[BLUE],
                                     labels={"event_date": "", "clicks": "Clicks"})), use_container_width=True)
        r.plotly_chart(style(px.line(daily, x="event_date", y="active_users", markers=True,
                                     title="Active users per day", color_discrete_sequence=[BLUE],
                                     labels={"event_date": "", "active_users": "Users"})), use_container_width=True)
    cat = pq("category_kpis").sort_values("clicks")
    l, r = st.columns(2)
    l.plotly_chart(style(px.bar(cat, x="clicks", y="category", orientation="h", title="Clicks by category",
                                color_discrete_sequence=[BLUE], labels={"category": "", "clicks": "Clicks"}),
                         460), use_container_width=True)
    cat2 = cat[cat["impressions"] >= 1000].sort_values("ctr")
    fig = px.bar(cat2, x="ctr", y="category", orientation="h", title="Click-through rate by category",
                 color_discrete_sequence=[BLUE], labels={"category": "", "ctr": "CTR"})
    fig.update_xaxes(tickformat=".1%")
    r.plotly_chart(style(fig, 460), use_container_width=True)

# ---------------------------------------------------------------- Trends
elif page == "Trends":
    st.title("Reading trends")
    hourly = pq("hourly_trends")
    cats = hourly.groupby("category")["clicks"].sum().sort_values(ascending=False).index.tolist()
    chosen = st.multiselect("Categories (max 3)", cats, default=cats[:3], max_selections=3)
    sub = hourly[hourly["category"].isin(chosen)].sort_values("hour_ts")
    color_map = {c_: col for c_, col in zip(chosen, [BLUE, ORANGE, AQUA])}
    st.plotly_chart(style(px.line(sub, x="hour_ts", y="clicks", color="category", title="Clicks per hour",
                                  color_discrete_map=color_map, labels={"hour_ts": "", "clicks": "Clicks"})),
                    use_container_width=True)
    hourly["hour"] = pd.to_datetime(hourly["hour_ts"]).dt.hour
    hm = hourly.groupby(["category", "hour"])[["clicks", "impressions"]].sum().reset_index()
    hm["ctr"] = hm["clicks"] / hm["impressions"]
    hm = hm[hm["category"].isin(cats[:12])]
    fig = px.density_heatmap(hm, x="hour", y="category", z="ctr", histfunc="avg", nbinsx=24,
                             color_continuous_scale=SEQ, title="CTR by hour of day (UTC) and category",
                             labels={"hour": "Hour of day", "category": ""})
    fig.update_coloraxes(colorbar_title="CTR", colorbar_tickformat=".0%")
    st.plotly_chart(style(fig, 460), use_container_width=True)
    st.subheader("Most-clicked articles")
    top = pq("top_news")
    pick = st.selectbox("Category", ["All"] + sorted(top["category"].dropna().unique().tolist()))
    if pick != "All":
        top = top[top["category"] == pick]
    st.dataframe(top.head(50)[["title", "category", "shown", "clicked", "ctr", "history_readers"]],
                 use_container_width=True, hide_index=True,
                 column_config={"ctr": st.column_config.NumberColumn("CTR", format="%.3f")})

# ---------------------------------------------------------------- Personalized feed
elif page == "Personalized feed":
    st.title("Personalized feed")
    st.caption("Fresh Guardian articles ranked for each user by the cold-start MLlib model "
               "(click probability × recency, max 4 per category).")
    profiles = pq("user_profiles")
    recs = pq("recommendations_fresh")
    interest = pq("user_interest")
    if profiles.empty:
        st.warning("No user profiles exported.")
        st.stop()
    uid = st.selectbox("User", profiles["user_id"].tolist(),
                       help="The most active users (by clicks) are listed first.")
    p = profiles[profiles["user_id"] == uid].iloc[0]
    c = st.columns(5)
    c[0].metric("Clicks", int(p["clicks"]))
    c[1].metric("Impressions", int(p["impressions"]))
    c[2].metric("User CTR", f"{p['ctr']:.1%}")
    c[3].metric("History length", int(p["history_length"]) if pd.notna(p["history_length"]) else 0)
    c[4].metric("Segment", f"{int(p['segment'])}" if pd.notna(p.get("segment")) else "—")
    st.caption(f"Segment profile: **{p.get('segment_name', '—')}** · Top interests: **{p['top_3_categories']}**")
    l, r = st.columns([2, 3])
    ui = interest[interest["user_id"] == uid].sort_values("share")
    fig = px.bar(ui, x="share", y="category", orientation="h", title="Interest profile (share of click history)",
                 color_discrete_sequence=[BLUE], labels={"share": "", "category": ""})
    fig.update_xaxes(tickformat=".0%")
    l.plotly_chart(style(fig, 420), use_container_width=True)
    with r:
        st.markdown("#### Recommended now")
        if recs.empty:
            st.info("No Guardian articles were ingested in this run, so there is no fresh feed.")
        else:
            feed = recs[recs["user_id"] == uid].sort_values("rec_rank")
            st.dataframe(feed[["rec_rank", "title", "category", "age_hours", "p_click", "url"]],
                         hide_index=True, use_container_width=True, height=400,
                         column_config={"rec_rank": "#", "url": st.column_config.LinkColumn("Link", display_text="open"),
                                        "age_hours": st.column_config.NumberColumn("Age (h)", format="%.1f"),
                                        "p_click": st.column_config.NumberColumn("P(click)", format="%.3f")})

# ---------------------------------------------------------------- Model results
elif page == "Model results":
    st.title("Model results")
    if not eval_m:
        st.warning("No evaluation metrics exported.")
        st.stop()
    st.caption("Evaluated on the MIND dev split - a later day the model never saw in training. "
               "Metrics are the official MIND ranking metrics, averaged per impression.")
    rows = []
    for name, label in [("model", "MLlib click model"), ("baseline_most_shown", "Baseline: most-shown first"),
                        ("baseline_random", "Baseline: random order")]:
        for k in ["AUC", "MRR", "nDCG@5", "nDCG@10", "HitRate@5"]:
            rows.append({"Ranker": label, "Metric": k, "Value": eval_m[name][k]})
    mdf = pd.DataFrame(rows)
    fig = px.bar(mdf, x="Metric", y="Value", color="Ranker", barmode="group", text_auto=".3f",
                 color_discrete_sequence=[BLUE, ORANGE, AQUA], title="Model vs baselines")
    fig.update_traces(textangle=0, textposition="outside", textfont_size=11)
    st.plotly_chart(style(fig, 420), use_container_width=True)
    st.dataframe(mdf.pivot(index="Ranker", columns="Metric", values="Value").round(4), use_container_width=True)

    l, r = st.columns(2)
    if train_m:
        cm = pd.DataFrame([{"Model": k.replace("_", " ").title(), "ROC-AUC": v["dev_auc"], "PR-AUC": v["dev_auc_pr"]}
                           for k, v in train_m["click_models"].items()])
        cm.loc[len(cm)] = ["Cold-start content model", train_m["cold_start_model_dev_auc"], None]
        l.markdown(f"#### Classifier comparison\nBest model selected: **{train_m['best_click_model']}**")
        l.dataframe(cm.round(4), hide_index=True, use_container_width=True)
        bf = eval_m["model"].get("best_f1_threshold")
        if bf:
            l.markdown(f"At threshold **{bf['threshold']:.2f}**: precision {bf['precision']:.3f}, "
                       f"recall {bf['recall']:.3f}, F1 {bf['f1']:.3f}")
        if train_m.get("gbt_feature_importance"):
            fi = pd.DataFrame(train_m["gbt_feature_importance"], columns=["feature", "importance"]).sort_values("importance")
            title = "GBT feature importance"
        else:
            fi = pd.DataFrame(train_m["logreg_coefficients"], columns=["feature", "importance"])
            fi["importance"] = fi["importance"].abs()
            fi = fi.sort_values("importance")
            title = "Logistic-regression |coefficient|"
        r.plotly_chart(style(px.bar(fi, x="importance", y="feature", orientation="h", title=title,
                                    color_discrete_sequence=[BLUE], labels={"feature": "", "importance": ""}), 420),
                       use_container_width=True)

    st.subheader("See the ranking for one real impression")
    ps = pq("prediction_samples")
    if not ps.empty:
        imp = st.selectbox("Impression", sorted(ps["impression_key"].unique()))
        one = ps[ps["impression_key"] == imp].sort_values("rank_in_impression")
        st.caption(f"User {one['user_id'].iloc[0]} was shown {len(one)} articles and clicked "
                   f"{int(one['label'].sum())} (✅). The model's order is below.")
        one = one.assign(clicked=one["label"].map({1: "✅", 0: ""}))
        st.dataframe(one[["rank_in_impression", "clicked", "title", "category", "score"]], hide_index=True,
                     use_container_width=True,
                     column_config={"rank_in_impression": "Model rank",
                                    "score": st.column_config.NumberColumn("P(click)", format="%.3f")})

# ---------------------------------------------------------------- Segments
elif page == "User segments":
    st.title("User segments (K-Means)")
    seg = pq("segments")
    if train_m:
        st.caption(f"k = {train_m['kmeans_best_k']} chosen by silhouette score "
                   f"({max(train_m['kmeans_silhouette_by_k'].values()):.3f}).")
    seg["label"] = seg["segment"].astype(str) + ": " + seg["segment_name"]
    st.plotly_chart(style(px.bar(seg.sort_values("count"), x="count", y="label", orientation="h",
                                 title="Users per segment", color_discrete_sequence=[BLUE],
                                 labels={"label": "", "count": "Users"})), use_container_width=True)
    sp = pq("segment_profiles")
    fig = px.density_heatmap(sp, x="category", y=sp["segment"].astype(str), z="share", histfunc="avg",
                             color_continuous_scale=SEQ, title="Average interest share per segment",
                             labels={"y": "Segment", "category": ""})
    fig.update_coloraxes(colorbar_title="Share", colorbar_tickformat=".0%")
    st.plotly_chart(style(fig, 420), use_container_width=True)

# ---------------------------------------------------------------- DQ & pipeline
else:
    st.title("Data quality & pipeline")
    if valid:
        status = "✅ all checks passed" if valid["all_passed"] else "❌ some checks failed"
        st.markdown(f"**End-to-end validation:** {valid['passed']}/{valid['total']} checks — {status} "
                    f"(run {valid['validated_at_utc'][:19]} UTC)")
        st.dataframe(pd.DataFrame(valid["checks"]), hide_index=True, use_container_width=True)
    st.subheader("Silver-layer cleaning rules (rows before → after)")
    st.dataframe(pq("dq_report"), hide_index=True, use_container_width=True)
    man = kpis.get("last_ingestion_manifest")
    if man:
        st.subheader("Last ingestion manifest (lineage)")
        st.dataframe(pd.DataFrame(man.get("mind", [])), hide_index=True, use_container_width=True)
        for g in man.get("guardian", []):
            st.markdown(f"Guardian API run `{g['run_id']}`: **{g['articles']}** articles in "
                        f"{len(g['pages'])} pages since {g['from_date']}")
    fa = pq("fresh_articles")
    if not fa.empty:
        st.subheader("Fresh Guardian articles in the catalogue")
        st.dataframe(fa.sort_values("published_ts", ascending=False), hide_index=True, use_container_width=True)
