"""STEP 9 (data part) - export the project's real outputs for the Streamlit dashboard.

The dashboard never invents numbers: it only reads the files written here,
which are small extracts of the Gold layer and the Step 7/8 outputs.
"""
import glob
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    DASHBOARD_DATA, GOLD, LANDING, OUTPUT, SILVER, TOP_USERS_FOR_FRESH_RECS, banner, get_spark,
    now_utc, write_json,
)

from pyspark.sql import functions as F  # noqa: E402


def to_parquet(df, name):
    pdf = df.toPandas()
    pdf.to_parquet(os.path.join(DASHBOARD_DATA, f"{name}.parquet"), index=False)
    print(f"  dashboard/data/{name}.parquet  ({len(pdf):,} rows)")


def run(spark=None):
    spark = spark or get_spark("Step9-Export")
    banner("STEP 9: EXPORT PROJECT OUTPUTS FOR THE DASHBOARD")
    os.makedirs(DASHBOARD_DATA, exist_ok=True)
    g = lambda n: spark.read.parquet(os.path.join(GOLD, n))  # noqa: E731

    to_parquet(g("daily_kpis").orderBy("event_date"), "daily_kpis")
    to_parquet(g("category_kpis"), "category_kpis")
    to_parquet(g("hourly_trends"), "hourly_trends")
    to_parquet(g("news_popularity").filter("shown > 0").orderBy(F.desc("clicked")).limit(500)
               .select("news_id", "title", "category", "subcategory", "shown", "clicked", "ctr",
                       "smoothed_ctr", "history_readers", "url"), "top_news")

    users = (g("user_profile").orderBy(F.desc("clicks"), "user_id").limit(TOP_USERS_FOR_FRESH_RECS))
    seg = g("user_segments")
    to_parquet(users.join(seg, "user_id", "left"), "user_profiles")
    interest = g("user_interest_matrix").join(users.select("user_id"), "user_id")
    cats = [c for c in interest.columns if c != "user_id"]
    stack = f"stack({len(cats)}, " + ", ".join(f"'{c}', `{c}`" for c in cats) + ") as (category, share)"
    to_parquet(interest.selectExpr("user_id", stack).filter("share > 0"), "user_interest")
    to_parquet(seg.groupBy("segment", "segment_name").count().orderBy("segment"), "segments")
    seg_profile = (spark.read.parquet(os.path.join(GOLD, "user_interest_matrix")).join(seg.select("user_id", "segment"), "user_id")
                   .groupBy("segment").agg(*[F.avg(c).alias(c) for c in cats]))
    to_parquet(seg_profile.selectExpr("segment", stack), "segment_profiles")

    preds = spark.read.parquet(os.path.join(OUTPUT, "predictions_dev"))
    sample_imps = (preds.groupBy("impression_key", "user_id").agg(F.sum("label").alias("p"), F.count("*").alias("n"))
                   .filter("p > 0 AND n BETWEEN 8 AND 60").orderBy(F.rand(42)).limit(300))
    news = spark.read.parquet(os.path.join(SILVER, "news")).select("news_id", "title")
    to_parquet(preds.join(sample_imps.select("impression_key"), "impression_key")
               .join(news, "news_id", "left"), "prediction_samples")

    fresh_recs = os.path.join(OUTPUT, "recommendations_fresh")
    if os.path.exists(fresh_recs):
        to_parquet(spark.read.parquet(fresh_recs), "recommendations_fresh")
        to_parquet(g("fresh_articles").select("news_id", "title", "category", "section_name",
                                              "published_ts", "age_hours", "url"), "fresh_articles")
    to_parquet(spark.read.parquet(os.path.join(SILVER, "_dq_report")), "dq_report")

    for f in ("training_metrics.json", "evaluation_metrics.json", "validation_report.json"):
        src = os.path.join(OUTPUT, f)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(DASHBOARD_DATA, f))

    totals = spark.read.parquet(os.path.join(SILVER, "impressions")).agg(
        F.countDistinct("user_id").alias("users"), F.count("*").alias("impressions"),
        F.sum("clicked").alias("clicks"), F.countDistinct("impression_key").alias("sessions")).first().asDict()
    totals["articles_mind"] = spark.read.parquet(os.path.join(SILVER, "news")).count()
    fa = os.path.join(SILVER, "guardian_articles")
    totals["articles_guardian"] = spark.read.parquet(fa).count() if os.path.exists(fa) else 0
    totals["ctr"] = totals["clicks"] / totals["impressions"]
    manifests = sorted(glob.glob(os.path.join(LANDING, "_manifests", "*.json")))
    totals["last_ingestion_manifest"] = json.load(open(manifests[-1])) if manifests else None
    totals["exported_at_utc"] = now_utc().isoformat()
    write_json(totals, os.path.join(DASHBOARD_DATA, "kpis.json"))
    size = sum(os.path.getsize(p) for p in glob.glob(os.path.join(DASHBOARD_DATA, "*")))
    print(f"\nDashboard data written to {DASHBOARD_DATA} ({size/1e6:.1f} MB)")


if __name__ == "__main__":
    run()
