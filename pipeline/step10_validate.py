"""STEP 10 (validation part) - automated end-to-end checks of the whole pipeline.

Fails (non-zero exit -> Airflow task turns red) if any layer is missing or
empty, if keys are duplicated/null in Silver, if row counts don't reconcile
between layers, or if the model is no better than random.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import BRONZE, GOLD, MODELS, OUTPUT, SILVER, banner, get_spark, now_utc, write_json  # noqa: E402

from pyspark.sql import functions as F  # noqa: E402


def run(spark=None):
    spark = spark or get_spark("Step10-Validate")
    banner("STEP 10: END-TO-END PIPELINE VALIDATION")
    checks = []

    def check(name, ok, detail=""):
        checks.append({"check": name, "passed": bool(ok), "detail": str(detail)})
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")

    def rows(path):
        return spark.read.parquet(path).count() if os.path.exists(path) else 0

    for layer, tables in {
        BRONZE: ["mind_news", "mind_behaviors"],
        SILVER: ["news", "behaviors", "impressions", "user_history", "articles", "_dq_report"],
        GOLD: ["daily_kpis", "category_kpis", "hourly_trends", "news_popularity", "user_profile",
               "user_interest_matrix", "ml_features", "user_segments"],
        OUTPUT: ["predictions_dev", "recommendations_dev_top10"],
    }.items():
        for t in tables:
            n = rows(os.path.join(layer, t))
            check(f"{os.path.basename(layer)}/{t} exists and is non-empty", n > 0, f"{n:,} rows")

    bronze_meta = spark.read.parquet(os.path.join(BRONZE, "mind_news")).columns
    check("bronze keeps lineage metadata", all(c in bronze_meta for c in
          ["_source", "_source_file", "_ingest_ts", "_batch_id"]), bronze_meta[-5:])

    news = spark.read.parquet(os.path.join(SILVER, "news"))
    check("silver/news: news_id unique", news.count() == news.select("news_id").distinct().count())
    check("silver/news: no null titles", news.filter(F.col("title").isNull()).count() == 0)
    imps = spark.read.parquet(os.path.join(SILVER, "impressions"))
    check("silver/impressions: clicked is 0/1 only",
          imps.filter(~F.col("clicked").isin(0, 1)).count() == 0)
    check("silver/impressions: event_ts typed as timestamp",
          dict(imps.dtypes)["event_ts"] == "timestamp", dict(imps.dtypes)["event_ts"])

    n_feat = spark.read.parquet(os.path.join(GOLD, "ml_features")).count()
    check("gold/ml_features reconciles with silver/impressions (1 row per impression)",
          n_feat == imps.count(), f"{n_feat:,} vs {imps.count():,}")
    n_dev = imps.filter("split = 'dev'").count()
    n_pred = rows(os.path.join(OUTPUT, "predictions_dev"))
    check("every dev impression got a prediction", n_pred == n_dev, f"{n_pred:,} vs {n_dev:,}")

    for m in ("click_model", "cold_start_model", "user_kmeans"):
        check(f"model saved: {m}", os.path.isdir(os.path.join(MODELS, m)))

    ev = json.load(open(os.path.join(OUTPUT, "evaluation_metrics.json")))
    check("model beats random baseline on per-impression AUC",
          ev["model"]["AUC"] > ev["baseline_random"]["AUC"] + 0.02,
          f"{ev['model']['AUC']:.4f} vs {ev['baseline_random']['AUC']:.4f}")
    check("model beats random baseline on nDCG@10",
          ev["model"]["nDCG@10"] > ev["baseline_random"]["nDCG@10"],
          f"{ev['model']['nDCG@10']:.4f} vs {ev['baseline_random']['nDCG@10']:.4f}")
    print(f"  [INFO] vs most-shown baseline: nDCG@10 {ev['model']['nDCG@10']:.4f} "
          f"vs {ev['baseline_most_shown']['nDCG@10']:.4f}")

    passed = sum(c["passed"] for c in checks)
    report = {"validated_at_utc": now_utc().isoformat(), "passed": passed, "total": len(checks),
              "all_passed": passed == len(checks), "checks": checks}
    write_json(report, os.path.join(OUTPUT, "validation_report.json"))
    print(f"\n{passed}/{len(checks)} checks passed -> {OUTPUT}/validation_report.json")
    if passed != len(checks):
        sys.exit(1)


if __name__ == "__main__":
    run()
