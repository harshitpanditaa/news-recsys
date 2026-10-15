"""STEP 8 - Generate predictions & validate results.

1. Score every dev-split impression with the best click model and save the
   predictions (Parquet) with each article's rank inside its impression.
2. Evaluate as a RECOMMENDER, using the official MIND metrics, computed with
   Spark window functions per impression:
     AUC (mean per impression), MRR, nDCG@5, nDCG@10, plus Hit-Rate@5
   and compare against two baselines: random order and "most-shown first".
3. Produce personalised top-10 feeds of FRESH Guardian articles for the most
   active users (cold-start model x recency decay, max 4 per category for
   diversity) and save them (Parquet).
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    GOLD, MODELS, OUTPUT, RANDOM_SEED, SILVER, TOP_USERS_FOR_FRESH_RECS,
    banner, get_spark, now_utc, write_json,
)

from pyspark.ml import PipelineModel  # noqa: E402
from pyspark.ml.evaluation import BinaryClassificationEvaluator  # noqa: E402
from pyspark.ml.functions import vector_to_array  # noqa: E402
from pyspark.sql import Window  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402

RECENCY_HALF_LIFE_HOURS = 24.0
MAX_PER_CATEGORY = 4


def ranking_metrics(df, score_col):
    """MIND-style metrics. df: impression_key, label, <score_col>."""
    w_desc = Window.partitionBy("impression_key").orderBy(F.desc(score_col), "news_id")
    w_asc = Window.partitionBy("impression_key").orderBy(F.asc(score_col), "news_id")
    w_ideal = Window.partitionBy("impression_key").orderBy(F.desc("label"), "news_id")
    r = (df.withColumn("rank", F.row_number().over(w_desc))
           .withColumn("rank_asc", F.row_number().over(w_asc))
           .withColumn("ideal_rank", F.row_number().over(w_ideal))
           .withColumn("disc", F.col("label") / F.log2(F.col("rank") + 1))
           .withColumn("ideal_disc", F.col("label") / F.log2(F.col("ideal_rank") + 1)))
    per = (r.groupBy("impression_key").agg(
        F.sum("label").alias("P"), F.count("*").alias("n"),
        F.sum(F.col("label") * F.col("rank_asc")).alias("pos_rank_sum"),
        (F.sum(F.col("label") / F.col("rank")) / F.sum("label")).alias("mrr"),
        F.sum(F.when(F.col("rank") <= 5, F.col("disc")).otherwise(0)).alias("dcg5"),
        F.sum(F.when(F.col("ideal_rank") <= 5, F.col("ideal_disc")).otherwise(0)).alias("idcg5"),
        F.sum(F.when(F.col("rank") <= 10, F.col("disc")).otherwise(0)).alias("dcg10"),
        F.sum(F.when(F.col("ideal_rank") <= 10, F.col("ideal_disc")).otherwise(0)).alias("idcg10"),
        F.max(F.when(F.col("rank") <= 5, F.col("label")).otherwise(0)).alias("hit5"))
        .withColumn("N", F.col("n") - F.col("P"))
        .filter((F.col("P") > 0) & (F.col("N") > 0))
        .withColumn("auc", (F.col("pos_rank_sum") - F.col("P") * (F.col("P") + 1) / 2)
                    / (F.col("P") * F.col("N"))))
    m = per.agg(F.avg("auc").alias("AUC"), F.avg("mrr").alias("MRR"),
                F.avg(F.col("dcg5") / F.col("idcg5")).alias("nDCG@5"),
                F.avg(F.col("dcg10") / F.col("idcg10")).alias("nDCG@10"),
                F.avg("hit5").alias("HitRate@5"), F.count("*").alias("impressions_evaluated")).first()
    return {k: (float(v) if k != "impressions_evaluated" else int(v)) for k, v in m.asDict().items()}


def run(spark=None):
    spark = spark or get_spark("Step8-Predict")
    banner("STEP 8: PREDICTIONS + EVALUATION + FRESH RECOMMENDATIONS")

    feats = spark.read.parquet(os.path.join(GOLD, "ml_features"))
    dev = feats.filter(F.col("split") == "dev")
    model = PipelineModel.load(os.path.join(MODELS, "click_model"))

    pred = (model.transform(dev)
            .withColumn("score", vector_to_array("probability")[1])
            .withColumn("rank_in_impression", F.row_number().over(
                Window.partitionBy("impression_key").orderBy(F.desc("score"), "news_id")))
            .select("impression_key", "user_id", "news_id", "category", "label", "score",
                    "rank_in_impression", "rawPrediction", "log_times_shown"))
    pred.drop("rawPrediction").write.mode("overwrite").parquet(os.path.join(OUTPUT, "predictions_dev"))
    print(f"Saved {spark.read.parquet(os.path.join(OUTPUT, 'predictions_dev')).count():,} predictions "
          f"to output/predictions_dev")

    global_auc = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC").evaluate(pred)
    scored = spark.read.parquet(os.path.join(OUTPUT, "predictions_dev")) \
                  .withColumn("random_score", F.rand(RANDOM_SEED)).cache()
    print("Computing ranking metrics (model vs baselines) ...")
    metrics = {
        "model": ranking_metrics(scored, "score"),
        "baseline_most_shown": ranking_metrics(scored, "log_times_shown"),
        "baseline_random": ranking_metrics(scored, "random_score"),
    }
    metrics["model"]["global_ROC_AUC"] = global_auc
    print(f"\n{'':<22}{'AUC':>8}{'MRR':>8}{'nDCG@5':>8}{'nDCG@10':>9}{'Hit@5':>8}")
    for name, m in metrics.items():
        print(f"{name:<22}{m['AUC']:>8.4f}{m['MRR']:>8.4f}{m['nDCG@5']:>8.4f}{m['nDCG@10']:>9.4f}{m['HitRate@5']:>8.4f}")

    # threshold-based classification metrics at the best-F1 threshold on dev
    pdf = scored.select("label", "score")
    best = None
    for t in [x / 100 for x in range(5, 95, 5)]:
        c = pdf.select(
            F.sum(((F.col("score") >= t) & (F.col("label") == 1)).cast("int")).alias("tp"),
            F.sum(((F.col("score") >= t) & (F.col("label") == 0)).cast("int")).alias("fp"),
            F.sum(((F.col("score") < t) & (F.col("label") == 1)).cast("int")).alias("fn")).first()
        p = c.tp / (c.tp + c.fp) if (c.tp + c.fp) else 0.0
        r = c.tp / (c.tp + c.fn) if (c.tp + c.fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        if best is None or f1 > best["f1"]:
            best = {"threshold": t, "precision": p, "recall": r, "f1": f1,
                    "tp": c.tp, "fp": c.fp, "fn": c.fn}
    metrics["model"]["best_f1_threshold"] = best
    print(f"\nBest-F1 threshold {best['threshold']:.2f}: precision={best['precision']:.3f} "
          f"recall={best['recall']:.3f} F1={best['f1']:.3f}")

    top10 = scored.filter("rank_in_impression <= 10").drop("random_score")
    top10.write.mode("overwrite").parquet(os.path.join(OUTPUT, "recommendations_dev_top10"))

    # ---------------- fresh Guardian articles ----------------
    fresh_path = os.path.join(GOLD, "fresh_articles")
    if os.path.exists(fresh_path):
        fresh = spark.read.parquet(fresh_path)
        n_fresh = fresh.count()
        users = (spark.read.parquet(os.path.join(GOLD, "user_profile"))
                 .orderBy(F.desc("clicks"), "user_id").limit(TOP_USERS_FOR_FRESH_RECS)
                 .select("user_id", "history_length"))
        aff_raw = spark.read.parquet(os.path.join(SILVER, "processed", "user_category_affinity"))
        aff = (aff_raw.groupBy("user_id", "category").agg(F.sum("hist_clicks").alias("c"))
               .withColumn("user_cat_affinity", F.col("c") / F.sum("c").over(Window.partitionBy("user_id")))
               .withColumn("user_cat_rank", F.dense_rank().over(
                   Window.partitionBy("user_id").orderBy(F.desc("c")))).drop("c"))
        now = now_utc()
        cand = (users.crossJoin(F.broadcast(fresh.select(
                    "news_id", "category", "subcategory", "title", "url", "section_name",
                    "published_ts", "age_hours", "category_ctr", "title_len", "abstract_len")))
                .join(aff, ["user_id", "category"], "left")
                .fillna({"user_cat_affinity": 0.0, "user_cat_rank": 99})
                .withColumn("subcat_affinity", F.lit(0.0))
                .withColumn("is_top_category", (F.col("user_cat_rank") == 1).cast("int"))
                .withColumn("log_hist_len", F.log1p("history_length"))
                .withColumn("cold_user", (F.col("history_length") == 0).cast("int"))
                .withColumn("event_hour", F.lit(now.hour))
                .withColumn("is_weekend", F.lit(int(now.weekday() >= 5))))
        cold = PipelineModel.load(os.path.join(MODELS, "cold_start_model"))
        decay = math.log(2) / RECENCY_HALF_LIFE_HOURS
        recs = (cold.transform(cand)
                .withColumn("p_click", vector_to_array("probability")[1])
                .withColumn("recency_weight", F.exp(-decay * F.greatest(F.col("age_hours"), F.lit(0.0))))
                .withColumn("final_score", F.col("p_click") * (0.5 + 0.5 * F.col("recency_weight")))
                # diversity rule: at most MAX_PER_CATEGORY articles per category in one feed
                .withColumn("cat_pos", F.row_number().over(
                    Window.partitionBy("user_id", "category").orderBy(F.desc("final_score"), "news_id")))
                .filter(F.col("cat_pos") <= MAX_PER_CATEGORY)
                .withColumn("rec_rank", F.row_number().over(
                    Window.partitionBy("user_id").orderBy(F.desc("final_score"), "news_id")))
                .filter("rec_rank <= 10")
                .select("user_id", "rec_rank", "news_id", "title", "category", "section_name", "url",
                        "published_ts", F.round("age_hours", 1).alias("age_hours"),
                        F.round("user_cat_affinity", 3).alias("user_cat_affinity"),
                        F.round("p_click", 4).alias("p_click"), F.round("final_score", 4).alias("final_score")))
        recs.write.mode("overwrite").parquet(os.path.join(OUTPUT, "recommendations_fresh"))
        saved = spark.read.parquet(os.path.join(OUTPUT, "recommendations_fresh"))
        print(f"\nFresh-article feeds: {saved.select('user_id').distinct().count():,} users x top-10 "
              f"from {n_fresh:,} Guardian articles")
        one = saved.select("user_id").orderBy("user_id").first()[0]
        print(f"Example personalised feed for {one}:")
        saved.filter(F.col("user_id") == one).orderBy("rec_rank").select(
            "rec_rank", "category", "title", "user_cat_affinity", "p_click", "age_hours").show(truncate=60)
        metrics["fresh_recommendations"] = {"users": saved.select("user_id").distinct().count(),
                                            "fresh_articles": n_fresh,
                                            "recency_half_life_hours": RECENCY_HALF_LIFE_HOURS}

    write_json(metrics, os.path.join(OUTPUT, "evaluation_metrics.json"))
    print(f"\nSaved evaluation metrics to {OUTPUT}/evaluation_metrics.json")


if __name__ == "__main__":
    run()
