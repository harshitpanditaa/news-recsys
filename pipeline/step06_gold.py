"""STEP 6 - Gold layer (business-ready & analysis-ready datasets).

Business / dashboard tables
  gold/daily_kpis          users, impressions, clicks, CTR per day
  gold/category_kpis       volume, CTR and share of clicks per category
  gold/hourly_trends       clicks and CTR per category per hour
  gold/news_popularity     per-article shown/clicked/CTR (+ Bayesian-smoothed CTR)
  gold/user_profile        per-user activity, CTR and top interests
  gold/user_interest_matrix  user x category share of history (input to clustering)
ML tables
  gold/ml_features         one row per (impression, candidate article) with features + label
  gold/category_ctr_train  category click rate learned on the TRAIN split only
  gold/fresh_articles      Guardian articles with mapped category and age in hours

Leakage note: the only label-derived feature (category CTR) is learned from the
train split and applied to dev. Popularity features use exposure counts and click
HISTORIES (clicks that happened before the impression), not the impression labels.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import GOLD, SILVER, banner, get_spark  # noqa: E402

from pyspark.sql import Window  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402

PROC = os.path.join(SILVER, "processed")


def save(df, name, partition=None):
    path = os.path.join(GOLD, name)
    w = df.write.mode("overwrite")
    (w.partitionBy(partition) if partition else w).parquet(path)
    out = df.sparkSession.read.parquet(path)
    print(f"  gold/{name:<22} {out.count():>10,} rows")
    return out


def run(spark=None):
    spark = spark or get_spark("Step6-Gold")
    banner("STEP 6: GOLD - business-ready KPIs, profiles and ML feature tables")

    news = spark.read.parquet(os.path.join(SILVER, "news"))
    enriched = spark.read.parquet(os.path.join(PROC, "impressions_enriched"))
    beh = spark.read.parquet(os.path.join(SILVER, "behaviors"))
    hist = spark.read.parquet(os.path.join(SILVER, "user_history"))
    aff = spark.read.parquet(os.path.join(PROC, "user_category_affinity"))
    exposure = spark.read.parquet(os.path.join(PROC, "news_exposure"))
    hist_pop = spark.read.parquet(os.path.join(PROC, "history_popularity"))

    # ---------------- business tables ----------------
    daily = (enriched.groupBy("event_date")
             .agg(F.countDistinct("user_id").alias("active_users"),
                  F.countDistinct("impression_key").alias("sessions"),
                  F.count("*").alias("impressions"), F.sum("clicked").alias("clicks"))
             .withColumn("ctr", F.col("clicks") / F.col("impressions")))
    save(daily, "daily_kpis")

    total_clicks = enriched.agg(F.sum("clicked")).first()[0]
    cat = (enriched.groupBy("category")
           .agg(F.count("*").alias("impressions"), F.sum("clicked").alias("clicks"),
                F.countDistinct("user_id").alias("users"), F.countDistinct("news_id").alias("articles"))
           .withColumn("ctr", F.col("clicks") / F.col("impressions"))
           .withColumn("click_share", F.col("clicks") / F.lit(total_clicks)))
    save(cat, "category_kpis")

    hourly = (enriched.withColumn("hour_ts", F.date_trunc("hour", "event_ts"))
              .groupBy("hour_ts", "category")
              .agg(F.count("*").alias("impressions"), F.sum("clicked").alias("clicks"))
              .withColumn("ctr", F.col("clicks") / F.col("impressions")))
    save(hourly, "hourly_trends")

    global_ctr = enriched.agg(F.avg("clicked")).first()[0]
    prior_weight = 20.0  # Bayesian smoothing: pull low-volume articles toward the global CTR
    pop = (exposure.groupBy("news_id").agg(F.sum("times_shown").alias("shown"),
                                           F.sum("times_clicked").alias("clicked"))
           .join(hist_pop.groupBy("news_id").agg(F.sum("hist_users").alias("history_readers")),
                 "news_id", "full")
           .fillna(0)
           .join(news.select("news_id", "category", "subcategory", "title", "url"), "news_id", "left")
           .withColumn("ctr", F.when(F.col("shown") > 0, F.col("clicked") / F.col("shown")))
           .withColumn("smoothed_ctr", (F.col("clicked") + prior_weight * global_ctr)
                       / (F.col("shown") + prior_weight))
           .withColumn("popularity_rank", F.dense_rank().over(
               Window.orderBy(F.desc("clicked"), F.desc("history_readers")))))
    save(pop, "news_popularity")

    user_tot = aff.groupBy("user_id", "category").agg(F.sum("hist_clicks").alias("c"))
    w = Window.partitionBy("user_id").orderBy(F.desc("c"), "category")
    top_cats = (user_tot.withColumn("r", F.row_number().over(w)).filter("r <= 3")
                .groupBy("user_id").agg(
                    F.max(F.when(F.col("r") == 1, F.col("category"))).alias("top_category"),
                    F.concat_ws(", ", F.collect_list("category")).alias("top_3_categories")))
    profile = (enriched.groupBy("user_id")
               .agg(F.countDistinct("impression_key").alias("sessions"),
                    F.count("*").alias("impressions"), F.sum("clicked").alias("clicks"),
                    F.min("event_ts").alias("first_seen"), F.max("event_ts").alias("last_seen"),
                    F.countDistinct("event_date").alias("active_days"))
               .withColumn("ctr", F.col("clicks") / F.col("impressions"))
               .join(beh.groupBy("user_id").agg(F.max("hist_len").alias("history_length")), "user_id", "left")
               .join(top_cats, "user_id", "left")
               .fillna({"top_category": "unknown", "top_3_categories": ""}))
    save(profile, "user_profile")

    categories = [r[0] for r in news.select("category").distinct().orderBy("category").collect()]
    matrix = (user_tot.groupBy("user_id").pivot("category", categories).sum("c").fillna(0))
    row_total = sum(F.col(c) for c in categories)
    matrix = matrix.select("user_id", *[(F.col(c) / row_total).alias(c) for c in categories])
    save(matrix, "user_interest_matrix")

    # ---------------- ML feature table ----------------
    cat_ctr = (enriched.filter(F.col("split") == "train").groupBy("category")
               .agg(F.avg("clicked").alias("category_ctr")))
    cat_ctr = save(cat_ctr, "category_ctr_train")

    sub_aff = (hist.join(F.broadcast(news.select("news_id", "subcategory")), "news_id", "left")
               .groupBy("split", "user_id", "subcategory").agg(F.count("*").alias("n"))
               .withColumn("subcat_affinity", F.col("n") / F.sum("n").over(Window.partitionBy("split", "user_id")))
               .drop("n"))

    feats = (enriched
             .join(aff.select("split", "user_id", "category", F.col("affinity").alias("user_cat_affinity"),
                              F.col("cat_rank").alias("user_cat_rank")),
                   ["split", "user_id", "category"], "left")
             .join(sub_aff, ["split", "user_id", "subcategory"], "left")
             .join(exposure.select("split", "news_id", "times_shown"), ["split", "news_id"], "left")
             .join(hist_pop, ["split", "news_id"], "left")
             .join(F.broadcast(cat_ctr), "category", "left")
             .fillna({"user_cat_affinity": 0.0, "subcat_affinity": 0.0, "user_cat_rank": 99,
                      "times_shown": 0, "hist_users": 0, "category_ctr": float(global_ctr),
                      "title_len": 0, "abstract_len": 0})
             .withColumn("log_times_shown", F.log1p("times_shown"))
             .withColumn("log_hist_users", F.log1p("hist_users"))
             .withColumn("log_hist_len", F.log1p("hist_len"))
             .withColumn("is_top_category", (F.col("user_cat_rank") == 1).cast("int"))
             .withColumn("cold_user", (F.col("hist_len") == 0).cast("int"))
             .select("split", "impression_key", "user_id", "news_id", "category",
                     F.col("clicked").alias("label"), "user_cat_affinity", "subcat_affinity",
                     "user_cat_rank", "is_top_category", "log_times_shown", "log_hist_users",
                     "category_ctr", "log_hist_len", "cold_user", "title_len", "abstract_len",
                     "event_hour", "is_weekend"))
    save(feats, "ml_features", partition="split")

    g_path = os.path.join(SILVER, "guardian_articles")
    if os.path.exists(g_path):
        fresh = (spark.read.parquet(g_path)
                 .withColumn("age_hours", (F.unix_timestamp(F.current_timestamp())
                                           - F.unix_timestamp("published_ts")) / 3600.0)
                 .join(F.broadcast(cat_ctr), "category", "left")
                 .fillna({"category_ctr": float(global_ctr)}))
        save(fresh, "fresh_articles")

    print("\nSample gold/user_profile:")
    spark.read.parquet(os.path.join(GOLD, "user_profile")).orderBy(F.desc("clicks")).show(5, truncate=False)
    print("Sample gold/ml_features:")
    spark.read.parquet(os.path.join(GOLD, "ml_features")).show(5)


if __name__ == "__main__":
    run()
