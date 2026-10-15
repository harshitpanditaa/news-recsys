"""STEP 5 - PySpark processing.

Demonstrates (and saves the results of) every PySpark technique the course asks for:
  filtering, joins, aggregations, transformations, Spark SQL, window functions.

Outputs (Parquet, silver/processed/*) are reused by the Gold layer in Step 6:
  impressions_enriched    impressions JOIN article catalogue
  user_category_affinity  share of each user's history per category (+ rank window)
  news_exposure           how often each article was shown / clicked per split
  history_popularity      how often each article appears in users' click histories
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SILVER, banner, get_spark  # noqa: E402

from pyspark.sql import Window  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402

PROC = os.path.join(SILVER, "processed")


def run(spark=None):
    spark = spark or get_spark("Step5-Processing")
    banner("STEP 5: PYSPARK PROCESSING - filter, join, aggregate, transform, SQL, windows")

    news = spark.read.parquet(os.path.join(SILVER, "news"))
    imps = spark.read.parquet(os.path.join(SILVER, "impressions"))
    hist = spark.read.parquet(os.path.join(SILVER, "user_history"))
    beh = spark.read.parquet(os.path.join(SILVER, "behaviors"))

    # 1) JOIN + TRANSFORMATION: attach article attributes to each impression
    print("\n[1] JOIN impressions x news (broadcast join, news is the small side)")
    enriched = (imps.join(F.broadcast(news.select("news_id", "category", "subcategory",
                                                  "title_len", "abstract_len")),
                          on="news_id", how="left")
                    .withColumn("category", F.coalesce("category", F.lit("unknown")))
                    .withColumn("day_of_week", F.date_format("event_ts", "E"))
                    .withColumn("is_weekend", F.dayofweek("event_ts").isin(1, 7).cast("int")))
    enriched.write.mode("overwrite").partitionBy("split").parquet(os.path.join(PROC, "impressions_enriched"))
    enriched = spark.read.parquet(os.path.join(PROC, "impressions_enriched"))
    enriched.select("impression_key", "user_id", "news_id", "category", "clicked",
                    "event_ts", "day_of_week").show(5, truncate=False)

    # 2) FILTERING
    print("[2] FILTER: clicked impressions only, and 'heavy readers' (history >= 50)")
    clicks = enriched.filter(F.col("clicked") == 1)
    print(f"    total impressions: {enriched.count():,} | clicks: {clicks.count():,}")
    heavy = beh.filter(F.col("hist_len") >= 50).select("user_id").distinct()
    print(f"    heavy-reader users: {heavy.count():,} of {beh.select('user_id').distinct().count():,}")

    # 3) AGGREGATIONS
    print("[3] AGGREGATE: impressions, clicks and CTR per category")
    cat_stats = (enriched.groupBy("category")
                 .agg(F.count("*").alias("impressions"), F.sum("clicked").alias("clicks"),
                      F.countDistinct("user_id").alias("users"))
                 .withColumn("ctr", F.round(F.col("clicks") / F.col("impressions"), 4))
                 .orderBy(F.desc("clicks")))
    cat_stats.show(20, truncate=False)

    # 4) SPARK SQL
    print("[4] SPARK SQL: click-through rate by hour of day")
    enriched.createOrReplaceTempView("impressions")
    spark.sql("""
        SELECT event_hour,
               COUNT(*)                         AS impressions,
               SUM(clicked)                     AS clicks,
               ROUND(SUM(clicked) / COUNT(*), 4) AS ctr
        FROM impressions
        GROUP BY event_hour
        ORDER BY event_hour
    """).show(24)

    print("[4b] SPARK SQL: top 10 most-clicked articles with titles (JOIN in SQL)")
    news.createOrReplaceTempView("news")
    spark.sql("""
        SELECT i.news_id, n.category, n.title, SUM(i.clicked) AS clicks, COUNT(*) AS shown
        FROM impressions i JOIN news n ON i.news_id = n.news_id
        GROUP BY i.news_id, n.category, n.title
        ORDER BY clicks DESC
        LIMIT 10
    """).show(truncate=60)

    # 5) WINDOW FUNCTIONS
    print("[5a] WINDOW: per-user category affinity + rank of each category for that user")
    user_cat = (hist.join(F.broadcast(news.select("news_id", "category")), "news_id", "left")
                    .fillna({"category": "unknown"})
                    .groupBy("split", "user_id", "category").agg(F.count("*").alias("hist_clicks")))
    w_user = Window.partitionBy("split", "user_id")
    user_cat = (user_cat
                .withColumn("user_total", F.sum("hist_clicks").over(w_user))
                .withColumn("affinity", F.col("hist_clicks") / F.col("user_total"))
                .withColumn("cat_rank", F.dense_rank().over(w_user.orderBy(F.desc("hist_clicks")))))
    user_cat.write.mode("overwrite").partitionBy("split").parquet(os.path.join(PROC, "user_category_affinity"))
    spark.read.parquet(os.path.join(PROC, "user_category_affinity")).orderBy("user_id", "cat_rank").show(8)

    print("[5b] WINDOW: top-3 trending articles per day (row_number over clicks)")
    daily = clicks.groupBy("event_date", "news_id", "category").agg(F.count("*").alias("clicks"))
    w_day = Window.partitionBy("event_date").orderBy(F.desc("clicks"))
    daily.withColumn("rank", F.row_number().over(w_day)).filter("rank <= 3") \
         .orderBy("event_date", "rank").show(15)

    print("[5c] WINDOW: 3-hour rolling average of clicks per category (rangeBetween)")
    hourly = (enriched.withColumn("hour_ts", F.date_trunc("hour", "event_ts"))
              .groupBy("category", "hour_ts").agg(F.sum("clicked").alias("clicks")))
    w_roll = (Window.partitionBy("category").orderBy(F.col("hour_ts").cast("long"))
              .rangeBetween(-2 * 3600, 0))
    hourly.withColumn("rolling_3h_clicks", F.round(F.avg("clicks").over(w_roll), 2)) \
          .filter(F.col("category") == "news").orderBy("hour_ts").show(8)

    print("[5d] WINDOW: minutes between a user's consecutive impressions (lag)")
    w_sess = Window.partitionBy("user_id").orderBy("event_ts")
    gaps = (beh.select("user_id", "event_ts")
            .withColumn("prev_ts", F.lag("event_ts").over(w_sess))
            .withColumn("gap_min", (F.col("event_ts").cast("long") - F.col("prev_ts").cast("long")) / 60))
    gaps.filter(F.col("gap_min").isNotNull()).select(
        F.round(F.expr("percentile_approx(gap_min, 0.5)"), 1).alias("median_gap_min"),
        F.round(F.avg("gap_min"), 1).alias("mean_gap_min")).show()

    # Reusable aggregates for Gold / ML (no click labels used -> no leakage)
    exposure = (imps.groupBy("split", "news_id")
                .agg(F.count("*").alias("times_shown"), F.sum("clicked").alias("times_clicked")))
    exposure.write.mode("overwrite").partitionBy("split").parquet(os.path.join(PROC, "news_exposure"))
    hist_pop = hist.groupBy("split", "news_id").agg(F.countDistinct("user_id").alias("hist_users"))
    hist_pop.write.mode("overwrite").partitionBy("split").parquet(os.path.join(PROC, "history_popularity"))
    print(f"Saved processed tables under {PROC}")


if __name__ == "__main__":
    run()
