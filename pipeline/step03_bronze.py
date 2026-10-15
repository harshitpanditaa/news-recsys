"""STEP 3 - Bronze layer (raw / near-raw data + lineage metadata).

No cleaning happens here. Every column is kept as a STRING exactly as in the
source, and four metadata columns are added to every row:
    _source        which system the row came from (MIND-small / Guardian API)
    _source_file   the exact landing file the row was read from
    _ingest_ts     when the row entered Bronze (UTC)
    _batch_id      id of this load, so a bad batch can be traced or replayed
Format: Parquet, partitioned by split (MIND) or ingest date (Guardian).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import BRONZE, LANDING, banner, get_spark, now_utc  # noqa: E402

from pyspark.sql import functions as F  # noqa: E402
from pyspark.sql.types import StringType, StructField, StructType  # noqa: E402

NEWS_COLS = ["news_id", "category", "subcategory", "title", "abstract", "url",
             "title_entities", "abstract_entities"]
BEHAVIOR_COLS = ["impression_id", "user_id", "time", "history", "impressions"]


def tsv_schema(cols):
    return StructType([StructField(c, StringType(), True) for c in cols])


def add_metadata(df, source, batch_id):
    return (df.withColumn("_source", F.lit(source))
              .withColumn("_source_file", F.input_file_name())
              .withColumn("_ingest_ts", F.current_timestamp())
              .withColumn("_batch_id", F.lit(batch_id)))


def run(spark=None):
    spark = spark or get_spark("Step3-Bronze")
    banner("STEP 3: BRONZE - raw data + source/ingestion metadata (Parquet)")
    batch_id = now_utc().strftime("%Y%m%dT%H%M%SZ")

    # ---- MIND (TSV files, no header) -------------------------------------
    for name, cols in (("news", NEWS_COLS), ("behaviors", BEHAVIOR_COLS)):
        df = (spark.read
              .option("sep", "\t").option("header", "false")
              .option("quote", "")          # titles contain quotes: read literally
              .option("mode", "PERMISSIVE")
              .schema(tsv_schema(cols))
              .csv(os.path.join(LANDING, "mind", "*", f"{name}.tsv")))
        df = add_metadata(df, "MIND-small", batch_id)
        df = df.withColumn("split", F.regexp_extract("_source_file", r"/mind/([a-z]+)/", 1))
        out = os.path.join(BRONZE, f"mind_{name}")
        df.write.mode("overwrite").partitionBy("split").parquet(out)
        written = spark.read.parquet(out)
        print(f"\nbronze/mind_{name}: {written.count():,} rows")
        written.groupBy("split").count().orderBy("split").show()

    # ---- Guardian API (raw JSON pages) -----------------------------------
    g_path = os.path.join(LANDING, "guardian", "run=*", "*.json")
    try:
        raw = (spark.read.option("multiLine", "true").json(g_path)
               .withColumn("_source_file", F.input_file_name()))
    except Exception:
        raw = None
    if raw is not None and "response" in raw.columns:
        articles = (raw.select("_source_file", F.explode("response.results").alias("a"))
                       .select("_source_file", "a.*"))
        # keep the API payload as-is; nested 'fields' struct flattened to raw strings
        fields = [F.col(f"fields.{c}").cast("string").alias(f"field_{c}")
                  for c in articles.select("fields.*").columns]
        articles = articles.select(
            *[F.col(c).cast("string") for c in articles.columns if c != "fields"], *fields)
        articles = (articles.withColumn("_source", F.lit("Guardian Open Platform API"))
                            .withColumn("_ingest_ts", F.current_timestamp())
                            .withColumn("_batch_id", F.lit(batch_id))
                            # one partition per API run -> re-running is idempotent
                            .withColumn("_api_run", F.regexp_extract("_source_file", r"run=([0-9TZ]+)", 1)))
        out = os.path.join(BRONZE, "guardian_articles")
        articles.write.mode("overwrite").partitionBy("_api_run").parquet(out)
        print(f"bronze/guardian_articles: {spark.read.parquet(out).count():,} rows")
    else:
        print("No Guardian landing files found - skipping Guardian bronze table")

    print("\nSample Bronze rows (note metadata columns):")
    spark.read.parquet(os.path.join(BRONZE, "mind_news")).select(
        "news_id", "category", "title", "_source", "_source_file", "_ingest_ts", "_batch_id", "split"
    ).show(3, truncate=40)
    spark.read.parquet(os.path.join(BRONZE, "mind_behaviors")).printSchema()


if __name__ == "__main__":
    run()
