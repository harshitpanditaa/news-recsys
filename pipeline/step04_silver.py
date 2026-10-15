"""STEP 4 - Silver layer (cleaned, validated, correctly typed data).

Every cleaning rule is logged into a data-quality (DQ) report with the row
count before and after the rule, so the effect of each rule is provable.

Rules applied
  Missing values     : null/blank title -> row dropped; null abstract -> "" + flag;
                       null history -> empty list; null category -> 'unknown'
  Duplicates         : duplicate news_id (same article in train+dev), duplicate
                       impressions, duplicate Guardian articles across API runs
  Incorrect types    : impression_id -> int, time string -> timestamp,
                       clicked flag -> int, Guardian wordcount -> int, dates -> timestamp
  Inconsistent values: trimmed/lower-cased categories, HTML stripped from Guardian
                       trail text, Guardian sections mapped to MIND categories
  Invalid records    : malformed news ids (must match N<digits>), unparseable
                       timestamps, impression tokens not in the form N123-0/1,
                       clicked flags other than 0/1
Outputs (Parquet): silver/news, silver/behaviors, silver/impressions,
silver/user_history, silver/guardian_articles, silver/articles (unified catalogue),
silver/_dq_report
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    BRONZE, DEFAULT_CATEGORY, GUARDIAN_TO_MIND, SILVER, banner, get_spark,
)

from pyspark.sql import Window  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402
from pyspark.sql.types import (  # noqa: E402
    ArrayType, DoubleType, StringType, StructField, StructType,
)

ENTITY_SCHEMA = ArrayType(StructType([
    StructField("Label", StringType()), StructField("Type", StringType()),
    StructField("WikidataId", StringType()), StructField("Confidence", DoubleType()),
]))


class DQ:
    """Collects (table, rule, rows_before, rows_after) for the DQ report."""

    def __init__(self):
        self.rows = []

    def step(self, table, rule, df_before, df_after):
        before, after = df_before.count(), df_after.count()
        self.rows.append((table, rule, before, after, before - after))
        print(f"  [{table}] {rule:<55} {before:>10,} -> {after:>10,}  (removed {before - after:,})")
        return df_after


def blank_to_null(c):
    return F.when(F.trim(F.col(c)) == "", None).otherwise(F.trim(F.col(c)))


def clean_news(spark, dq):
    df = spark.read.parquet(os.path.join(BRONZE, "mind_news")).cache()
    for c in ["news_id", "category", "subcategory", "title", "abstract", "url"]:
        df = df.withColumn(c, blank_to_null(c))

    d1 = dq.step("news", "drop rows with missing news_id or title", df,
                 df.filter(F.col("news_id").isNotNull() & F.col("title").isNotNull()))
    d2 = dq.step("news", "drop invalid news_id (must match ^N[0-9]+$)", d1,
                 d1.filter(F.col("news_id").rlike(r"^N[0-9]+$")))
    # same article appears in both train and dev files -> keep the latest copy
    w = Window.partitionBy("news_id").orderBy(F.col("_ingest_ts").desc(), F.col("split").asc())
    d3 = dq.step("news", "remove duplicate news_id (train/dev overlap)", d2,
                 d2.withColumn("_rn", F.row_number().over(w)).filter("_rn = 1").drop("_rn"))

    out = (d3
           .withColumn("category", F.coalesce(F.lower("category"), F.lit("unknown")))
           .withColumn("subcategory", F.coalesce(F.lower("subcategory"), F.lit("unknown")))
           .withColumn("abstract_missing", F.col("abstract").isNull())
           .withColumn("abstract", F.coalesce("abstract", F.lit("")))
           .withColumn("title_len", F.size(F.split("title", r"\s+")))
           .withColumn("abstract_len",
                       F.when(F.col("abstract") == "", 0).otherwise(F.size(F.split("abstract", r"\s+"))))
           .withColumn("title_entities_parsed", F.from_json("title_entities", ENTITY_SCHEMA))
           .withColumn("title_entity_labels",
                       F.coalesce(F.expr("transform(title_entities_parsed, e -> e.Label)"),
                                  F.array().cast("array<string>")))
           .withColumn("n_title_entities", F.size("title_entity_labels"))
           .select("news_id", "category", "subcategory", "title", "abstract", "url",
                   "title_len", "abstract_len", "abstract_missing", "title_entity_labels",
                   "n_title_entities", "_source", "_source_file", "_ingest_ts", "_batch_id"))
    out.write.mode("overwrite").parquet(os.path.join(SILVER, "news"))
    df.unpersist()


def clean_behaviors(spark, dq):
    df = spark.read.parquet(os.path.join(BRONZE, "mind_behaviors")).cache()
    df = (df.withColumn("user_id", blank_to_null("user_id"))
            .withColumn("impression_id_int", F.col("impression_id").cast("int"))
            # MIND time format example: 11/15/2019 8:55:22 AM
            .withColumn("event_ts", F.to_timestamp("time", "M/d/yyyy h:mm:ss a")))

    d1 = dq.step("behaviors", "drop missing user_id / impressions", df,
                 df.filter(F.col("user_id").isNotNull() & F.col("impressions").isNotNull()))
    d2 = dq.step("behaviors", "drop non-integer impression_id (bad type)", d1,
                 d1.filter(F.col("impression_id_int").isNotNull()))
    d3 = dq.step("behaviors", "drop unparseable timestamps", d2,
                 d2.filter(F.col("event_ts").isNotNull()))
    d4 = dq.step("behaviors", "drop invalid user_id (must match ^U[0-9]+$)", d3,
                 d3.filter(F.col("user_id").rlike(r"^U[0-9]+$")))
    d5 = dq.step("behaviors", "remove duplicate (split, impression_id)", d4,
                 d4.dropDuplicates(["split", "impression_id_int"]))

    beh = (d5
           .withColumn("impression_id", F.col("impression_id_int"))
           .withColumn("impression_key", F.concat_ws("-", "split", "impression_id"))
           .withColumn("history_arr",
                       F.when(F.trim(F.coalesce("history", F.lit(""))) == "",
                              F.array().cast("array<string>"))
                        .otherwise(F.split(F.trim("history"), r"\s+")))
           .withColumn("hist_len", F.size("history_arr"))
           .withColumn("event_date", F.to_date("event_ts"))
           .withColumn("event_hour", F.hour("event_ts"))
           .select("impression_key", "impression_id", "user_id", "event_ts", "event_date",
                   "event_hour", "history_arr", "hist_len", "impressions", "split",
                   "_source", "_ingest_ts", "_batch_id"))
    beh.write.mode("overwrite").partitionBy("split").parquet(os.path.join(SILVER, "behaviors"))
    beh = spark.read.parquet(os.path.join(SILVER, "behaviors"))

    # one row per (impression, candidate article) with the click label
    tok = (beh.select("impression_key", "impression_id", "user_id", "event_ts", "event_date",
                      "event_hour", "hist_len", "split",
                      F.posexplode(F.split(F.trim("impressions"), r"\s+")).alias("position", "token")))
    d6 = dq.step("impressions", "drop malformed tokens (expected N123-0 or N123-1)", tok,
                 tok.filter(F.col("token").rlike(r"^N[0-9]+-[01]$")))
    imps = (d6.withColumn("news_id", F.split("token", "-").getItem(0))
              .withColumn("clicked", F.split("token", "-").getItem(1).cast("int"))
              .drop("token"))
    imps = dq.step("impressions", "remove duplicate (impression, news_id)", imps,
                   imps.dropDuplicates(["impression_key", "news_id"]))
    imps.write.mode("overwrite").partitionBy("split").parquet(os.path.join(SILVER, "impressions"))

    hist = (beh.select("user_id", "split", F.posexplode("history_arr").alias("hist_pos", "news_id")))
    hist = dq.step("user_history", "remove duplicate (split, user, news) history rows", hist,
                   hist.dropDuplicates(["split", "user_id", "news_id"]))
    hist.write.mode("overwrite").partitionBy("split").parquet(os.path.join(SILVER, "user_history"))
    df.unpersist()


def clean_guardian(spark, dq):
    path = os.path.join(BRONZE, "guardian_articles")
    if not os.path.exists(path):
        print("  (no Guardian bronze table - skipping)")
        return None
    df = spark.read.parquet(path)
    mapping = F.create_map(*[F.lit(x) for kv in GUARDIAN_TO_MIND.items() for x in kv])

    d1 = dq.step("guardian", "keep only type = article with a title", df,
                 df.filter((F.col("type") == "article") & F.col("webTitle").isNotNull()
                           & (F.trim("webTitle") != "")))
    d2 = d1.withColumn("published_ts", F.to_timestamp("webPublicationDate"))
    d2 = dq.step("guardian", "drop unparseable publication dates", d1, d2.filter(F.col("published_ts").isNotNull()))
    w = Window.partitionBy("id").orderBy(F.col("_api_run").desc())
    d3 = dq.step("guardian", "remove duplicate article id across API runs", d2,
                 d2.withColumn("_rn", F.row_number().over(w)).filter("_rn = 1").drop("_rn"))

    trail = "field_trailText" if "field_trailText" in d3.columns else None
    words = "field_wordcount" if "field_wordcount" in d3.columns else None
    out = (d3
           .withColumn("news_id", F.concat(F.lit("G:"), F.col("id")))
           .withColumn("subcategory", F.lower(F.trim("sectionId")))
           .withColumn("category", F.coalesce(mapping[F.col("subcategory")], F.lit(DEFAULT_CATEGORY)))
           .withColumn("title", F.trim("webTitle"))
           .withColumn("abstract",
                       F.trim(F.regexp_replace(F.coalesce(F.col(trail) if trail else F.lit(""), F.lit("")),
                                               "<[^>]+>", "")))
           .withColumn("wordcount", (F.col(words) if words else F.lit(None)).cast("int"))
           .withColumn("title_len", F.size(F.split("title", r"\s+")))
           .withColumn("abstract_len",
                       F.when(F.col("abstract") == "", 0).otherwise(F.size(F.split("abstract", r"\s+"))))
           .select("news_id", "category", "subcategory", "title", "abstract",
                   F.col("webUrl").alias("url"), "published_ts", "wordcount", "title_len",
                   "abstract_len", F.col("sectionName").alias("section_name"),
                   "_source", "_source_file", "_api_run", "_ingest_ts"))
    out.write.mode("overwrite").parquet(os.path.join(SILVER, "guardian_articles"))
    return True


def build_catalogue(spark, has_guardian):
    news = spark.read.parquet(os.path.join(SILVER, "news")).select(
        "news_id", "category", "subcategory", "title", "abstract", "url",
        F.lit(None).cast("timestamp").alias("published_ts"), "title_len", "abstract_len",
        F.lit("MIND").alias("source"))
    cat = news
    if has_guardian:
        g = spark.read.parquet(os.path.join(SILVER, "guardian_articles")).select(
            "news_id", "category", "subcategory", "title", "abstract", "url", "published_ts",
            "title_len", "abstract_len", F.lit("Guardian").alias("source"))
        cat = news.unionByName(g)
    cat.write.mode("overwrite").parquet(os.path.join(SILVER, "articles"))
    print("\nUnified article catalogue (silver/articles) by source:")
    spark.read.parquet(os.path.join(SILVER, "articles")).groupBy("source").count().show()


def run(spark=None):
    spark = spark or get_spark("Step4-Silver")
    banner("STEP 4: SILVER - cleaning, validation, type fixing, de-duplication")
    dq = DQ()
    clean_news(spark, dq)
    clean_behaviors(spark, dq)
    has_g = clean_guardian(spark, dq)
    build_catalogue(spark, has_g)

    report = spark.createDataFrame(dq.rows, "table string, rule string, rows_before long, rows_after long, rows_removed long")
    report.write.mode("overwrite").parquet(os.path.join(SILVER, "_dq_report"))
    print("\nDATA QUALITY REPORT")
    report.show(50, truncate=False)

    print("Null check after cleaning (silver/news):")
    news = spark.read.parquet(os.path.join(SILVER, "news"))
    news.select([F.sum(F.col(c).isNull().cast("int")).alias(c)
                 for c in ["news_id", "category", "title", "abstract"]]).show()
    print("Typed schema (silver/impressions):")
    spark.read.parquet(os.path.join(SILVER, "impressions")).printSchema()


if __name__ == "__main__":
    run()
