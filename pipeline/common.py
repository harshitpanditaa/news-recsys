"""Shared settings for the Personalized News Recommendation pipeline.

Every step script imports this module so that paths and the Spark session are
configured in exactly one place.

Folder layout (all under DATA_DIR):
    landing/   raw files exactly as downloaded (zip, tsv, json)
    bronze/    raw data as Parquet + ingestion metadata columns
    silver/    cleaned, validated, typed data
    gold/      business/analysis-ready tables and ML features
    models/    saved Spark MLlib models
    output/    predictions, recommendations, metrics, validation report
"""
import json
import os
from datetime import datetime, timezone

PROJECT_DIR = os.environ.get(
    "NEWSREC_HOME", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
DATA_DIR = os.environ.get("NEWSREC_DATA", os.path.join(PROJECT_DIR, "data"))

LANDING = os.path.join(DATA_DIR, "landing")
BRONZE = os.path.join(DATA_DIR, "bronze")
SILVER = os.path.join(DATA_DIR, "silver")
GOLD = os.path.join(DATA_DIR, "gold")
MODELS = os.path.join(DATA_DIR, "models")
OUTPUT = os.path.join(DATA_DIR, "output")
DASHBOARD_DATA = os.environ.get(
    "NEWSREC_DASHBOARD_DATA", os.path.join(PROJECT_DIR, "dashboard", "data")
)

# MIND-small (Microsoft News Dataset). Free for research use under the
# Microsoft Research License Terms: https://msnews.github.io/
# The original Azure blob (mind201910small.blob.core.windows.net) has refused public
# downloads since 2024, so we use the mirrors that Microsoft Recommenders points to,
# tried in order.
MIND_MIRRORS = [
    "https://huggingface.co/datasets/Recommenders/MIND/resolve/main",
    "https://recodatasets.z20.web.core.windows.net/newsrec",
    "https://mind201910small.blob.core.windows.net/release",
]
MIND_SPLITS = {"train": "MINDsmall_train.zip", "dev": "MINDsmall_dev.zip"}

# The Guardian Open Platform (free, non-commercial developer key):
# https://open-platform.theguardian.com/access/
GUARDIAN_URL = "https://content.guardianapis.com/search"
GUARDIAN_API_KEY = os.environ.get("GUARDIAN_API_KEY", "test")
GUARDIAN_DAYS_BACK = int(os.environ.get("GUARDIAN_DAYS_BACK", "3"))
GUARDIAN_MAX_PAGES = int(os.environ.get("GUARDIAN_MAX_PAGES", "10"))

# Guardian section -> MIND category, so fresh articles share the category
# vocabulary that users' click histories are expressed in.
GUARDIAN_TO_MIND = {
    "sport": "sports", "football": "sports",
    "business": "finance", "money": "finance",
    "food": "foodanddrink",
    "lifeandstyle": "lifestyle", "fashion": "lifestyle",
    "travel": "travel",
    "film": "movies", "music": "music", "tv-and-radio": "tv",
    "culture": "entertainment", "books": "entertainment",
    "artanddesign": "entertainment", "stage": "entertainment",
    "games": "games",
    "wellness": "health", "healthcare-network": "health",
    "weather": "weather",
}
DEFAULT_CATEGORY = "news"

# How many most-active users get fresh-article recommendations.
TOP_USERS_FOR_FRESH_RECS = int(os.environ.get("TOP_USERS_FOR_FRESH_RECS", "5000"))
RANDOM_SEED = 42


def get_spark(app_name="NewsRecSys"):
    import sys
    from pyspark.sql import SparkSession

    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)  # workers use the same Python
    spark = (
        SparkSession.builder.appName(app_name)
        .master(os.environ.get("SPARK_MASTER", "local[*]"))
        .config("spark.driver.memory", os.environ.get("SPARK_DRIVER_MEMORY", "8g"))
        .config("spark.sql.shuffle.partitions", os.environ.get("SPARK_SHUFFLE_PARTITIONS", "16"))
        .config("spark.sql.sources.partitionOverwriteMode", "dynamic")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.legacy.timeParserPolicy", "CORRECTED")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def now_utc():
    return datetime.now(timezone.utc)


def write_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def banner(text):
    line = "=" * 78
    print(f"\n{line}\n{text}\n{line}")
