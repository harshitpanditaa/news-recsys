"""Airflow DAG - orchestrates the whole Personalized News Recommendation pipeline.

ingest -> bronze -> silver -> processing -> gold -> train -> predict -> validate -> export_dashboard

Each task runs one pipeline script, so a failure shows exactly which stage broke,
and retries/alerts are handled by Airflow. Scheduled daily so new Guardian
articles are pulled and every user's fresh feed is refreshed each morning.

Environment variables (set before starting Airflow):
    NEWSREC_HOME    path to this project (folder containing pipeline/)
    NEWSREC_PYTHON  python interpreter that has PySpark installed (default: python3)
    GUARDIAN_API_KEY  your Guardian Open Platform key
"""
import os
from datetime import datetime, timedelta

from airflow import DAG

try:  # Airflow 3.x
    from airflow.providers.standard.operators.bash import BashOperator
except ImportError:  # Airflow 2.x
    from airflow.operators.bash import BashOperator

HOME = os.environ.get("NEWSREC_HOME", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PY = os.environ.get("NEWSREC_PYTHON", "python3")

STEPS = [
    ("ingest_mind_and_guardian", "step02_ingest.py"),
    ("build_bronze", "step03_bronze.py"),
    ("build_silver", "step04_silver.py"),
    ("pyspark_processing", "step05_processing.py"),
    ("build_gold", "step06_gold.py"),
    ("train_mllib_models", "step07_ml.py"),
    ("predict_and_evaluate", "step08_predict.py"),
    ("validate_pipeline", "step10_validate.py"),
    ("export_dashboard_data", "step09_export_dashboard.py"),
]

default_args = {
    "owner": "news-recsys",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="news_recsys_pipeline",
    description="MIND + Guardian API -> Bronze/Silver/Gold -> MLlib -> recommendations -> dashboard",
    default_args=default_args,
    start_date=datetime(2026, 10, 1),
    schedule="@daily",
    catchup=False,
    max_active_runs=1,
    tags=["big-data", "pyspark", "recommendation"],
) as dag:
    previous = None
    for task_id, script in STEPS:
        task = BashOperator(
            task_id=task_id,
            bash_command=f'cd "{HOME}" && "{PY}" pipeline/{script}',
            env={**os.environ},
        )
        if previous is not None:
            previous >> task
        previous = task
