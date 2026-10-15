"""Run the whole pipeline locally in order (same order as the Airflow DAG)."""
import subprocess
import sys

STEPS = ["step02_ingest", "step03_bronze", "step04_silver", "step05_processing", "step06_gold",
         "step07_ml", "step08_predict", "step10_validate", "step09_export_dashboard"]
for s in STEPS:
    print(f"\n>>> running {s}")
    subprocess.run([sys.executable, f"pipeline/{s}.py"], check=True)
print("\nPipeline finished. Start the dashboard with:  streamlit run dashboard/app.py")
