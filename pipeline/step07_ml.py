"""STEP 7 - Analytics / MLlib.

Three Spark MLlib models:
  1. Click-prediction CLASSIFIER (Logistic Regression vs Gradient-Boosted Trees)
     predicts P(user clicks article | impression). Ranking candidates by this
     probability = the personalised recommendation. Best model chosen on dev AUC.
  2. COLD-START content model (Logistic Regression, no popularity features)
     used to score brand-new Guardian articles that have no click history yet.
  3. User SEGMENTATION with K-Means CLUSTERING on each user's category-interest
     vector (k chosen by silhouette score).

Train = MIND train split; validation = MIND dev split (a later, unseen day).
Negatives in train are down-sampled 1:4 to keep training fast on Colab.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import GOLD, MODELS, OUTPUT, RANDOM_SEED, banner, get_spark, write_json  # noqa: E402

from pyspark.ml import Pipeline  # noqa: E402
from pyspark.ml.classification import GBTClassifier, LogisticRegression  # noqa: E402
from pyspark.ml.clustering import KMeans  # noqa: E402
from pyspark.ml.evaluation import BinaryClassificationEvaluator, ClusteringEvaluator  # noqa: E402
from pyspark.ml.feature import OneHotEncoder, StandardScaler, StringIndexer, VectorAssembler  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402

FULL_FEATURES = ["user_cat_affinity", "subcat_affinity", "is_top_category", "log_times_shown",
                 "log_hist_users", "category_ctr", "log_hist_len", "cold_user", "title_len",
                 "abstract_len", "event_hour", "is_weekend"]
COLD_FEATURES = ["user_cat_affinity", "subcat_affinity", "is_top_category", "category_ctr",
                 "log_hist_len", "cold_user", "title_len", "abstract_len", "event_hour", "is_weekend"]


def feature_stages(numeric):
    return [
        StringIndexer(inputCol="category", outputCol="category_idx", handleInvalid="keep"),
        OneHotEncoder(inputCol="category_idx", outputCol="category_ohe", handleInvalid="keep"),
        VectorAssembler(inputCols=numeric + ["category_ohe"], outputCol="raw_features"),
        StandardScaler(inputCol="raw_features", outputCol="features", withMean=False),
    ]


def run(spark=None):
    spark = spark or get_spark("Step7-MLlib")
    banner("STEP 7: MLlib - click classifier, cold-start model, user clustering")
    feats = spark.read.parquet(os.path.join(GOLD, "ml_features"))
    train_all = feats.filter(F.col("split") == "train")
    dev = feats.filter(F.col("split") == "dev").cache()

    pos = train_all.filter("label = 1")
    neg = train_all.filter("label = 0")
    n_pos, n_neg = pos.count(), neg.count()
    frac = min(1.0, 4.0 * n_pos / max(n_neg, 1))
    train = pos.unionByName(neg.sample(False, frac, seed=RANDOM_SEED)).cache()
    print(f"Train: {n_pos:,} clicks, {n_neg:,} non-clicks -> sampled {train.count():,} rows (1:4)")
    print(f"Dev (evaluation): {dev.count():,} rows")

    auc_eval = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction",
                                             metricName="areaUnderROC")
    pr_eval = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction",
                                            metricName="areaUnderPR")
    results = {}
    candidates = {
        "logistic_regression": LogisticRegression(labelCol="label", featuresCol="features",
                                                  maxIter=50, regParam=0.001),
        "gbt": GBTClassifier(labelCol="label", featuresCol="features", maxIter=30,
                             maxDepth=5, seed=RANDOM_SEED),
    }
    fitted = {}
    for name, clf in candidates.items():
        print(f"\nTraining {name} ...")
        model = Pipeline(stages=feature_stages(FULL_FEATURES) + [clf]).fit(train)
        pred = model.transform(dev)
        results[name] = {"dev_auc": auc_eval.evaluate(pred), "dev_auc_pr": pr_eval.evaluate(pred)}
        print(f"  {name}: dev ROC-AUC = {results[name]['dev_auc']:.4f} | PR-AUC = {results[name]['dev_auc_pr']:.4f}")
        fitted[name] = model

    best = max(results, key=lambda k: results[k]["dev_auc"])
    print(f"\nBest click model: {best}")
    fitted[best].write().overwrite().save(os.path.join(MODELS, "click_model"))

    lr = fitted["logistic_regression"].stages[-1]
    coef = sorted(zip(FULL_FEATURES, lr.coefficients.toArray()[:len(FULL_FEATURES)]),
                  key=lambda x: -abs(x[1]))
    print("Logistic-regression coefficients (standardised features):")
    for f_, c in coef:
        print(f"  {f_:<20} {c:+.4f}")
    importance = None
    if best == "gbt":
        imp = fitted["gbt"].stages[-1].featureImportances.toArray()[:len(FULL_FEATURES)]
        importance = sorted(zip(FULL_FEATURES, map(float, imp)), key=lambda x: -x[1])

    print("\nTraining cold-start content model (for brand-new Guardian articles) ...")
    cold = Pipeline(stages=feature_stages(COLD_FEATURES) + [
        LogisticRegression(labelCol="label", featuresCol="features", maxIter=50, regParam=0.001)]).fit(train)
    cold_auc = auc_eval.evaluate(cold.transform(dev))
    print(f"  cold-start model: dev ROC-AUC = {cold_auc:.4f}")
    cold.write().overwrite().save(os.path.join(MODELS, "cold_start_model"))

    print("\nK-Means user segmentation on category-interest vectors ...")
    matrix = spark.read.parquet(os.path.join(GOLD, "user_interest_matrix"))
    cats = [c for c in matrix.columns if c != "user_id"]
    vec = VectorAssembler(inputCols=cats, outputCol="features").transform(matrix).cache()
    sample = vec.sample(False, min(1.0, 20000 / max(vec.count(), 1)), seed=RANDOM_SEED)
    sil = {}
    for k in range(4, 9):
        km = KMeans(k=k, seed=RANDOM_SEED, featuresCol="features").fit(sample)
        sil[k] = ClusteringEvaluator(featuresCol="features").evaluate(km.transform(sample))
        print(f"  k={k}: silhouette={sil[k]:.4f}")
    best_k = max(sil, key=sil.get)
    km = KMeans(k=best_k, seed=RANDOM_SEED, featuresCol="features").fit(vec)
    km.write().overwrite().save(os.path.join(MODELS, "user_kmeans"))
    seg = km.transform(vec).select("user_id", F.col("prediction").alias("segment"), *cats)

    # Name each segment after its two strongest interests (from the cluster centres)
    names = {}
    for i, centre in enumerate(km.clusterCenters()):
        top = sorted(zip(cats, centre), key=lambda x: -x[1])[:2]
        names[i] = " + ".join(f"{c} ({v:.0%})" for c, v in top)
    name_map = F.create_map(*[F.lit(x) for kv in names.items() for x in kv])
    seg = seg.withColumn("segment_name", name_map[F.col("segment")])
    seg.select("user_id", "segment", "segment_name").write.mode("overwrite").parquet(
        os.path.join(GOLD, "user_segments"))
    print(f"Chosen k={best_k}. Segment sizes:")
    spark.read.parquet(os.path.join(GOLD, "user_segments")).groupBy("segment", "segment_name") \
         .count().orderBy("segment").show(truncate=False)

    write_json({
        "click_models": results, "best_click_model": best, "train_rows_sampled": train.count(),
        "train_clicks": n_pos, "train_nonclicks": n_neg, "dev_rows": dev.count(),
        "logreg_coefficients": coef, "gbt_feature_importance": importance,
        "cold_start_model_dev_auc": cold_auc,
        "kmeans_silhouette_by_k": sil, "kmeans_best_k": best_k, "segment_names": names,
    }, os.path.join(OUTPUT, "training_metrics.json"))
    print(f"Saved models in {MODELS} and metrics in {OUTPUT}/training_metrics.json")


if __name__ == "__main__":
    run()
