import pandas as pd

from config import (
    DATA_PATH,
    PLOTS_DIR,
    TABLES_DIR,
    RANDOM_STATE,
    K_RANGE_ELBOW,
    K_RANGE_METRICS,
    FINAL_K,
    SILHOUETTE_SAMPLE_FRACTION,
    TARGET_CLUSTER_FOR_RULES,
    MIN_SUPPORT,
    MIN_CONFIDENCE,
    TOP_N_RULES,
)

from data_prep import (
    load_data,
    inspect_data,
    cast_columns,
    print_unique_values,
    plot_categorical_distributions,
    prepare_features,
    get_clustering_data,
)

from clustering import (
    compute_elbow_sse,
    compute_cluster_metrics,
    plot_elbow,
    plot_cluster_metrics,
    custom_kmeans,
    cosine_distance,
    run_kmeans,
    summarize_clusters,
    compute_final_metrics,
)

from market_basket import (
    build_cluster_transactions,
    apriori_with_confidence,
    print_rules,
)


def main():
    # ---------------------------------------------------------------
    # 1. Load and inspect data
    # ---------------------------------------------------------------
    df = load_data(DATA_PATH)
    df = cast_columns(df)

    inspect_data(df)
    print_unique_values(df)

    plot_categorical_distributions(
        df,
        save_path=PLOTS_DIR / "categorical_distributions.png"
    )

    # ---------------------------------------------------------------
    # 2. Prepare features for clustering
    # ---------------------------------------------------------------
    df_prepared = prepare_features(df)
    data, data_df = get_clustering_data(df_prepared)

    # ---------------------------------------------------------------
    # 3. Evaluate number of clusters
    # ---------------------------------------------------------------
    k_elbow, sse = compute_elbow_sse(
        data=data,
        k_range=K_RANGE_ELBOW,
        random_state=RANDOM_STATE
    )

    plot_elbow(
        k_values=k_elbow,
        sse=sse,
        save_path=PLOTS_DIR / "elbow_method.png"
    )

    metrics_df = compute_cluster_metrics(
        data=data,
        k_range=K_RANGE_METRICS,
        sample_fraction=SILHOUETTE_SAMPLE_FRACTION,
        random_state=RANDOM_STATE
    )

    print("\nCluster selection metrics:")
    print(metrics_df)

    metrics_df.to_csv(TABLES_DIR / "cluster_selection_metrics.csv", index=False)

    plot_cluster_metrics(
        metrics_df=metrics_df,
        save_path=PLOTS_DIR / "cluster_selection_metrics.png"
    )

    # ---------------------------------------------------------------
    # 4. Optional: run custom K-Means for comparison
    # ---------------------------------------------------------------
    print("\nRunning custom K-Means...")
    _custom_labels, _custom_centroids = custom_kmeans(
        data=data,
        k=FINAL_K,
        distance_function=cosine_distance,
        max_iter=30,
        random_state=RANDOM_STATE
    )

    # ---------------------------------------------------------------
    # 5. Final clustering with scikit-learn
    # ---------------------------------------------------------------
    print("\nRunning scikit-learn K-Means...")
    labels, centroids, model = run_kmeans(
        data=data,
        k=FINAL_K,
        random_state=RANDOM_STATE
    )

    df_result, cluster_counts, cluster_profile = summarize_clusters(df_prepared, labels)

    print("\nCluster sizes:")
    print(cluster_counts)

    print("\nCluster profile:")
    print(cluster_profile)

    cluster_counts.to_csv(TABLES_DIR / "cluster_counts.csv", header=["count"])
    cluster_profile.to_csv(TABLES_DIR / "cluster_profile.csv")

    final_metrics = compute_final_metrics(
        data=data,
        labels=labels,
        sample_fraction=SILHOUETTE_SAMPLE_FRACTION,
        random_state=RANDOM_STATE
    )

    final_metrics_df = pd.DataFrame([final_metrics])
    final_metrics_df.to_csv(TABLES_DIR / "final_clustering_metrics.csv", index=False)

    print("\nFinal clustering metrics:")
    print(final_metrics_df)

    # ---------------------------------------------------------------
    # 6. Association rules on a selected cluster
    # ---------------------------------------------------------------
    transactions = build_cluster_transactions(
        df_result=df_result,
        cluster_id=TARGET_CLUSTER_FOR_RULES
    )

    frequent_items, frequent_itemsets, rules = apriori_with_confidence(
        transactions=transactions,
        min_support=MIN_SUPPORT,
        min_confidence=MIN_CONFIDENCE,
        top_n=TOP_N_RULES
    )

    print_rules(rules)

    if rules:
        pd.DataFrame(rules).to_csv(TABLES_DIR / "association_rules.csv", index=False)

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()

