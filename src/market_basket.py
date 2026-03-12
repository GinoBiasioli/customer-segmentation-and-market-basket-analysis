from itertools import combinations


def build_cluster_transactions(df_result, cluster_id):
    """
    Build a transaction list from a selected cluster.
    Each transaction corresponds to the list of products purchased by a user.
    """
    cluster_data = df_result[df_result["Cluster"] == cluster_id]
    transactions = cluster_data.groupby("User_ID")["Product_ID"].apply(list).tolist()
    return transactions


def apriori_with_confidence(transactions, min_support=0.06, min_confidence=0.5, top_n=5):
    """
    Find frequent itemsets and generate simple association rules.

    Parameters
    ----------
    transactions : list[list]
        List of baskets, where each basket is a list of product IDs.
    min_support : float
        Minimum support threshold.
    min_confidence : float
        Minimum confidence threshold.
    top_n : int
        Number of top rules to return, ranked by interest.

    Returns
    -------
    frequent_items : dict
        Frequent single items and their support values.
    all_frequent_itemsets : dict
        Frequent itemsets by size.
    rules : list[dict]
        Top association rules.
    """
    total_transactions = len(transactions)
    item_support = {}

    # Count item occurrences
    for transaction in transactions:
        for item in transaction:
            item_support[item] = item_support.get(item, 0) + 1

    # Keep only frequent single items
    frequent_items = {}
    for item, count in item_support.items():
        support = count / total_transactions
        if support >= min_support:
            frequent_items[item] = support

    def generate_frequent_itemsets(items, k):
        candidates = list(combinations(items, k))
        itemset_counts = {tuple(sorted(candidate)): 0 for candidate in candidates}

        for transaction in transactions:
            transaction_set = set(transaction)
            for candidate in itemset_counts:
                if set(candidate).issubset(transaction_set):
                    itemset_counts[candidate] += 1

        frequent_itemsets = {}
        for itemset, count in itemset_counts.items():
            support = count / total_transactions
            if support >= min_support:
                frequent_itemsets[itemset] = support

        return frequent_itemsets

    # Generate itemsets up to size 2
    all_frequent_itemsets = {}
    for k in range(2, 3):
        all_frequent_itemsets[k] = generate_frequent_itemsets(list(frequent_items.keys()), k)

    # Generate rules
    rules = []

    for k, itemsets in all_frequent_itemsets.items():
        for itemset, itemset_support in itemsets.items():
            for rhs_size in range(1, k):
                for rhs in combinations(itemset, rhs_size):
                    lhs = tuple(sorted(set(itemset) - set(rhs)))

                    support_lhs = sum(
                        1 for transaction in transactions if set(lhs).issubset(set(transaction))
                    ) / total_transactions

                    support_rhs = sum(
                        1 for transaction in transactions if set(rhs).issubset(set(transaction))
                    ) / total_transactions

                    confidence = itemset_support / support_lhs
                    interest = confidence - support_rhs

                    if confidence >= min_confidence:
                        rules.append({
                            "rule": f"{lhs} -> {rhs}",
                            "confidence": confidence,
                            "support_lhs": support_lhs,
                            "support_rhs": support_rhs,
                            "interest": interest,
                        })

    rules = sorted(rules, key=lambda x: x["interest"], reverse=True)[:top_n]

    return frequent_items, all_frequent_itemsets, rules


def print_rules(rules):
    """
    Print association rules in a readable format.
    """
    print("\nTop association rules:")
    for rule in rules:
        print(f"Rule: {rule['rule']}")
        print(f"  Confidence: {rule['confidence']:.2f}")
        print(f"  Support LHS: {rule['support_lhs']:.2f}")
        print(f"  Support RHS: {rule['support_rhs']:.2f}")
        print(f"  Interest: {rule['interest']:.2f}\n")
