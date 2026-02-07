import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

def run_apriori_analysis(transactions, min_support=0.05, min_confidence=0.1, min_lift=1.0):
    """
    Runs Apriori algorithm and generates association rules.
    
    Args:
        transactions: List of lists (transaction data)
        min_support: Minimum support threshold
        min_confidence: Minimum confidence threshold
        min_lift: Minimum lift threshold (default 1.0)
        
    Returns:
        rules_df: DataFrame containing the association rules
    """
    if not transactions:
        return pd.DataFrame()

    # 1. One-hot encoding
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_onehot = pd.DataFrame(te_ary, columns=te.columns_)

    # 2. Frequent Itemsets
    try:
        frequent_itemsets = apriori(df_onehot, min_support=min_support, use_colnames=True)
    except Exception as e:
        print(f"Apriori failed: {e}")
        return pd.DataFrame()

    if frequent_itemsets.empty:
        return pd.DataFrame()

    # 3. Association Rules
    try:
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    except Exception as e:
        print(f"Association rules generation failed: {e}")
        return pd.DataFrame()

    if rules.empty:
        return pd.DataFrame()

    # Filter by Lift
    rules = rules[rules['lift'] >= min_lift]
    
    # Sort by Lift descending
    rules = rules.sort_values(by='lift', ascending=False)
    
    # Format antecedents and consequents as strings for display
    rules['antecedents_str'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
    rules['consequents_str'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
    
    return rules
