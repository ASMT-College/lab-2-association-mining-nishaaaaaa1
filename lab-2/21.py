import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Sample transactions dataset
data = {
    'TransactionID': [1, 2, 3, 4, 5],
    'Items': [
        ['Bread', 'Milk'],
        ['Bread', 'Diaper', 'Beer', 'Eggs'],
        ['Milk', 'Diaper', 'Beer', 'Coke'],
        ['Bread', 'Milk', 'Diaper', 'Beer'],
        ['Bread', 'Milk', 'Diaper', 'Coke']
    ]
}

# Load into DataFrame
df = pd.DataFrame(data)
print("Original Transactions:\n", df)

# Convert to one-hot encoded format
encoded_df = df['Items'].apply(lambda x: pd.Series(1, index=x)).fillna(0)
print("\nOne-Hot Encoded Data:\n", encoded_df)

# Apply Apriori (support ≥ 0.6)
frequent_itemsets = apriori(encoded_df, min_support=0.6, use_colnames=True)
print("\nFrequent Itemsets:\n", frequent_itemsets)

# Generate association rules (confidence ≥ 0.7)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
print("\nAssociation Rules:\n", rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

frequent_itemsets.to_csv('frequent_itemsets.csv', index=False)
rules.to_csv('association_rules.csv', index=False)
print("\nCSV files 'frequent_itemsets.csv' and 'association_rules.csv' created successfully!")
