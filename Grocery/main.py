import gradio as gr
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Load dataset
data = pd.read_csv("Datasets/groceries.csv")
transactions = data.apply(lambda x: x.dropna().tolist(), axis=1).tolist()

# Encode transactions
encoder = TransactionEncoder()
encoder_ary = encoder.fit(transactions).transform(transactions)
df = pd.DataFrame(encoder_ary, columns=encoder.columns_)

# Run Apriori
frequent_itemsets = apriori(df, min_support=0.01, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.03)

# Recommendation function
def recommend(product):
    prd = product.lower()
    prd_rules = rules[rules['antecedents'].apply(lambda x: prd in x)]
    if not prd_rules.empty:
        # Flatten consequents
        items = set()
        for cons in prd_rules['consequents']:
            items.update(cons)  # unpack frozenset
        
        # Pick top 10 recommendations
        top_items = list(items)[:10]
        
        # Format in bullet points
        formatted = "\n".join([f"• {item}" for item in top_items])
        return f"Top recommendations for '{prd}':\n\n{formatted}"
    else:
        return f"No recommendations found for '{prd}'"

# Gradio app
demo = gr.Interface(
    fn=recommend,
    inputs=gr.Textbox(label="Enter a product name"),
    outputs=gr.Textbox(label="Recommendations"),
    title="Market Basket Recommendation",
    description="Enter a product name to see frequently bought-together items."
)

if __name__ == "__main__":
    demo.launch()
