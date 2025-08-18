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

# All products for checkboxes
all_products = sorted(df.columns.tolist())

# Recommendation function
def recommend(selected_products):
    if not selected_products:
        return "⚠️ Please select at least one product."
    
    selected_set = set([p.lower() for p in selected_products])
    
    # Find rules where ANY of the selected products are in antecedents
    prd_rules = rules[rules['antecedents'].apply(lambda x: len(selected_set.intersection(x)) > 0)]
    
    if not prd_rules.empty:
        # Collect consequents with score
        recs = []
        for _, row in prd_rules.iterrows():
            for cons in row['consequents']:
                if cons not in selected_set:
                    recs.append((cons, row['lift'], row['support']))
        
        if not recs:
            return f"No new recommendations found for {', '.join(selected_products)}"
        
        # Sort by lift (stronger recommendations first)
        recs = sorted(recs, key=lambda x: (-x[1], -x[2]))
        
        # Take top 10 unique recommendations
        seen = set()
        top_items = []
        for item, lift, support in recs:
            if item not in seen:
                seen.add(item)
                top_items.append(f"• {item}")
            if len(top_items) >= 10:
                break
        
        return f"Since you selected {', '.join(selected_products)}, you may also like:\n\n" + "\n".join(top_items)
    
    else:
        return f"No recommendations found for {', '.join(selected_products)}"

# Gradio app
demo = gr.Interface(
    fn=recommend,
    inputs=gr.CheckboxGroup(choices=all_products, label="Select the products you are buying"),
    outputs=gr.Textbox(label="Recommendations"),
    title="Smart Market Basket Recommendation",
    description="Select multiple products to get recommendations for other frequently bought-together items."
)

if __name__ == "__main__":
    demo.launch()
