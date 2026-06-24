import pandas as pd

df = pd.read_csv("cleaned_shopping_trends.csv")

# Feature Engineering
df['Category_Avg_Purchase'] = (
    df.groupby('Category')['Purchase Amount (USD)']
    .transform('mean')
)

df['Location_Total_Revenue'] = (
    df.groupby('Location')['Purchase Amount (USD)']
    .transform('sum')
)

df['Item_Avg_Rating'] = (
    df.groupby('Item Purchased')['Review Rating']
    .transform('mean')
)

df['Payment_Method_Count'] = (
    df.groupby('Payment Method')['Payment Method']
    .transform('count')
)

# Drop unwanted columns
df = df.drop(columns=[
    'Customer ID',
    'Size',
    'Color',
    'Shipping Type',
    'Discount Applied',
    'Promo Code Used',
    'Preferred Payment Method',
    'Frequency of Purchases'
])

df.to_csv("featured_shopping_trends.csv", index=False)

print("Feature Engineering Completed!")
print("New Shape:", df.shape)