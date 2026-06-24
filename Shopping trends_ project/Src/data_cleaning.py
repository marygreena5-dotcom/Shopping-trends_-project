import pandas as pd

df = pd.read_csv(r"D:\mini project\csv\shopping_trends.csv")

print("Original Shape:", df.shape)

df.drop_duplicates(inplace=True)

df.columns = df.columns.str.strip()

df.to_csv("cleaned_shopping_trends.csv", index=False)

print("Cleaned dataset saved successfully!")