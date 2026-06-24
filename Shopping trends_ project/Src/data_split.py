# Load featured dataset
import pandas as pd
import joblib 
from sklearn.model_selection import train_test_split
df = pd.read_csv("featured_shopping_trends.csv")

# Features
X = df[[
    'Age',
    'Gender',
    'Item Purchased',
    'Category',
    'Location',
    'Season',
    'Review Rating',
    'Payment Method',
    'Previous Purchases',
    'Category_Avg_Purchase',
    'Location_Total_Revenue',
    'Item_Avg_Rating',
    'Payment_Method_Count'
]]

# Target
y = df['Subscription Status']

# One Hot Encoding
X = pd.get_dummies(X)

# Save feature names
joblib.dump(X.columns.tolist(), "feature_names.pkl")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Save split data
joblib.dump(X_train, "X_train.pkl")
joblib.dump(X_test, "X_test.pkl")
joblib.dump(y_train, "y_train.pkl")
joblib.dump(y_test, "y_test.pkl")

print("Data Split Completed!")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)