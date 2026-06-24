import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load training and testing data
X_train = joblib.load("X_train.pkl")
X_test = joblib.load("X_test.pkl")

y_train = joblib.load("y_train.pkl")
y_test = joblib.load("y_test.pkl")

# Create Logistic Regression Model
model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced'
)

# Train Model
model.fit(X_train, y_train)

# Save Model
joblib.dump(model, "ml_model.pkl")

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("=" * 50)
print("MODEL EVALUATION")
print("=" * 50)

accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {accuracy:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nModel saved successfully as ml_model.pkl")