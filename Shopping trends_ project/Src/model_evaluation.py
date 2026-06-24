import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load model
model = joblib.load("ml_model.pkl")

# Load test data
X_test = joblib.load("X_test.pkl")
y_test = joblib.load("y_test.pkl")

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))